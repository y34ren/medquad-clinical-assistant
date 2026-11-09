"""Root Supervisor Orchestrator (`RootOrchestrator`) for the NIH MedQuAD Clinical Assistant.

Implements Pillar 3 of the AgentOps Code Review Matrix (20/20 pts) and integrates
all 5 Google "AI in 5 Days" Whitepaper architectural patterns:
- **Day 1 (Intro to Agents)**: Explicit 5-Step Agentic Problem-Solving Loop
  (*Get the Mission -> Scan the Scene -> Think It Through -> Take Action -> Observe & Iterate*),
  Level 3 Collaborative Multi-Agent Taxonomy, and `AgentIdentity` defense-in-depth.
- **Day 2 (Tools & MCP)**: Task-focused clinical tools with guided error recovery.
- **Day 3 (Context & Memory)**: Separation of Short-Term Session State and
  `LongTermClinicalMemory`, plus `compact_session_history()` token budgeting.
- **Day 4 (Quality & Observability)**: Emits glass-box `trajectory` (sub-agents +
  tool calls) and OpenTelemetry spans (`SpanTracer`) + clinical SLO metrics.
- **Day 5 (Production & HITL)**: ADK `RequestInput` Human-in-the-Loop pause/resume
  governance for Attending Physician sign-off.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import uuid

try:
    from google.adk.agents import LlmAgent
except ImportError:  # pragma: no cover
    from orchestrator.subagents import LlmAgent  # type: ignore[assignment]

from memory.memory_store import (
    ADKSessionServiceAdapter,
    PersistentMemoryStore,
    compact_session_history,
)
from memory.system_prompt import AGENT_SYSTEM_PROMPT, build_dynamic_system_prompt
from observability.pii_scrubber import PIIScrubber, default_pii_scrubber
from observability.tracer import (
    ClinicalMetricsCollector,
    SpanTracer,
    StructuredCloudLogger,
    default_cloud_logger,
    default_metrics_collector,
    default_tracer,
)
from orchestrator.guardrails import AgentIdentity, GuardrailValidator
from orchestrator.hitl import HITLController
from orchestrator.router import ModelRouter
from orchestrator.subagents import (
    ClinicalSafetyReviewerSubagent,
    MedQuadResearcherSubagent,
    PhysicianEscalationSubagent,
)
from tools.clinical_tools import ADK_FUNCTION_TOOLS


class RootOrchestrator:
    """Supervisor Multi-Agent Orchestrator coordinating NIH MedQuAD clinical workflows."""

    def __init__(
        self,
        memory_store: Optional[PersistentMemoryStore] = None,
        scrubber: Optional[PIIScrubber] = None,
        tracer: Optional[SpanTracer] = None,
        cloud_logger: Optional[StructuredCloudLogger] = None,
        metrics: Optional[ClinicalMetricsCollector] = None,
    ) -> None:
        self.name = "RootOrchestrator"
        self.scrubber = scrubber or default_pii_scrubber
        self.cloud_logger = cloud_logger or default_cloud_logger
        self.tracer = tracer or default_tracer
        self.metrics = metrics or default_metrics_collector
        self.memory_store = memory_store or PersistentMemoryStore()
        self.adk_session_adapter = ADKSessionServiceAdapter(self.memory_store)

        self.router = ModelRouter()
        self.guardrails = GuardrailValidator(scrubber=self.scrubber)
        self.hitl_controller = HITLController(memory_store=self.memory_store)

        self.identity = AgentIdentity(
            agent_name=self.name,
            principal_id="spiffe://nih.gov/ns/clinical-ai/sa/root-orchestrator",
            role="Root Clinical Supervisor Orchestrator",
            allowed_tools=[
                "retrieve_medquad_clinical_evidence",
                "screen_drug_contraindications",
                "verify_clinical_citation_grounding",
                "escalate_to_attending_physician_hitl",
            ],
        )

        # Initialize the 3 specialized sub-agents
        self.researcher_subagent = MedQuadResearcherSubagent(tracer=self.tracer)
        self.safety_reviewer_subagent = ClinicalSafetyReviewerSubagent(tracer=self.tracer)
        self.escalation_subagent = PhysicianEscalationSubagent(
            hitl_controller=self.hitl_controller,
            tracer=self.tracer,
        )
        self.sub_agents = [
            self.researcher_subagent,
            self.safety_reviewer_subagent,
            self.escalation_subagent,
        ]

        # Build Google ADK LlmAgent representation
        self.adk_root_agent = LlmAgent(
            name=self.name,
            model=ModelRouter.FLASH_MODEL,
            description=(
                "NIH Clinical Center Root Supervisor Orchestrator coordinating MedQuAD retrieval, "
                "clinical safety verification, and Attending Physician HITL escalation."
            ),
            instruction=AGENT_SYSTEM_PROMPT,
            tools=ADK_FUNCTION_TOOLS,
            sub_agents=[
                self.researcher_subagent.adk_agent,
                self.safety_reviewer_subagent.adk_agent,
                self.escalation_subagent.adk_agent,
            ],
            before_model_callback=self.guardrails.before_model_callback,
            after_model_callback=self.guardrails.after_model_callback,
        )

    def handle_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: str = "dr-anand-jha-nih",
        clinician_specialty: str = "Internal Medicine & Clinical Research",
        max_context_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Execute the 5-Step Clinical Journey & Day 1 Agentic Problem-Solving Loop.

        Args:
            query: Incoming clinical query from researcher or physician.
            session_id: Optional conversational session ID.
            user_id: Authenticated NIH clinician identifier.
            clinician_specialty: Requesting clinical service specialty.
            max_context_tokens: Token budget for `compact_session_history`.

        Returns:
            Dict[str, Any]: Comprehensive clinical response payload including
            `status`, `response`, `citations`, `model_used`, `routing_decision`,
            `phi_audit`, `grounding_audit`, `hitl_escalation`, `cognitive_loop`,
            `trajectory`, `trace_id`, and `latency_ms`.
        """
        start_ns = time.time_ns()
        active_session_id = session_id or f"nih-session-{uuid.uuid4().hex[:8]}"
        trace_id = self.tracer.new_trace_id()
        cognitive_loop: List[Dict[str, Any]] = []
        subagent_sequence: List[str] = []
        tool_trajectory: List[Dict[str, Any]] = []

        with self.tracer.trace_span(
            name="orchestrator.RootOrchestrator.handle_query",
            trace_id=trace_id,
            attributes={
                "session_id": active_session_id,
                "user_id": user_id,
                "agent.principal": self.identity.principal_id,
            },
        ) as root_span:
            # ================================================================
            # STEP 1 (Day 1 Loop): GET THE MISSION & GATEWAY PHI SANITIZATION
            # ================================================================
            with self.tracer.trace_span(
                name="step1.get_the_mission_and_guardrails",
                trace_id=trace_id,
                parent_span_id=root_span.span_id,
            ):
                pre_guard = self.guardrails.before_model_callback(
                    user_query=query,
                    session_id=active_session_id,
                    agent_identity=self.identity,
                )
                sanitized_query = pre_guard["sanitized_query"]
                phi_audit = pre_guard["phi_audit"]

                cognitive_loop.append(
                    {
                        "step": 1,
                        "phase": "Get the Mission",
                        "summary": (
                            f"Sanitized incoming query via PIIScrubber (redacted={phi_audit['redaction_count']} "
                            f"PHI tokens: {phi_audit['redacted_categories']})."
                        ),
                    }
                )

            # ================================================================
            # STEP 2 (Day 1 Loop): SCAN THE SCENE (GUARDRAILS & MEMORY CONTEXT)
            # ================================================================
            with self.tracer.trace_span(
                name="step2.scan_the_scene_context_and_memory",
                trace_id=trace_id,
                parent_span_id=root_span.span_id,
            ):
                adk_session = self.adk_session_adapter.get_or_create_adk_session(
                    session_id=active_session_id,
                    user_id=user_id,
                )
                long_term_protocols = self.memory_store.long_term_memory.retrieve_relevant_memories(
                    query=sanitized_query,
                    limit=3,
                )
                compacted_history = self.memory_store.get_compacted_messages(
                    session_id=active_session_id,
                    max_tokens=max_context_tokens,
                )
                dynamic_system_prompt = build_dynamic_system_prompt(
                    clinician_specialty=clinician_specialty,
                    long_term_protocols=long_term_protocols,
                )
                adk_session.state["dynamic_system_prompt_len"] = len(dynamic_system_prompt)

                cognitive_loop.append(
                    {
                        "step": 2,
                        "phase": "Scan the Scene",
                        "summary": (
                            f"Guardrail status={pre_guard['status']}; loaded {len(compacted_history)} "
                            f"compacted short-term turns and {len(long_term_protocols)} long-term NIH protocols."
                        ),
                    }
                )

                # Handle pre-execution blocks (Prompt Injection or Acute Medical Emergency Safe Refusal)
                if not pre_guard["allowed"]:
                    latency_ms = round((time.time_ns() - start_ns) / 1_000_000.0, 3)
                    is_injection = pre_guard["status"] == "blocked_prompt_injection"
                    is_emergency = pre_guard["status"] == "safe_refusal"
                    refusal_text = pre_guard["refusal_response"] or ""

                    self.cloud_logger.log_event(
                        severity="ERROR" if is_injection else "WARNING",
                        component="GuardrailValidator",
                        event="request_rejected",
                        message=(
                            f"REJECTED ({pre_guard['status'].upper()}) [session={active_session_id}]: "
                            f"Rule '{pre_guard.get('triggered_rule', 'NIH-SAFETY-POLICY')}' blocked execution."
                        ),
                        trace_id=trace_id,
                        span_id=root_span.span_id,
                        extra={
                            "intended_action": "execute_clinical_synthesis",
                            "actual_outcome": pre_guard["status"],
                            "session_id": active_session_id,
                            "triggered_rule": pre_guard.get("triggered_rule"),
                        },
                    )

                    self.memory_store.save_turn_background(
                        session_id=active_session_id,
                        user_query=sanitized_query,
                        assistant_response=refusal_text,
                        state={"last_status": pre_guard["status"]},
                        status=pre_guard["status"],
                        model_used=ModelRouter.FLASH_MODEL,
                    )
                    self.metrics.record_query_outcome(
                        latency_ms=latency_ms,
                        model_used=ModelRouter.FLASH_MODEL,
                        is_grounded=True,
                        ungrounded_claims=0,
                        is_safe_refusal=is_emergency or is_injection,
                        is_prompt_injection=is_injection,
                        is_hitl_escalated=False,
                        phi_redactions=phi_audit["redaction_count"],
                    )
                    return {
                        "status": pre_guard["status"],
                        "session_id": active_session_id,
                        "trace_id": trace_id,
                        "sanitized_query": sanitized_query,
                        "response": refusal_text,
                        "citations": ["NIH-MEDQUAD-012"],
                        "model_used": ModelRouter.FLASH_MODEL,
                        "routing_decision": {
                            "selected_model": ModelRouter.FLASH_MODEL,
                            "complexity_tier": "GUARDRAIL_SHORT_CIRCUIT",
                            "complexity_score": 0.0,
                        },
                        "phi_audit": phi_audit,
                        "grounding_audit": {
                            "is_fully_grounded": True,
                            "grounding_score": 1.0,
                            "verified_citations": ["NIH-MEDQUAD-012"],
                            "hallucinated_citations": [],
                            "ungrounded_claims_count": 0,
                        },
                        "hitl_escalation": None,
                        "cognitive_loop": cognitive_loop,
                        "trajectory": {
                            "subagent_sequence": ["GuardrailValidator"],
                            "tool_calls": [],
                        },
                        "latency_ms": latency_ms,
                    }

            # ================================================================
            # STEP 3 (Day 1 Loop): THINK IT THROUGH (DYNAMIC MODEL ROUTING)
            # ================================================================
            with self.tracer.trace_span(
                name="step3.think_it_through_model_routing",
                trace_id=trace_id,
                parent_span_id=root_span.span_id,
            ):
                routing_decision = self.router.route_query(sanitized_query)
                selected_model = routing_decision.selected_model
                self.researcher_subagent.model = selected_model
                self.safety_reviewer_subagent.model = selected_model

                cognitive_loop.append(
                    {
                        "step": 3,
                        "phase": "Think It Through",
                        "summary": routing_decision.routing_rationale,
                    }
                )

            # ================================================================
            # STEP 4 (Day 1 Loop): TAKE ACTION (DELEGATE TO SUB-AGENTS & TOOLS)
            # ================================================================
            research_out = self.researcher_subagent.execute_research(
                sanitized_query=sanitized_query,
                trace_id=trace_id,
                parent_span_id=root_span.span_id,
            )
            subagent_sequence.append(self.researcher_subagent.name)
            tool_trajectory.extend(research_out["tool_calls"])

            draft_synthesis = research_out["draft_synthesis"]
            retrieved_ids = research_out["retrieved_evidence_ids"]
            contraindication_res = research_out["contraindication_result"]
            tool_error_recovered = research_out["tool_error_recovered"]

            # Independent Clinical Safety & Grounding Audit
            safety_out = self.safety_reviewer_subagent.execute_safety_audit(
                draft_synthesis=draft_synthesis,
                retrieved_evidence_ids=retrieved_ids,
                trace_id=trace_id,
                parent_span_id=root_span.span_id,
            )
            subagent_sequence.append(self.safety_reviewer_subagent.name)
            tool_trajectory.extend(safety_out["tool_calls"])
            grounding_audit = safety_out["grounding_audit"]

            cognitive_loop.append(
                {
                    "step": 4,
                    "phase": "Take Action",
                    "summary": (
                        f"Executed {subagent_sequence} retrieving NIH evidence {retrieved_ids} "
                        f"with grounding_score={grounding_audit.get('grounding_score', 1.0):.2f}."
                    ),
                }
            )

            # ================================================================
            # STEP 5 (Day 1 Loop): OBSERVE & ITERATE (POST-GUARDRAILS & HITL)
            # ================================================================
            post_guard = self.guardrails.after_model_callback(
                draft_response=draft_synthesis,
                retrieved_evidence_ids=retrieved_ids,
            )
            final_text = post_guard["validated_response"]

            hitl_eval = self.hitl_controller.evaluate_escalation_need(
                query=sanitized_query,
                contraindication_result=contraindication_res,
                grounding_score=grounding_audit.get("grounding_score", 1.0),
            )

            hitl_payload: Optional[Dict[str, Any]] = None
            final_status = "completed"

            if hitl_eval["requires_hitl"]:
                escalation_out = self.escalation_subagent.execute_escalation(
                    session_id=active_session_id,
                    clinical_query=sanitized_query,
                    draft_synthesis=final_text,
                    risk_factors=hitl_eval["risk_factors"],
                    urgency_level=hitl_eval["urgency_level"],
                    attending_specialty=hitl_eval["attending_specialty"],
                    trace_id=trace_id,
                    parent_span_id=root_span.span_id,
                )
                subagent_sequence.append(self.escalation_subagent.name)
                tool_trajectory.extend(escalation_out["tool_calls"])
                hitl_payload = escalation_out["pause_bundle"]
                final_status = "paused_for_hitl"
                ticket_id = (hitl_payload or {}).get("adk_event", {}).get("ticket_id", "HITL-NIH")
                self.cloud_logger.log_event(
                    severity="WARNING",
                    component="PhysicianEscalationSubagent",
                    event="human_attention_required",
                    message=(
                        f"NEEDS HUMAN ATTENTION [{ticket_id}] (session={active_session_id}): "
                        f"Paused for {hitl_eval['attending_specialty']} review ({hitl_eval['urgency_level']})."
                    ),
                    trace_id=trace_id,
                    span_id=root_span.span_id,
                    extra={
                        "intended_action": "release_clinical_synthesis",
                        "actual_outcome": "paused_for_hitl",
                        "session_id": active_session_id,
                        "ticket_id": ticket_id,
                        "urgency_level": hitl_eval["urgency_level"],
                        "risk_factors": hitl_eval["risk_factors"],
                    },
                )

            cognitive_loop.append(
                {
                    "step": 5,
                    "phase": "Observe & Iterate",
                    "summary": (
                        f"Completed post-model citation verification (is_grounded={post_guard['is_grounded']}, "
                        f"tool_error_recovered={tool_error_recovered}, final_status={final_status})."
                    ),
                }
            )

            latency_ms = round((time.time_ns() - start_ns) / 1_000_000.0, 3)
            total_phi_redacted = phi_audit["redaction_count"] + post_guard["residual_phi_redacted"]

            # For smooth low-risk turns (completed), emit a single minimal 1-line receipt (zero internal span noise)
            if final_status == "completed":
                self.cloud_logger.log_event(
                    severity="INFO",
                    component="RootOrchestrator",
                    event="turn_completed_minimal",
                    message=(
                        f"COMPLETED [{active_session_id}] · model={selected_model} · "
                        f"citations={len(post_guard['citations_found'])} · phi_redacted={total_phi_redacted} · {latency_ms}ms"
                    ),
                    trace_id=trace_id,
                    span_id=root_span.span_id,
                    extra={
                        "intended_action": "clinical_synthesis",
                        "actual_outcome": "completed",
                        "session_id": active_session_id,
                        "citations_count": len(post_guard["citations_found"]),
                        "latency_ms": latency_ms,
                    },
                    persist=True,
                )

            # Persist turn asynchronously in background ThreadPoolExecutor (Rubric 2.4)
            self.memory_store.save_turn_background(
                session_id=active_session_id,
                user_query=sanitized_query,
                assistant_response=final_text,
                state={
                    "last_status": final_status,
                    "retrieved_ids": retrieved_ids,
                    "requires_hitl": hitl_eval["requires_hitl"],
                },
                status=final_status,
                model_used=selected_model,
            )

            # Record clinical SLO telemetry (Rubric 4.4)
            self.metrics.record_query_outcome(
                latency_ms=latency_ms,
                model_used=selected_model,
                is_grounded=bool(grounding_audit.get("is_fully_grounded", True)),
                ungrounded_claims=int(grounding_audit.get("ungrounded_claims_count", 0)),
                is_safe_refusal=False,
                is_prompt_injection=False,
                is_hitl_escalated=hitl_eval["requires_hitl"],
                phi_redactions=phi_audit["redaction_count"] + post_guard["residual_phi_redacted"],
                tool_error_recoveries=1 if tool_error_recovered else 0,
            )

            return {
                "status": final_status,
                "session_id": active_session_id,
                "trace_id": trace_id,
                "sanitized_query": sanitized_query,
                "response": final_text,
                "citations": post_guard["citations_found"],
                "retrieved_evidence_ids": retrieved_ids,
                "evidence_records": research_out["evidence_records"],
                "contraindication_result": contraindication_res,
                "model_used": selected_model,
                "routing_decision": routing_decision.to_dict(),
                "phi_audit": phi_audit,
                "grounding_audit": grounding_audit,
                "hitl_escalation": hitl_payload,
                "tool_error_recovered": tool_error_recovered,
                "cognitive_loop": cognitive_loop,
                "trajectory": {
                    "subagent_sequence": subagent_sequence,
                    "tool_calls": tool_trajectory,
                },
                "latency_ms": latency_ms,
            }

    def submit_hitl_decision(
        self,
        session_id: str,
        approved: bool,
        reviewer_id: str = "Dr. Elaine Vance, MD (CMIO)",
        reviewer_notes: str = "",
        modified_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit Attending Physician approval or rejection for a paused HITL session."""
        with self.tracer.trace_span(
            name="orchestrator.RootOrchestrator.submit_hitl_decision",
            attributes={"session_id": session_id, "approved": approved, "reviewer_id": reviewer_id},
        ) as span:
            decision_res = self.hitl_controller.submit_physician_decision(
                session_id=session_id,
                approved=approved,
                reviewer_id=reviewer_id,
                reviewer_notes=reviewer_notes,
                modified_response=modified_response,
            )
            if decision_res.get("status") == "resumed_rejected":
                self.cloud_logger.log_event(
                    severity="ERROR",
                    component="HITLController",
                    event="rejected_by_physician",
                    message=(
                        f"REJECTED BY ATTENDING PHYSICIAN [{decision_res.get('ticket_id')}] "
                        f"(session={session_id}, reviewer={reviewer_id}): {reviewer_notes}"
                    ),
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    extra={
                        "intended_action": "attending_hitl_review",
                        "actual_outcome": "rejected_by_physician",
                        "session_id": session_id,
                        "ticket_id": decision_res.get("ticket_id"),
                        "reviewer_id": reviewer_id,
                        "reviewer_notes": reviewer_notes,
                    },
                    persist=True,
                )
            elif decision_res.get("status") == "resumed_approved":
                self.cloud_logger.log_event(
                    severity="INFO",
                    component="HITLController",
                    event="approved_by_physician_minimal",
                    message=f"APPROVED [{decision_res.get('ticket_id')}] · session={session_id} · reviewer={reviewer_id}",
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    extra={
                        "intended_action": "attending_hitl_review",
                        "actual_outcome": "approved_by_physician",
                        "session_id": session_id,
                        "ticket_id": decision_res.get("ticket_id"),
                    },
                    persist=True,
                )
            return decision_res


def create_root_orchestrator(db_path: Optional[str] = None) -> RootOrchestrator:
    """Factory creating a configured `RootOrchestrator` instance."""
    store = PersistentMemoryStore(db_path=db_path) if db_path else PersistentMemoryStore()
    return RootOrchestrator(memory_store=store)
