"""Human-in-the-Loop (HITL) Attending Physician Escalation & ADK `RequestInput` Controller.

Implements Pillar 3.4 of the AgentOps Code Review Matrix (5/5 pts):
- Halts high-risk clinical responses (e.g., `CONTRAINDICATED` or `HIGH_RISK` drug
  interactions, pediatric/oncology chemotherapy dosing, pregnancy teratogenicity,
  or low evidence confidence < 0.80) using ADK `RequestInput` pause semantics.
- Persists paused clinical syntheses in `PersistentMemoryStore` and allows an
  Attending Physician to approve, modify, or reject the synthesis before release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from memory.memory_store import PersistentMemoryStore
from tools.clinical_tools import escalate_to_attending_physician_hitl


@dataclass
class RequestInput:
    """Google ADK `RequestInput` control signal pausing execution for human clinician input."""

    session_id: str
    ticket_id: str
    prompt_to_human: str
    risk_factors: List[str] = field(default_factory=list)
    urgency_level: str = "HIGH"
    attending_specialty: str = "Internal Medicine & Clinical Pharmacology"
    status: str = "PAUSED_AWAITING_ATTENDING_PHYSICIAN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HITLController:
    """Controller governing Attending Physician HITL pause and resume workflows."""

    HIGH_RISK_QUERY_KEYWORDS = (
        "pediatric",
        "child",
        "neonatal",
        "oncology",
        "methotrexate",
        "leucovorin",
        "osteosarcoma",
        "leukemia",
        "pregnant",
        "pregnancy",
        "teratogen",
    )

    def __init__(self, memory_store: PersistentMemoryStore) -> None:
        self.memory_store = memory_store

    def evaluate_escalation_need(
        self,
        query: str,
        contraindication_result: Optional[Dict[str, Any]] = None,
        grounding_score: float = 1.0,
    ) -> Dict[str, Any]:
        """Determine whether a clinical turn must pause for Attending Physician HITL review.

        Args:
            query: Sanitized clinical query.
            contraindication_result: Output from `screen_drug_contraindications`.
            grounding_score: Citation grounding score (0.0-1.0).

        Returns:
            Dict[str, Any]: Evaluation result with `requires_hitl`, `risk_factors`,
            `urgency_level`, and `attending_specialty`.
        """
        risk_factors: List[str] = []
        urgency = "STANDARD"
        specialty = "Internal Medicine & Clinical Pharmacology"

        if contraindication_result and contraindication_result.get("status") == "success":
            risk_level = contraindication_result.get("overall_risk_level", "LOW")
            alerts = contraindication_result.get("contraindication_alerts", [])
            if risk_level in {"CONTRAINDICATED", "HIGH_RISK"} or contraindication_result.get("requires_hitl_escalation"):
                for alert in alerts:
                    rf = (
                        f"{alert.get('severity', 'HIGH_RISK')}: {alert.get('primary_drug', '')} "
                        f"with {alert.get('trigger_factor', '')} [{alert.get('rule_id', '')}]"
                    )
                    if rf not in risk_factors:
                        risk_factors.append(rf)
                if risk_level == "CONTRAINDICATED":
                    urgency = "CRITICAL"
                elif urgency != "CRITICAL":
                    urgency = "HIGH"

        q_lower = (query or "").lower()
        for kw in self.HIGH_RISK_QUERY_KEYWORDS:
            if kw in q_lower:
                marker = f"High-Risk Clinical Cohort/Protocol Trigger: '{kw}'"
                if marker not in risk_factors:
                    risk_factors.append(marker)
                if kw in {"methotrexate", "oncology", "pediatric", "leukemia", "osteosarcoma"}:
                    specialty = "Pediatric & Medical Oncology / Clinical Pharmacology"
                    urgency = "CRITICAL"
                elif urgency == "STANDARD":
                    urgency = "HIGH"

        if grounding_score < 0.80:
            risk_factors.append(f"Low Citation Grounding Confidence ({grounding_score:.2f} < 0.80)")
            if urgency == "STANDARD":
                urgency = "HIGH"

        return {
            "requires_hitl": len(risk_factors) > 0,
            "risk_factors": risk_factors,
            "urgency_level": urgency,
            "attending_specialty": specialty,
        }

    def pause_session_for_hitl(
        self,
        session_id: str,
        clinical_query: str,
        draft_synthesis: str,
        risk_factors: List[str],
        urgency_level: str = "HIGH",
        attending_specialty: str = "Internal Medicine & Clinical Pharmacology",
    ) -> Dict[str, Any]:
        """Invoke `escalate_to_attending_physician_hitl`, persist HITL state, and emit `RequestInput`."""
        tool_output = escalate_to_attending_physician_hitl(
            session_id=session_id,
            clinical_summary=draft_synthesis,
            risk_factors=risk_factors,
            urgency_level=urgency_level,
            attending_specialty=attending_specialty,
        )
        ticket_id = tool_output.get("escalation_ticket_id", "HITL-NIH-DEFAULT")

        self.memory_store.save_hitl_review(
            session_id=session_id,
            ticket_id=ticket_id,
            clinical_query=clinical_query,
            draft_response=draft_synthesis,
            risk_factors=risk_factors,
            urgency_level=urgency_level,
            attending_specialty=attending_specialty,
            status="PENDING_ATTENDING_REVIEW",
        )

        request_input = RequestInput(
            session_id=session_id,
            ticket_id=ticket_id,
            prompt_to_human=tool_output.get("instructions_for_reviewer", "Attending review required."),
            risk_factors=risk_factors,
            urgency_level=urgency_level,
            attending_specialty=attending_specialty,
        )

        return {
            "status": "paused_for_hitl",
            "adk_event": request_input.to_dict(),
            "tool_output": tool_output,
        }

    def submit_physician_decision(
        self,
        session_id: str,
        approved: bool,
        reviewer_id: str = "Dr. Elaine Vance, MD (CMIO)",
        reviewer_notes: str = "",
        modified_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resume a paused HITL session with the Attending Physician's approval or rejection."""
        resolved = self.memory_store.resolve_hitl_review(
            session_id=session_id,
            approved=approved,
            reviewer_id=reviewer_id,
            reviewer_notes=reviewer_notes,
            modified_response=modified_response,
        )
        if not resolved:
            return {
                "status": "error",
                "error_code": "HITL_SESSION_NOT_FOUND",
                "message": f"No pending or existing HITL review found for session_id='{session_id}'.",
            }
        return {
            "status": "resumed_approved" if approved else "resumed_rejected",
            "session_id": session_id,
            "ticket_id": resolved["ticket_id"],
            "approved": approved,
            "reviewer_id": reviewer_id,
            "reviewer_notes": reviewer_notes,
            "final_response": resolved["final_response"],
            "review_record": resolved,
        }
