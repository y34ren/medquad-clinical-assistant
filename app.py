"""FastAPI Web Application & A2A Protocol Gateway for the NIH MedQuAD Clinical Assistant.

Exposes:
- `GET /` : Interactive NIH Clinical Research & Attending Physician HITL Dashboard (`static/index.html`).
- `GET /health` : Service health, secret configuration status, and clinical SLO readiness.
- `GET /.well-known/agent.json` : Day 5 Whitepaper Agent2Agent (A2A) Protocol Agent Card
  for cross-agent discovery, authentication, and capability negotiation.
- `POST /api/query` : Executes the 5-Step Clinical Journey via `RootOrchestrator`.
- `GET /api/pending-reviews` : Lists paused high-risk cases awaiting Attending Physician HITL sign-off.
- `POST /api/hitl/{session_id}` : Submits Attending Physician approval/rejection to resume a paused session.
- `GET /api/traces` : Returns OpenTelemetry distributed spans and structured GCP Cloud Logging audit events.
- `GET /api/metrics` : Returns runtime clinical SLO metrics and p95 latency telemetry.
- `GET /api/mcp/tools` : Exposes Day 2 Whitepaper MCP (Model Context Protocol) tool descriptors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pydantic
from pydantic import BaseModel, Field

from orchestrator.agent import RootOrchestrator, create_root_orchestrator
from secrets.secret_manager import default_secret_client
from tools.mcp_adapter import get_default_mcp_registry


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="NIH MedQuAD Clinical Assistant & HITL Governance Gateway",
    description=(
        "Enterprise Healthcare & Life Sciences (HCLS) Multi-Agent Clinical Decision Support System "
        "built with Google ADK, implementing all 5 Google 'AI in 5 Days' Whitepaper architectures."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

orchestrator: RootOrchestrator = create_root_orchestrator()
mcp_registry = get_default_mcp_registry()


# ============================================================================
# Pydantic Request Schemas
# ============================================================================


class ClinicalQueryRequest(pydantic.BaseModel):
    """Request payload for `POST /api/query`."""

    query: str = Field(
        ...,
        min_length=2,
        max_length=4000,
        description="Clinical research question or case summary to synthesize against NIH MedQuAD.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional clinical session ID for multi-turn context continuity.",
    )
    user_id: str = Field(
        default="dr-anand-jha-nih",
        description="Authenticated NIH researcher or physician identifier.",
    )
    clinician_specialty: str = Field(
        default="Internal Medicine & Clinical Research",
        description="Clinical department or specialty of the requesting user.",
    )


class HITLDecisionRequest(pydantic.BaseModel):
    """Request payload for `POST /api/hitl/{session_id}` Attending Physician sign-off."""

    approved: bool = Field(
        ...,
        description="True if the Attending Physician approves release of the synthesis; False to reject.",
    )
    reviewer_id: str = Field(
        default="Dr. Elaine Vance, MD (CMIO)",
        description="Credentialed Attending Physician identifier.",
    )
    reviewer_notes: str = Field(
        default="Verified contraindication management and renal dosing protocols per NIH consensus.",
        description="Clinical sign-off notes or required modifications.",
    )
    modified_response: Optional[str] = Field(
        default=None,
        description="Optional physician-edited synthesis overriding the draft.",
    )


# ============================================================================
# Day 5 Whitepaper: Agent2Agent (A2A) Protocol Agent Card
# ============================================================================

A2A_AGENT_CARD: Dict[str, Any] = {
    "name": "nih-medquad-clinical-assistant",
    "displayName": "NIH MedQuAD Clinical Assistant (HCLS Multi-Agent System)",
    "description": (
        "NIH Clinical Center multi-agent biomedical literature synthesis and pharmacology "
        "contraindication assistant with active HIPAA PHI scrubbing, 100% citation grounding, "
        "deterministic Safe Refusal boundaries, and Attending Physician HITL governance."
    ),
    "version": "1.0.0",
    "protocolVersion": "0.2.1",
    "url": "https://medquad-clinical-assistant.nih-clinical-center.run.app",
    "documentationUrl": "https://clinicalcenter.nih.gov/policies/clinical-ai-safety-governance",
    "provider": {
        "organization": "National Institutes of Health (NIH) Clinical Center (Simulated)",
        "department": "Department of Clinical Research Informatics & Biomedical Safety",
        "sponsors": [
            "Dr. Elaine Vance, MD — Chief Medical Information Officer (CMIO)",
            "Dr. Anand Jha, MD, PhD — Lead Clinical Researcher",
        ],
    },
    "authentication": {
        "schemes": ["OAuth2", "Bearer", "SPIFFE-mTLS"],
        "principalId": "spiffe://nih.gov/ns/clinical-ai/sa/root-orchestrator",
    },
    "capabilities": {
        "streaming": False,
        "pushNotifications": True,
        "stateTransitionHistory": True,
        "hitlEscalation": True,
        "hipaaPhiScrubbing": True,
        "mcpInteroperability": True,
    },
    "defaultInputModes": ["application/json", "text/plain"],
    "defaultOutputModes": ["application/json", "text/markdown"],
    "skills": [
        {
            "id": "medquad_evidence_retrieval",
            "name": "Grounded NIH MedQuAD Literature Synthesis",
            "description": "Retrieves peer-reviewed NIH MedQuAD Q&A guidelines with MeSH/ICD-10 indexing and inline citations.",
            "tags": ["nih", "medquad", "rag", "clinical-evidence", "icd10", "mesh"],
            "examples": [
                "What are the NIH NIDDK renal eGFR thresholds for Metformin in Type 2 Diabetes?",
                "What are the Revised Ghent Nosology criteria for Marfan Syndrome?",
            ],
        },
        {
            "id": "drug_contraindication_screening",
            "name": "Multi-Morbidity & Drug-Drug Contraindication Screening",
            "description": "Screens medications against comorbid conditions and concurrent drugs for severe interactions.",
            "tags": ["pharmacology", "contraindications", "drug-interactions", "patient-safety"],
            "examples": [
                "Screen Warfarin and Ibuprofen in a patient with Atrial Fibrillation and CKD Stage 4.",
                "Are fluoroquinolones safe in a patient with Myasthenia Gravis?",
            ],
        },
        {
            "id": "clinical_citation_grounding_audit",
            "name": "Independent Citation Grounding & Zero-Hallucination Verification",
            "description": "Audits clinical syntheses to guarantee 100% of claims map to retrieved NIH document IDs.",
            "tags": ["grounding", "audit", "zero-hallucination", "quality-control"],
        },
        {
            "id": "attending_physician_hitl_escalation",
            "name": "Attending Physician Human-in-the-Loop (HITL) Escalation",
            "description": "Pauses high-risk clinical responses via ADK RequestInput until an Attending Physician approves.",
            "tags": ["hitl", "attending-physician", "governance", "adk-request-input"],
        },
    ],
}


# ============================================================================
# FastAPI Routes
# ============================================================================


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    """Serve the interactive NIH Clinical Research & Attending Physician HITL Dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(
        content="<h1>NIH MedQuAD Clinical Assistant</h1><p>Dashboard static file not found.</p>",
        status_code=200,
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Return service health, secret configuration summary, and SLO compliance status."""
    metrics = orchestrator.metrics.get_metrics_summary()
    return {
        "status": "healthy",
        "service": "medquad-clinical-assistant",
        "version": "1.0.0",
        "hipaa_phi_scrubber": "active",
        "adk_orchestrator": orchestrator.name,
        "sub_agents": [sa.name for sa in orchestrator.sub_agents],
        "secrets_config": default_secret_client.get_runtime_config_summary(),
        "slo_compliant": metrics["slo_targets"]["slo_compliant"],
    }


@app.get("/.well-known/agent.json")
def get_a2a_agent_card() -> Dict[str, Any]:
    """Return the Day 5 Whitepaper Agent2Agent (A2A) Protocol Agent Card."""
    return A2A_AGENT_CARD


@app.post("/api/query")
def query_clinical_assistant(request: ClinicalQueryRequest) -> Dict[str, Any]:
    """Execute the 5-step clinical journey through `RootOrchestrator`."""
    result = orchestrator.handle_query(
        query=request.query,
        session_id=request.session_id,
        user_id=request.user_id,
        clinician_specialty=request.clinician_specialty,
    )
    return result


@app.get("/api/pending-reviews")
def list_pending_hitl_reviews() -> Dict[str, Any]:
    """List all high-risk clinical syntheses paused awaiting Attending Physician HITL review."""
    reviews = orchestrator.memory_store.list_pending_hitl_reviews()
    return {
        "status": "success",
        "pending_count": len(reviews),
        "pending_reviews": reviews,
    }


@app.post("/api/hitl/{session_id}")
def submit_attending_hitl_decision(session_id: str, decision: HITLDecisionRequest) -> Dict[str, Any]:
    """Submit Attending Physician approval or rejection for a paused HITL session."""
    result = orchestrator.submit_hitl_decision(
        session_id=session_id,
        approved=decision.approved,
        reviewer_id=decision.reviewer_id,
        reviewer_notes=decision.reviewer_notes,
        modified_response=decision.modified_response,
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message", "HITL session not found"))
    return result


@app.get("/api/traces")
def get_distributed_traces(limit: int = 40, include_routine_logs: bool = False) -> Dict[str, Any]:
    """Return recent distributed OpenTelemetry spans and actionable (attention/rejection) audit logs."""
    cloud_logs = (
        orchestrator.cloud_logger.get_recent_logs(limit=limit)
        if include_routine_logs
        else orchestrator.cloud_logger.get_actionable_logs(limit=limit)
    )
    return {
        "status": "success",
        "spans": orchestrator.tracer.get_recent_spans(limit=limit),
        "cloud_logs": cloud_logs,
    }


@app.get("/api/logs")
def get_structured_audit_logs(
    limit: int = 50,
    event_filter: Optional[str] = None,
    include_routine: bool = False,
) -> Dict[str, Any]:
    """Return PHI-scrubbed structured JSON audit logs focused on Human Attention & Rejections."""
    logs = (
        orchestrator.cloud_logger.get_recent_logs(limit=limit)
        if include_routine
        else orchestrator.cloud_logger.get_actionable_logs(limit=limit)
    )
    if event_filter:
        logs = [lg for lg in logs if lg.get("event") == event_filter]
    return {
        "status": "success",
        "log_policy": "exception_and_escalation_only" if not include_routine else "verbose_all",
        "log_file": str(orchestrator.cloud_logger.log_file_path),
        "count": len(logs),
        "logs": logs,
    }


@app.get("/api/metrics")
def get_clinical_metrics() -> Dict[str, Any]:
    """Return runtime clinical SLO metrics and latency percentiles."""
    return orchestrator.metrics.get_metrics_summary()


@app.get("/api/mcp/tools")
def get_mcp_tools() -> Dict[str, Any]:
    """Return Day 2 Whitepaper MCP (Model Context Protocol) tool descriptors."""
    return mcp_registry.export_mcp_manifest()


if __name__ == "__main__":  # pragma: no cover
    import os

    port = int(os.environ.get("PORT", "8080"))
    try:
        import uvicorn  # type: ignore[import-not-found]
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
    except ImportError:
        print(f"FastAPI app initialized successfully. Install uvicorn to serve over HTTP on port {port}.")
