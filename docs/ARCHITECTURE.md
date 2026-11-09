# NIH MedQuAD Clinical Assistant — System Architecture & Design Flowcharts

This document details the end-to-end architecture, multi-agent control flow, Human-in-the-Loop (HITL) clinical governance sequence, and Google Cloud infrastructure topology for the **NIH MedQuAD Clinical Assistant**.

---

## 1. End-to-End 5-Step Clinical Architecture & Multi-Agent Flow

```mermaid
flowchart TD
    Clinician["NIH Clinician / Researcher (Web UI or A2A Client)"]

    subgraph Gateway["Layer 1: Gateway Security & HIPAA Safe Harbor"]
        Scrubber["PIIScrubber (Cloud DLP + 11 HIPAA Safe Harbor Rules)"]
        PreGuard["GuardrailValidator (before_model_callback)"]
        Refusal["Deterministic Safe Refusal (911 Emergency / Anti-Injection)"]
    end

    subgraph Orchestration["Layer 2: ADK Supervisor, Routing & Context"]
        Root["RootOrchestrator (ADK LlmAgent — 5-Step Loop)"]
        Router["ModelRouter (gemini-2.5-flash vs gemini-2.5-pro)"]
        ShortMem["Short-Term Session Compaction (ADK EventsCompactionConfig)"]
        LongMem["LongTermClinicalMemory & SQLite Store (PHI-Scrubbed)"]
    end

    subgraph SubAgents["Layer 3: Specialized Clinical Sub-Agents & MCP Tools"]
        Researcher["MedQuadResearcherSubagent"]
        ToolMedQuad["Tool: retrieve_medquad_clinical_evidence()"]
        ToolContra["Tool: screen_drug_contraindications()"]
        Reviewer["ClinicalSafetyReviewerSubagent"]
        ToolVerify["Tool: verify_clinical_citation_grounding()"]
        Escalator["PhysicianEscalationSubagent"]
        ToolHITL["Tool: escalate_to_attending_physician_hitl()"]
    end

    subgraph Governance["Layer 4: Attending Physician HITL Queue"]
        Pause["ADK RequestInput Pause (PENDING_ATTENDING_REVIEW)"]
        Attending["Dr. Elaine Vance, MD (CMIO — Approve or Reject)"]
    end

    subgraph Delivery["Layer 5: Post-Guardrails & Tiered Observability"]
        PostGuard["GuardrailValidator (after_model_callback)"]
        OTEL["OpenTelemetry SpanTracer (Parent/Child Spans + p95 SLOs)"]
        AuditLog["Tiered Cloud Audit Log (Minimal Smooth Receipt vs Full Alert)"]
        FinalOut["Audited Clinical Response + Inline NIH Citations"]
    end

    Clinician -->|"POST /api/query"| Scrubber
    Scrubber -->|"Sanitized Query"| PreGuard
    PreGuard -->|"Acute Emergency or Jailbreak"| Refusal
    Refusal --> AuditLog
    PreGuard -->|"Allowed"| Root

    Root <--> ShortMem
    Root <--> LongMem
    Root --> Router
    Router --> Researcher

    Researcher --> ToolMedQuad
    ToolMedQuad -.->|"Guided Error Recovery Retry"| ToolMedQuad
    Researcher --> ToolContra
    Researcher --> Reviewer
    Reviewer --> ToolVerify

    ToolVerify -->|"Low Risk & 100% Grounded"| PostGuard
    ToolVerify -->|"CONTRAINDICATED / Oncology / Low Confidence"| Escalator
    Escalator --> ToolHITL
    ToolHITL --> Pause
    Pause --> Attending
    Attending -->|"Approve & Release"| PostGuard
    Attending -->|"Reject & Withhold"| AuditLog

    PostGuard --> OTEL
    PostGuard --> AuditLog
    PostGuard --> FinalOut
```

---

## 2. Attending Physician Human-in-the-Loop (HITL) Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as NIH Researcher
    participant API as FastAPI Gateway & PIIScrubber
    participant Orch as RootOrchestrator
    participant Res as MedQuadResearcherSubagent
    participant Rev as ClinicalSafetyReviewerSubagent
    participant Esc as PhysicianEscalationSubagent
    participant DB as SQLite Memory & HITL Queue
    actor CMIO as Attending Physician (CMIO)

    User->>API: POST /api/query (e.g. Ciprofloxacin in Myasthenia Gravis)
    API->>API: Scrub HIPAA Safe Harbor 18 PHI & run before_model_callback
    API->>Orch: Forward sanitized clinical query
    Orch->>Res: Route to gemini-2.5-pro (high clinical complexity)
    Res->>Res: retrieve_medquad_clinical_evidence() -> [NIH-MEDQUAD-003]
    Res->>Res: screen_drug_contraindications() -> CONTRAINDICATED [NIH-DRUG-003]
    Res-->>Orch: Draft synthesis + FDA Boxed Warning alert
    Orch->>Rev: verify_clinical_citation_grounding()
    Rev-->>Orch: 100% Grounded (Score 1.0), Contraindication Flagged
    Orch->>Esc: Trigger HITL escalation
    Esc->>DB: Save paused ticket (PENDING_ATTENDING_REVIEW) + emit WARNING log
    Orch-->>User: Return status="paused_for_hitl" (Ticket ID + Risk Factors)

    CMIO->>API: GET /api/pending-reviews (Views Queue Card in Dashboard)
    CMIO->>API: POST /api/hitl/{session_id} (Approve or Reject + Notes)
    API->>DB: Scrub notes, update status, log receipt/rejection
    API-->>CMIO: Return final released synthesis or withheld directive
```

---

## 3. Production Google Cloud & CI/CD Architecture

```mermaid
flowchart LR
    subgraph CICD["CI/CD & Quality Gates"]
        GH["GitHub Actions (.github/workflows/ci.yml)"]
        EvalGate["Unit Tests (21/21) + Golden Eval Harness (95/95)"]
        CB["Cloud Build (cloudbuild.yaml) + agents-cli"]
        TF["Terraform IaC (terraform/main.tf)"]
    end

    subgraph GCP["Google Cloud Production Environment"]
        CR["Cloud Run (FastAPI Dashboard + A2A /.well-known/agent.json)"]
        AR["Vertex AI Agent Runtime (ADK RootOrchestrator)"]
        VAIS["Vertex AI Search (NIH MedQuAD Corpus)"]
        DLP["Cloud Sensitive Data Protection (dlp_v2)"]
        SM["Secret Manager (Runtime Credentials)"]
        Obs["Cloud Trace (OpenTelemetry) + Cloud Logging"]
    end

    GH --> EvalGate
    EvalGate -->|"100% Pass"| CB
    TF --> GCP
    CB --> CR
    CB --> AR
    CR --> AR
    AR --> VAIS
    AR --> DLP
    AR --> SM
    AR --> Obs
```
