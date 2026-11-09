"""NIH MedQuAD Clinical Assistant Orchestration & Multi-Agent Hierarchy Package."""

from orchestrator.router import ModelRouter, RoutingDecision
from orchestrator.guardrails import AgentIdentity, GuardrailValidator
from orchestrator.hitl import HITLController, RequestInput
from orchestrator.subagents import (
    MedQuadResearcherSubagent,
    ClinicalSafetyReviewerSubagent,
    PhysicianEscalationSubagent,
)
from orchestrator.agent import RootOrchestrator, create_root_orchestrator

__all__ = [
    "ModelRouter",
    "RoutingDecision",
    "AgentIdentity",
    "GuardrailValidator",
    "HITLController",
    "RequestInput",
    "MedQuadResearcherSubagent",
    "ClinicalSafetyReviewerSubagent",
    "PhysicianEscalationSubagent",
    "RootOrchestrator",
    "create_root_orchestrator",
]
