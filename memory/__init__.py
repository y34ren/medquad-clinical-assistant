"""NIH MedQuAD Clinical Assistant Context & Memory Engineering Package."""

from memory.system_prompt import (
    AGENT_SYSTEM_PROMPT,
    MEDQUAD_RESEARCHER_PROMPT,
    CLINICAL_SAFETY_REVIEWER_PROMPT,
    PHYSICIAN_ESCALATION_PROMPT,
    build_dynamic_system_prompt,
)
from memory.memory_store import (
    estimate_token_count,
    compact_session_history,
    LongTermClinicalMemory,
    PersistentMemoryStore,
    ADKSessionServiceAdapter,
)

__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "MEDQUAD_RESEARCHER_PROMPT",
    "CLINICAL_SAFETY_REVIEWER_PROMPT",
    "PHYSICIAN_ESCALATION_PROMPT",
    "build_dynamic_system_prompt",
    "estimate_token_count",
    "compact_session_history",
    "LongTermClinicalMemory",
    "PersistentMemoryStore",
    "ADKSessionServiceAdapter",
]
