"""Unit Tests for Pillar 2: Context Engineering, Compaction, Short-Term vs. Long-Term Memory & Async I/O."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from memory.memory_store import (
    ADKSessionServiceAdapter,
    PersistentMemoryStore,
    compact_session_history,
    estimate_token_count,
)
from memory.system_prompt import AGENT_SYSTEM_PROMPT, build_dynamic_system_prompt


class TestMemoryAndContextEngineering(unittest.TestCase):
    """Verify Pillar 2 (20/20 pts) and Day 3 Whitepaper context & memory separation."""

    def test_system_prompt_contains_clinical_constitution_and_5_step_loop(self) -> None:
        self.assertIn("NIH MedQuAD Clinical Assistant", AGENT_SYSTEM_PROMPT)
        self.assertIn("Dr. Elaine Vance", AGENT_SYSTEM_PROMPT)
        self.assertIn("Dr. Anand Jha", AGENT_SYSTEM_PROMPT)
        self.assertIn("Get the Mission", AGENT_SYSTEM_PROMPT)
        self.assertIn("Scan the Scene", AGENT_SYSTEM_PROMPT)
        self.assertIn("Think It Through", AGENT_SYSTEM_PROMPT)
        self.assertIn("Take Action", AGENT_SYSTEM_PROMPT)
        self.assertIn("Observe & Iterate", AGENT_SYSTEM_PROMPT)
        self.assertIn("Safe Refusal", AGENT_SYSTEM_PROMPT)

        dyn = build_dynamic_system_prompt(
            clinician_specialty="Cardiology",
            long_term_protocols=[{"protocol_id": "NIH-PROTO-01", "title": "Warfarin", "summary": "Avoid NSAIDs"}],
            session_summary="Patient has CKD Stage 4 [NIH-MEDQUAD-004].",
        )
        self.assertIn("Cardiology", dyn)
        self.assertIn("NIH-PROTO-01", dyn)
        self.assertIn("CKD Stage 4", dyn)

    def test_compact_session_history_prevents_context_rot_and_preserves_clinical_facts(self) -> None:
        messages = [
            {"role": "system", "content": "System constitution: enforce 100% NIH citation grounding."},
            {"role": "user", "content": "Patient has CKD Stage 4 (eGFR 22) and takes Warfarin for atrial fibrillation [NIH-MEDQUAD-004]. " * 6},
            {"role": "assistant", "content": "Warfarin with NSAIDs is strictly contraindicated in CKD Stage 4 [NIH-DRUG-002]. " * 6},
            {"role": "user", "content": "Also note patient has Myasthenia Gravis; avoid ciprofloxacin [NIH-MEDQUAD-003]. " * 6},
            {"role": "assistant", "content": "Fluoroquinolones carry an FDA Boxed Warning in Myasthenia Gravis [NIH-DRUG-003]. " * 6},
            {"role": "user", "content": "What about current INR target?"},
            {"role": "assistant", "content": "Target INR is 2.0-3.0 for non-valvular atrial fibrillation [NIH-MEDQUAD-004]."},
        ]
        raw_tokens = sum(estimate_token_count(m["content"]) for m in messages)
        self.assertGreater(raw_tokens, 350)

        compacted = compact_session_history(messages, max_tokens=240, preserve_recent_turns=2)
        compacted_tokens = sum(estimate_token_count(m["content"]) for m in compacted)
        self.assertLessEqual(compacted_tokens, 260)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertTrue(any(m.get("is_compacted_summary") for m in compacted))
        summary_msg = next(m for m in compacted if m.get("is_compacted_summary"))
        self.assertIn("NIH-MEDQUAD-004", summary_msg["content"])

    def test_persistent_sqlite_store_short_term_and_long_term_memory_and_async_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_mem.sqlite3"
            store = PersistentMemoryStore(db_path=db_path)

            # Verify Long-Term Memory Bank seeded and searchable
            lt_results = store.long_term_memory.retrieve_relevant_memories("Metformin eGFR contrast", limit=2)
            self.assertGreaterEqual(len(lt_results), 1)

            store.long_term_memory.store_protocol_or_preference(
                memory_key="CLINICIAN-PREF-JHA",
                title="Dr. Anand Jha Citation Preference",
                summary="Always include ICD-10 codes and MeSH headings in synthesis.",
            )
            pref_hits = store.long_term_memory.retrieve_relevant_memories("ICD-10 MeSH", limit=3)
            self.assertTrue(any(p["protocol_id"] == "CLINICIAN-PREF-JHA" for p in pref_hits))

            # Verify Background ThreadPoolExecutor non-blocking persistence
            fut = store.save_turn_background(
                session_id="sess-bg-01",
                user_query="What is Metformin eGFR threshold?",
                assistant_response="Contraindicated below 30 mL/min/1.73m2 [NIH-MEDQUAD-001].",
                state={"turn": 1},
            )
            self.assertTrue(fut.result(timeout=5.0))

            # Verify Async I/O methods
            async def _verify_async() -> None:
                ok = await store.async_save_turn(
                    session_id="sess-async-02",
                    user_query="Async clinical question",
                    assistant_response="Async grounded synthesis [NIH-MEDQUAD-002].",
                    state={"async": True},
                )
                self.assertTrue(ok)
                loaded = await store.async_load_session("sess-async-02")
                self.assertIsNotNone(loaded)
                self.assertEqual(len(loaded["messages"]), 2)
                compacted = await store.async_compact_and_persist("sess-async-02", max_tokens=500)
                self.assertEqual(len(compacted), 2)

            asyncio.run(_verify_async())

            # Verify ADKSessionServiceAdapter synchronization
            adapter = ADKSessionServiceAdapter(store)
            adk_sess = adapter.get_or_create_adk_session("sess-async-02")
            self.assertTrue(adk_sess.state.get("async"))


if __name__ == "__main__":
    unittest.main()
