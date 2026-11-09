"""Unit Tests for Pillar 1: Clinical Tools, Pydantic Schemas, Guided Error Recovery & MCP Adapter."""

from __future__ import annotations

import unittest

from tools.clinical_tools import (
    CLINICAL_TOOLS,
    TOOL_JSON_SCHEMAS,
    MedQuadEvidenceInput,
    escalate_to_attending_physician_hitl,
    retrieve_medquad_clinical_evidence,
    screen_drug_contraindications,
    verify_clinical_citation_grounding,
)
from tools.mcp_adapter import MCPServerRegistry, get_default_mcp_registry


class TestClinicalToolsAndMCP(unittest.TestCase):
    """Verify Pillar 1 (20/20 pts) and Day 2 Whitepaper MCP tool interoperability."""

    def test_descriptive_naming_and_comprehensive_docstrings(self) -> None:
        expected_names = {
            "retrieve_medquad_clinical_evidence",
            "screen_drug_contraindications",
            "verify_clinical_citation_grounding",
            "escalate_to_attending_physician_hitl",
        }
        actual_names = {fn.__name__ for fn in CLINICAL_TOOLS}
        self.assertEqual(actual_names, expected_names)

        for fn in CLINICAL_TOOLS:
            doc = fn.__doc__ or ""
            self.assertIn("Args:", doc, f"Missing 'Args:' in docstring of {fn.__name__}")
            self.assertIn("Returns:", doc, f"Missing 'Returns:' in docstring of {fn.__name__}")
            self.assertGreater(len(doc), 250)

    def test_explicit_pydantic_json_schemas_exported(self) -> None:
        self.assertEqual(len(TOOL_JSON_SCHEMAS), 4)
        for tool_name, bundle in TOOL_JSON_SCHEMAS.items():
            self.assertIn("input_schema", bundle)
            self.assertIn("output_schema", bundle)
            self.assertEqual(bundle["input_schema"]["type"], "object")
            self.assertIn("properties", bundle["input_schema"])
            self.assertIn("required", bundle["input_schema"])

        schema = MedQuadEvidenceInput.model_json_schema()
        self.assertIn("condition_query", schema["properties"])

    def test_retrieve_medquad_clinical_evidence_success(self) -> None:
        res = retrieve_medquad_clinical_evidence("Type 2 Diabetes Mellitus and Metformin eGFR", max_results=2)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["total_matches"], 1)
        self.assertIn("NIH-MEDQUAD-001", res["retrieved_doc_ids"])
        first_doc = res["evidence_records"][0]
        self.assertTrue(first_doc["source_url"].startswith("https://"))

    def test_guided_error_handling_returns_recovery_suggestion(self) -> None:
        # 1. Too short / empty query
        err1 = retrieve_medquad_clinical_evidence("ab")
        self.assertEqual(err1["status"], "error")
        self.assertIn("recovery_suggestion", err1)
        self.assertIn("MeSH", err1["recovery_suggestion"])

        # 2. Non-existent medical token
        err2 = retrieve_medquad_clinical_evidence("invalid_nonexistent_xyz_code_000")
        self.assertEqual(err2["status"], "error")
        self.assertIn("recovery_suggestion", err2)

        # 3. Invalid drug contraindication input
        err3 = screen_drug_contraindications("", comorbid_conditions=["CKD Stage 4"], patient_age_group="invalid_cohort")
        self.assertEqual(err3["status"], "error")
        self.assertIn("recovery_suggestion", err3)

        # 4. Invalid citation grounding input
        err4 = verify_clinical_citation_grounding("short", retrieved_evidence_ids=[])
        self.assertEqual(err4["status"], "error")
        self.assertIn("recovery_suggestion", err4)

        # 5. Invalid HITL escalation input
        err5 = escalate_to_attending_physician_hitl("s", "short", risk_factors=[], urgency_level="INVALID")
        self.assertEqual(err5["status"], "error")
        self.assertIn("recovery_suggestion", err5)

    def test_screen_drug_contraindications_detects_severe_interactions(self) -> None:
        res = screen_drug_contraindications(
            primary_medication="warfarin",
            comorbid_conditions=["Atrial Fibrillation", "CKD Stage 4"],
            concomitant_medications=["ibuprofen"],
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["overall_risk_level"], "CONTRAINDICATED")
        self.assertTrue(res["requires_hitl_escalation"])
        rule_ids = [a["rule_id"] for a in res["contraindication_alerts"]]
        self.assertIn("NIH-DRUG-002", rule_ids)

    def test_verify_clinical_citation_grounding_detects_hallucinations(self) -> None:
        grounded_draft = (
            "Metformin is contraindicated in patients with eGFR below 30 mL/min/1.73m2 "
            "due to lactic acidosis risk [NIH-MEDQUAD-001]."
        )
        ok_res = verify_clinical_citation_grounding(grounded_draft, ["NIH-MEDQUAD-001"])
        self.assertTrue(ok_res["is_fully_grounded"])
        self.assertEqual(ok_res["grounding_score"], 1.0)

        hallucinated_draft = (
            "Metformin can be used at any eGFR without restriction [NIH-MEDQUAD-999]."
        )
        bad_res = verify_clinical_citation_grounding(hallucinated_draft, ["NIH-MEDQUAD-001"])
        self.assertFalse(bad_res["is_fully_grounded"])
        self.assertIn("NIH-MEDQUAD-999", bad_res["hallucinated_citations"])

    def test_mcp_adapter_descriptors_and_invocation(self) -> None:
        registry: MCPServerRegistry = get_default_mcp_registry()
        tools = registry.list_mcp_tools()
        self.assertEqual(len(tools), 4)
        for descriptor in tools:
            self.assertIn("name", descriptor)
            self.assertIn("inputSchema", descriptor)
            self.assertIn("annotations", descriptor)
            self.assertTrue(descriptor["annotations"]["clinicalGovernance"]["hipaaCompliant"])

        call_res = registry.call_mcp_tool(
            "retrieve_medquad_clinical_evidence",
            {"condition_query": "Marfan Syndrome", "max_results": 1},
        )
        self.assertFalse(call_res["isError"])
        self.assertEqual(call_res["structuredContent"]["status"], "success")

        unknown_res = registry.call_mcp_tool("non_existent_clinical_tool", {})
        self.assertTrue(unknown_res["isError"])


if __name__ == "__main__":
    unittest.main()
