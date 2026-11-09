"""Unit Tests for FastAPI Web Gateway, Day 5 A2A Agent Card & Golden Evaluation Harness."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import app
from eval.eval_harness import run_evaluation_suite


class TestFastAPIAndA2ACard(unittest.TestCase):
    """Verify FastAPI REST endpoints, A2A Agent Card (`/.well-known/agent.json`), and Eval Harness."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_and_dashboard_routes(self) -> None:
        dash_res = self.client.get("/")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn("NIH MedQuAD Clinical Assistant", dash_res.text)

        health_res = self.client.get("/health")
        self.assertEqual(health_res.status_code, 200)
        body = health_res.json()
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["hipaa_phi_scrubber"], "active")

    def test_day5_a2a_agent_card_endpoint(self) -> None:
        res = self.client.get("/.well-known/agent.json")
        self.assertEqual(res.status_code, 200)
        card = res.json()
        self.assertEqual(card["name"], "nih-medquad-clinical-assistant")
        self.assertIn("Dr. Elaine Vance", str(card["provider"]["sponsors"]))
        self.assertTrue(card["capabilities"]["hitlEscalation"])
        self.assertTrue(card["capabilities"]["hipaaPhiScrubbing"])
        self.assertEqual(len(card["skills"]), 4)

    def test_query_hitl_traces_metrics_and_mcp_endpoints(self) -> None:
        # 1. Standard grounded query
        q_res = self.client.post(
            "/api/query",
            json={
                "query": "What are the clinical evidence guidelines and eGFR thresholds for Metformin in Type 2 Diabetes Mellitus?",
                "session_id": "api-test-sess-01",
            },
        )
        self.assertEqual(q_res.status_code, 200)
        q_data = q_res.json()
        self.assertEqual(q_data["status"], "completed")
        self.assertIn("NIH-MEDQUAD-001", q_data["citations"])

        # 2. High-risk query triggering HITL escalation
        hitl_q = self.client.post(
            "/api/query",
            json={
                "query": "Synthesize anticoagulation management and screen contraindications for Warfarin and Ibuprofen in Atrial Fibrillation and CKD Stage 4.",
                "session_id": "api-test-hitl-02",
            },
        )
        self.assertEqual(hitl_q.status_code, 200)
        self.assertEqual(hitl_q.json()["status"], "paused_for_hitl")

        # 3. Check pending HITL reviews
        pending_res = self.client.get("/api/pending-reviews")
        self.assertEqual(pending_res.status_code, 200)
        self.assertGreaterEqual(pending_res.json()["pending_count"], 1)

        # 4. Submit Attending Physician approval
        approve_res = self.client.post(
            "/api/hitl/api-test-hitl-02",
            json={
                "approved": True,
                "reviewer_id": "Dr. Elaine Vance, MD (CMIO)",
                "reviewer_notes": "Approved via REST API test.",
            },
        )
        self.assertEqual(approve_res.status_code, 200)
        self.assertEqual(approve_res.json()["status"], "resumed_approved")

        # 5. Verify traces, metrics, and MCP tools endpoints
        traces_res = self.client.get("/api/traces")
        self.assertEqual(traces_res.status_code, 200)
        self.assertIn("spans", traces_res.json())

        metrics_res = self.client.get("/api/metrics")
        self.assertEqual(metrics_res.status_code, 200)
        self.assertGreaterEqual(metrics_res.json()["total_queries"], 2)

        mcp_res = self.client.get("/api/mcp/tools")
        self.assertEqual(mcp_res.status_code, 200)
        self.assertEqual(len(mcp_res.json()["tools"]), 4)

    def test_golden_evaluation_harness_achieves_95_of_95(self) -> None:
        report = run_evaluation_suite()
        self.assertTrue(report["overall_passed"])
        self.assertEqual(report["rubric_score"], "95/95")
        self.assertEqual(report["aggregate_slo_metrics"]["trajectory_accuracy"], 1.0)
        self.assertEqual(report["aggregate_slo_metrics"]["phi_redaction_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
