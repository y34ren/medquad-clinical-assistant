"""Unit Tests for Pillar 4 & 5.2: HIPAA PHI Scrubbing, Distributed Tracing, Cloud Logging, SLOs & Secrets."""

from __future__ import annotations

import os
import unittest

from observability.pii_scrubber import PIIScrubber
from observability.tracer import ClinicalMetricsCollector, SpanTracer, StructuredCloudLogger
from secrets.secret_manager import SecretManagerClient


class TestObservabilityAndSecrets(unittest.TestCase):
    """Verify Pillar 4 (20/20 pts) and Pillar 5.2 (5/5 pts)."""

    def test_hipaa_phi_scrubber_redacts_all_identifier_categories(self) -> None:
        scrubber = PIIScrubber()
        raw_note = (
            "Patient Johnathan Smith (SSN 123-45-6789, DOB: 05/14/1968, MRN-8849201, "
            "email jsmith@example.com, phone (301) 555-0199) presented for follow-up."
        )
        audit = scrubber.scrub_text(raw_note)
        self.assertTrue(audit["contains_phi"])
        self.assertGreaterEqual(audit["redaction_count"], 6)
        scrubbed = audit["scrubbed_text"]
        self.assertNotIn("Johnathan Smith", scrubbed)
        self.assertNotIn("123-45-6789", scrubbed)
        self.assertNotIn("05/14/1968", scrubbed)
        self.assertNotIn("MRN-8849201", scrubbed)
        self.assertNotIn("jsmith@example.com", scrubbed)
        self.assertNotIn("(301) 555-0199", scrubbed)
        self.assertIn("[REDACTED-SSN]", scrubbed)
        self.assertIn("[REDACTED-MRN]", scrubbed)
        self.assertIn("[REDACTED-DOB]", scrubbed)
        self.assertIn("[REDACTED-EMAIL]", scrubbed)
        self.assertIn("[REDACTED-PHONE]", scrubbed)
        self.assertIn("[REDACTED-PATIENT-NAME]", scrubbed)

        # Verify extended HIPAA Safe Harbor 18 rules (Age > 89, Medicare/Health Plan ID, Address, Card, IP, spaced SSN)
        extended_note = (
            "92-year-old female at 742 Evergreen Terrace, Apt 4B (SSN: 987 65 4321, "
            "Medicare ID: 1EG4-TE5-MK72, Card 4532-1122-3344-5566, portal IP 203.0.113.42)."
        )
        ext_audit = scrubber.scrub_text(extended_note)
        ext_scrubbed = ext_audit["scrubbed_text"]
        self.assertIn("[REDACTED-AGE-90+]", ext_scrubbed)
        self.assertIn("[REDACTED-ADDRESS]", ext_scrubbed)
        self.assertIn("[REDACTED-SSN]", ext_scrubbed)
        self.assertIn("[REDACTED-HEALTH-PLAN-ID]", ext_scrubbed)
        self.assertIn("[REDACTED-ACCOUNT-NUMBER]", ext_scrubbed)
        self.assertIn("[REDACTED-IP]", ext_scrubbed)
        self.assertNotIn("742 Evergreen Terrace", ext_scrubbed)
        self.assertNotIn("987 65 4321", ext_scrubbed)
        self.assertNotIn("1EG4-TE5-MK72", ext_scrubbed)
        self.assertNotIn("4532-1122-3344-5566", ext_scrubbed)
        self.assertNotIn("203.0.113.42", ext_scrubbed)

    def test_distributed_span_tracer_and_structured_cloud_logger(self) -> None:
        scrubber = PIIScrubber()
        logger = StructuredCloudLogger(scrubber=scrubber)
        tracer = SpanTracer(scrubber=scrubber, cloud_logger=logger)

        with tracer.trace_span("parent_clinical_span", attributes={"patient": "Patient Jane Doe"}) as parent:
            with tracer.trace_span("child_tool_span", trace_id=parent.trace_id, parent_span_id=parent.span_id) as child:
                self.assertEqual(child.trace_id, parent.trace_id)
                self.assertEqual(child.parent_span_id, parent.span_id)

        spans = tracer.get_spans_by_trace(parent.trace_id)
        self.assertEqual(len(spans), 2)
        parent_rec = next(s for s in spans if s["name"] == "parent_clinical_span")
        self.assertNotIn("Jane Doe", str(parent_rec["attributes"]))

        logs = logger.get_recent_logs(limit=5)
        self.assertGreaterEqual(len(logs), 2)
        self.assertIn("logging.googleapis.com/trace", logs[0])
        self.assertIn("logging.googleapis.com/spanId", logs[0])
        self.assertIn("severity", logs[0])

    def test_clinical_metrics_collector_computes_slos_and_p95(self) -> None:
        collector = ClinicalMetricsCollector()
        collector.record_query_outcome(
            latency_ms=42.5,
            model_used="gemini-2.5-flash",
            is_grounded=True,
            phi_redactions=3,
        )
        collector.record_query_outcome(
            latency_ms=85.0,
            model_used="gemini-2.5-pro",
            is_grounded=True,
            is_hitl_escalated=True,
        )
        summary = collector.get_metrics_summary()
        self.assertEqual(summary["total_queries"], 2)
        self.assertEqual(summary["grounded_citations_rate"], 1.0)
        self.assertEqual(summary["phi_redactions_total"], 3)
        self.assertEqual(summary["hitl_escalations_count"], 1)
        self.assertTrue(summary["slo_targets"]["slo_compliant"])

    def test_secret_manager_client_env_fallback_caching_and_masking(self) -> None:
        os.environ["TEST_CLINICAL_SECRET_XYZ"] = "super-secret-nih-token-99"
        try:
            client = SecretManagerClient(enable_gcp_lookup=False)
            val = client.get_secret("TEST_CLINICAL_SECRET_XYZ")
            self.assertEqual(val, "super-secret-nih-token-99")
            masked = client.mask_secret(val)
            self.assertNotIn("super-secret", masked)
            self.assertIn("(masked)", masked)

            with self.assertRaises(KeyError):
                client.get_secret("NON_EXISTENT_SECRET_KEY_0000")
        finally:
            os.environ.pop("TEST_CLINICAL_SECRET_XYZ", None)


if __name__ == "__main__":
    unittest.main()
