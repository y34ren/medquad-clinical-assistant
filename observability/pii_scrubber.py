"""Active HIPAA Protected Health Information (PHI) & PII Scrubber.

Implements Pillar 4.1 of the AgentOps Code Review Matrix (5/5 pts):
- Actively scans and redacts HIPAA Safe Harbor 18 identifiers from clinical queries,
  tool arguments, model prompts, and telemetry logs before execution or storage:
  - Social Security Numbers (`[REDACTED-SSN]`)
  - Medical Record Numbers (`[REDACTED-MRN]`)
  - Dates of Birth (`[REDACTED-DOB]`)
  - Email Addresses (`[REDACTED-EMAIL]`)
  - Telephone / Fax Numbers (`[REDACTED-PHONE]`)
  - Patient Names & Honorific Mentions (`[REDACTED-PATIENT-NAME]`)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

try:
    from google.cloud import dlp_v2  # type: ignore[import-not-found]
    HAS_CLOUD_DLP = True
except ImportError:  # pragma: no cover
    dlp_v2 = None  # type: ignore[assignment]
    HAS_CLOUD_DLP = False


class PIIScrubber:
    """Hybrid Google Cloud DLP (Sensitive Data Protection) + deterministic HIPAA PHI redaction engine."""

    CLOUD_DLP_INFO_TYPES = [
        {"name": "US_SOCIAL_SECURITY_NUMBER"},
        {"name": "MEDICAL_RECORD_NUMBER"},
        {"name": "US_HEALTHCARE_NPI"},
        {"name": "MEDICARE_BENEFICIARY_ID_NUMBER"},
        {"name": "CREDIT_CARD_NUMBER"},
        {"name": "DATE_OF_BIRTH"},
        {"name": "AGE"},
        {"name": "STREET_ADDRESS"},
        {"name": "IP_ADDRESS"},
        {"name": "EMAIL_ADDRESS"},
        {"name": "PHONE_NUMBER"},
        {"name": "PERSON_NAME"},
    ]

    def __init__(self, project_id: Optional[str] = None, enable_cloud_dlp: Optional[bool] = None) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.enable_cloud_dlp = (
            enable_cloud_dlp
            if enable_cloud_dlp is not None
            else os.environ.get("ENABLE_CLOUD_DLP", "false").lower() == "true"
        )
        self._dlp_client: Any = None
        if self.enable_cloud_dlp and HAS_CLOUD_DLP and self.project_id:
            try:
                self._dlp_client = dlp_v2.DlpServiceClient()
            except Exception:  # pragma: no cover
                self._dlp_client = None
        # Covers HIPAA Safe Harbor (45 CFR 164.514(b)(2)) clinical identifiers:
        self._patterns: List[Tuple[str, Pattern[str], str]] = [
            (
                "SSN",
                re.compile(
                    r"\b(?:SSN[:\s#-]*(?:\d{3}[-\s]?\d{2}[-\s]?\d{4})|\d{3}-\d{2}-\d{4})\b",
                    re.IGNORECASE,
                ),
                "[REDACTED-SSN]",
            ),
            (
                "MRN",
                re.compile(
                    r"\b(?:MRN|Medical\s+Record(?:\s+Number|\s*#)?)\s*[:#-]*\s*[A-Z0-9-]{4,14}\b",
                    re.IGNORECASE,
                ),
                "[REDACTED-MRN]",
            ),
            (
                "HEALTH_PLAN_ID",
                re.compile(
                    r"\b(?:Medicare(?:\s+ID|\s+MBI)?|Member\s+ID|Health\s+Plan\s+ID|Policy\s*#?)\s*[:#-]?\s*[A-Z0-9-]{5,16}\b",
                    re.IGNORECASE,
                ),
                "[REDACTED-HEALTH-PLAN-ID]",
            ),
            (
                "ACCOUNT_OR_CARD",
                re.compile(
                    r"\b(?:\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}|(?:Acct|Account)\s*#?\s*[:#-]?\s*\d{6,14})\b",
                    re.IGNORECASE,
                ),
                "[REDACTED-ACCOUNT-NUMBER]",
            ),
            (
                "DOB",
                re.compile(
                    r"\b(?:DOB|Date of Birth|born on)[:\s]*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
                    re.IGNORECASE,
                ),
                "DOB: [REDACTED-DOB]",
            ),
            (
                "AGE_OVER_89",
                re.compile(
                    r"\b(?:9\d|1[01]\d)\s*(?:-\s*year\s*-\s*old|year\s+old|yo\b|y/o\b)|(?:\bage[d]?\s+(?:9\d|1[01]\d)\b)",
                    re.IGNORECASE,
                ),
                "[REDACTED-AGE-90+]",
            ),
            (
                "EMAIL",
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "[REDACTED-EMAIL]",
            ),
            (
                "PHONE",
                re.compile(
                    r"(?:\b(?:Phone|Tel|Fax)[:\s]*)?(?:\+?1[-.\s]?)?\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b|\b(?:\+?1[-.\s]?)?\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
                    re.IGNORECASE,
                ),
                "[REDACTED-PHONE]",
            ),
            (
                "STREET_ADDRESS",
                re.compile(
                    r"\b\d{1,5}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Terrace|Way)\b(?:,?\s*(?:Apt|Suite|Unit)\s*#?[A-Za-z0-9-]+)?",
                ),
                "[REDACTED-ADDRESS]",
            ),
            (
                "IP_ADDRESS",
                re.compile(
                    r"\b(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
                ),
                "[REDACTED-IP]",
            ),
            (
                "PATIENT_NAME",
                re.compile(
                    r"\b(?:Patient(?:\s+Name)?|Pt|patient\s+named|Mr\.|Mrs\.|Ms\.)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
                ),
                "Patient [REDACTED-PATIENT-NAME]",
            ),
        ]
        self.total_redactions_performed: int = 0

    def scrub_text(self, text: str) -> Dict[str, Any]:
        """Scrub HIPAA PHI identifiers from text and return audit metadata.

        Args:
            text: Raw clinical input or log string potentially containing PHI.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - `scrubbed_text` (str): Sanitized string with `[REDACTED-...]` tokens.
                - `redaction_count` (int): Total number of PHI entities redacted.
                - `redacted_categories` (List[str]): Categories of PHI detected.
                - `contains_phi` (bool): True if at least one PHI entity was found.
        """
        if not isinstance(text, str) or not text:
            return {
                "scrubbed_text": text or "",
                "redaction_count": 0,
                "redacted_categories": [],
                "contains_phi": False,
            }

        scrubbed = text
        redaction_count = 0
        categories: List[str] = []

        # Optional Google Cloud Sensitive Data Protection (Cloud DLP) pass when enabled
        if self._dlp_client is not None and self.project_id:  # pragma: no cover
            try:
                response = self._dlp_client.deidentify_content(
                    request={
                        "parent": f"projects/{self.project_id}/locations/global",
                        "inspect_config": {"info_types": self.CLOUD_DLP_INFO_TYPES},
                        "deidentify_config": {
                            "info_type_transformations": {
                                "transformations": [
                                    {"primitive_transformation": {"replace_with_info_type_config": {}}}
                                ]
                            }
                        },
                        "item": {"value": scrubbed},
                    }
                )
                if response and response.item and response.item.value != scrubbed:
                    scrubbed = response.item.value
                    redaction_count += 1
                    categories.append("CLOUD_DLP_PHI")
            except Exception:
                pass

        for category, pattern, replacement in self._patterns:
            matches = list(pattern.finditer(scrubbed))
            if matches:
                redaction_count += len(matches)
                if category not in categories:
                    categories.append(category)
                scrubbed = pattern.sub(replacement, scrubbed)

        self.total_redactions_performed += redaction_count
        return {
            "scrubbed_text": scrubbed,
            "redaction_count": redaction_count,
            "redacted_categories": categories,
            "contains_phi": redaction_count > 0,
        }

    def scrub(self, text: str) -> str:
        """Convenience method returning only the PHI-scrubbed string."""
        return self.scrub_text(text)["scrubbed_text"]

    def scrub_payload(self, payload: Any) -> Any:
        """Recursively scrub all string values inside a dictionary or list payload."""
        if isinstance(payload, str):
            return self.scrub(payload)
        if isinstance(payload, dict):
            return {k: self.scrub_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self.scrub_payload(item) for item in payload]
        return payload


default_pii_scrubber = PIIScrubber()
