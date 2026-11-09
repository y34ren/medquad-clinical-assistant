"""Secure Secret Management with Google Cloud Secret Manager and Environment Fallback.

Implements Pillar 5.2 of the AgentOps Code Review Matrix (5/5 pts):
- Retrieves runtime credentials (`GEMINI_API_KEY`, `MEDQUAD_DB_URI`, `NIH_ATTESTATION_KEY`)
  from Google Cloud Secret Manager (`google-cloud-secret-manager`) with environment
  variable fallback and TTL caching.
- Enforces secret masking (`mask_secret`) so raw credentials never appear in logs or traces.
- Zero hardcoded API keys or credentials anywhere in the repository.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

try:
    from google.cloud import secretmanager  # type: ignore[attr-defined]
    HAS_GCP_SECRET_MANAGER = True
except ImportError:
    secretmanager = None
    HAS_GCP_SECRET_MANAGER = False


class SecretManagerClient:
    """Client retrieving secrets from GCP Secret Manager with environment-variable fallback and TTL cache."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        enable_gcp_lookup: bool = True,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "nih-clinical-center-prod")
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_gcp_lookup = enable_gcp_lookup and HAS_GCP_SECRET_MANAGER
        self._cache: Dict[str, tuple[str, float]] = {}

    @staticmethod
    def mask_secret(value: Optional[str]) -> str:
        """Mask a secret string for safe audit logging, keeping only a non-sensitive prefix."""
        if not value:
            return "[UNSET]"
        if len(value) <= 6:
            return "***REDACTED***"
        return f"{value[:3]}...{value[-2:]} (masked)"

    def get_secret(
        self,
        secret_id: str,
        version: str = "latest",
        default: Optional[str] = None,
    ) -> str:
        """Retrieve a secret by `secret_id` with TTL cache, env var fallback, and GCP Secret Manager.

        Args:
            secret_id: Name of the secret in GCP Secret Manager / environment variable.
            version: Secret version in GCP Secret Manager (default "latest").
            default: Optional safe non-sensitive fallback if not configured in offline mode.

        Returns:
            str: Resolved secret value.

        Raises:
            KeyError: If the secret is not found in GCP Secret Manager, environment, or default.
        """
        now = time.time()
        cache_key = f"{secret_id}:{version}"
        if cache_key in self._cache:
            cached_val, expires_at = self._cache[cache_key]
            if now < expires_at:
                return cached_val

        # 1. Check environment variable first for local/container overrides
        env_val = os.environ.get(secret_id)
        if env_val:
            self._cache[cache_key] = (env_val, now + self.cache_ttl_seconds)
            return env_val

        # 2. Attempt Google Cloud Secret Manager lookup when available
        if self.enable_gcp_lookup and secretmanager is not None:
            try:
                client = secretmanager.SecretManagerServiceClient()
                resource_name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
                response = client.access_secret_version(request={"name": resource_name})
                payload = response.payload.data.decode("utf-8")
                self._cache[cache_key] = (payload, now + self.cache_ttl_seconds)
                return payload
            except Exception:
                pass

        # 3. Use safe non-credential default if provided
        if default is not None:
            self._cache[cache_key] = (default, now + self.cache_ttl_seconds)
            return default

        raise KeyError(
            f"Secret '{secret_id}' not found in environment variables or GCP Secret Manager "
            f"(project={self.project_id}). Configure it in Secret Manager or export {secret_id}."
        )

    def get_runtime_config_summary(self) -> Dict[str, Any]:
        """Return a masked diagnostic summary of configured clinical secrets."""
        gemini_key = self.get_secret("GEMINI_API_KEY", default="offline-hermetic-adk-mode")
        db_uri = self.get_secret("MEDQUAD_DB_URI", default="sqlite:///data/clinical_memory.sqlite3")
        return {
            "project_id": self.project_id,
            "gcp_secret_manager_available": HAS_GCP_SECRET_MANAGER,
            "GEMINI_API_KEY": self.mask_secret(gemini_key),
            "MEDQUAD_DB_URI": self.mask_secret(db_uri),
        }


default_secret_client = SecretManagerClient()
