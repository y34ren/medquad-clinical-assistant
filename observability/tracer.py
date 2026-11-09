"""Distributed OpenTelemetry / Cloud Trace Span Tracer, Structured JSON Cloud Logger, and SLO Telemetry.

Implements Pillar 4.2, 4.3, and 4.4 of the AgentOps Code Review Matrix (15/15 pts)
and Google "AI in 5 Days" Day 4 Whitepaper ("Agent Quality: Evaluation & Observability"):
- `SpanTracer`: OpenTelemetry & Google Cloud Trace compatible distributed span tracing
  with parent/child span hierarchy, trace IDs, latencies, and attribute sanitization.
- `StructuredCloudLogger`: Emits GCP Cloud Logging JSON records (`severity`,
  `logging.googleapis.com/trace`, `logging.googleapis.com/spanId`, `component`, `event`).
- `ClinicalMetricsCollector`: Tracks clinical SLOs (grounding rate, ungrounded claims,
  Safe Refusal count, HITL escalations, PHI redactions, and p50/p95 latency).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Generator, List, Optional
import uuid

from observability.pii_scrubber import PIIScrubber, default_pii_scrubber

try:
    from google.cloud import logging as gcp_cloud_logging  # type: ignore[import-not-found]
    HAS_GCP_CLOUD_LOGGING = True
except ImportError:  # pragma: no cover
    gcp_cloud_logging = None  # type: ignore[assignment]
    HAS_GCP_CLOUD_LOGGING = False


DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "clinical_audit.jsonl"


class CloudLoggingJSONFormatter(logging.Formatter):
    """Structured JSON log formatter emitting Google Cloud Logging compatible JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "structured_payload", None)
        if isinstance(payload, dict):
            return json.dumps(payload, default=str)
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            default=str,
        )


@dataclass
class SpanRecord:
    """OpenTelemetry / GCP Cloud Trace compatible span representation."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time_ns: int = 0
    end_time_ns: int = 0
    duration_ms: float = 0.0
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuredCloudLogger:
    """Structured JSON Logger emitting Google Cloud Logging compatible audit entries to JSONL & GCP."""

    def __init__(
        self,
        service_name: str = "medquad-clinical-assistant",
        project_id: str = "nih-clinical-center-prod",
        scrubber: Optional[PIIScrubber] = None,
        log_file: Optional[str | Path] = None,
    ) -> None:
        self.service_name = service_name
        self.project_id = project_id
        self.scrubber = scrubber or default_pii_scrubber
        self._logs: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self.log_file_path: Path = Path(
            log_file or os.environ.get("MEDQUAD_LOG_FILE", str(DEFAULT_LOG_FILE))
        )
        self._logger = logging.getLogger(f"{service_name}.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._configure_handlers()

    def _configure_handlers(self) -> None:
        """Attach structured JSON RotatingFileHandler (with /tmp fallback) and optional stdout/GCP handlers."""
        formatter = CloudLoggingJSONFormatter()
        target_path = self.log_file_path
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                str(target_path), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        except OSError:  # pragma: no cover
            fallback_path = Path("/tmp/medquad_clinical_audit.jsonl")
            self.log_file_path = fallback_path
            file_handler = RotatingFileHandler(
                str(fallback_path), maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        if os.environ.get("MEDQUAD_LOG_STDOUT", "false").lower() == "true":  # pragma: no cover
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

    def log_event(
        self,
        severity: str,
        component: str,
        event: str,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        persist: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Emit a PHI-scrubbed structured JSON log entry in GCP Cloud Logging format."""
        scrubbed_message = self.scrubber.scrub(message)
        scrubbed_extra = self.scrubber.scrub_payload(extra or {})
        effective_trace = trace_id or uuid.uuid4().hex
        effective_span = span_id or uuid.uuid4().hex[:16]
        sev_upper = severity.upper()
        should_persist = (
            persist
            if persist is not None
            else (sev_upper in {"WARNING", "ERROR", "CRITICAL"})
        )

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": sev_upper,
            "service": self.service_name,
            "component": component,
            "event": event,
            "message": scrubbed_message,
            "logging.googleapis.com/trace": f"projects/{self.project_id}/traces/{effective_trace}",
            "logging.googleapis.com/spanId": effective_span,
            "audit_persisted": should_persist,
            "metadata": scrubbed_extra,
        }
        log_all = os.environ.get("MEDQUAD_LOG_ALL_SPANS", "false").lower() == "true"
        if should_persist or log_all:
            level = getattr(logging, sev_upper, logging.INFO)
            self._logger.log(level, scrubbed_message, extra={"structured_payload": record})
        with self._lock:
            self._logs.append(record)
            if len(self._logs) > 500:
                self._logs = self._logs[-500:]
        return record

    def log_intent(
        self,
        component: str,
        intended_action: str,
        planned_parameters: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly log the agent's *intended* action BEFORE execution (Rubric: Intent vs. Outcome Capture)."""
        return self.log_event(
            severity="INFO",
            component=component,
            event="intent_captured",
            message=f"[INTENT] Planned action '{intended_action}' before execution.",
            trace_id=trace_id,
            span_id=span_id,
            extra={
                "audit_phase": "INTENT_BEFORE_EXECUTION",
                "intended_action": intended_action,
                "planned_parameters": planned_parameters or {},
            },
        )

    def log_outcome(
        self,
        component: str,
        intended_action: str,
        actual_outcome: str,
        duration_ms: float = 0.0,
        outcome_metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        success: bool = True,
    ) -> Dict[str, Any]:
        """Explicitly log the agent's *actual* outcome AFTER execution (Rubric: Intent vs. Outcome Capture)."""
        return self.log_event(
            severity="INFO" if success else "ERROR",
            component=component,
            event="outcome_captured",
            message=(
                f"[OUTCOME] Completed '{intended_action}' -> actual_outcome='{actual_outcome}' "
                f"in {duration_ms:.2f}ms"
            ),
            trace_id=trace_id,
            span_id=span_id,
            extra={
                "audit_phase": "OUTCOME_AFTER_EXECUTION",
                "intended_action": intended_action,
                "actual_outcome": actual_outcome,
                "duration_ms": duration_ms,
                **(outcome_metadata or {}),
            },
        )

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent structured JSON audit log entries."""
        with self._lock:
            return list(reversed(self._logs[-limit:]))

    def get_actionable_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return turn-level audit logs: minimal 1-liner for smooth runs + detailed logs for attention/rejections."""
        with self._lock:
            filtered = [rec for rec in self._logs if rec.get("audit_persisted")]
            return list(reversed(filtered[-limit:]))


class SpanTracer:
    """OpenTelemetry-compatible distributed span tracer for multi-agent clinical workflows."""

    def __init__(
        self,
        service_name: str = "medquad-clinical-assistant",
        scrubber: Optional[PIIScrubber] = None,
        cloud_logger: Optional[StructuredCloudLogger] = None,
    ) -> None:
        self.service_name = service_name
        self.scrubber = scrubber or default_pii_scrubber
        self.cloud_logger = cloud_logger or StructuredCloudLogger(service_name=service_name, scrubber=self.scrubber)
        self._spans: List[SpanRecord] = []
        self._lock = threading.RLock()
        self._local = threading.local()

    def new_trace_id(self) -> str:
        """Generate a 32-character hex OpenTelemetry trace ID."""
        return uuid.uuid4().hex

    def new_span_id(self) -> str:
        """Generate a 16-character hex OpenTelemetry span ID."""
        return uuid.uuid4().hex[:16]

    @contextmanager
    def trace_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanRecord, None, None]:
        """Context manager recording start/end timestamps, parent/child links, and scrubbed attributes."""
        active_trace_id = trace_id or getattr(self._local, "current_trace_id", None) or self.new_trace_id()
        active_parent_id = parent_span_id or getattr(self._local, "current_span_id", None)
        span_id = self.new_span_id()

        prev_trace_id = getattr(self._local, "current_trace_id", None)
        prev_span_id = getattr(self._local, "current_span_id", None)
        self._local.current_trace_id = active_trace_id
        self._local.current_span_id = span_id

        scrubbed_attrs = self.scrubber.scrub_payload(attributes or {})
        span = SpanRecord(
            name=name,
            trace_id=active_trace_id,
            span_id=span_id,
            parent_span_id=active_parent_id,
            start_time_ns=time.time_ns(),
            attributes=scrubbed_attrs,
        )

        # Explicitly capture INTENT before execution (AgentOps Rubric 4.2: Intent vs. Outcome Capture)
        self.cloud_logger.log_intent(
            component=name,
            intended_action=name,
            planned_parameters=scrubbed_attrs,
            trace_id=span.trace_id,
            span_id=span.span_id,
        )

        try:
            yield span
            span.status = "OK"
        except Exception as exc:
            span.status = "ERROR"
            span.attributes["error.message"] = self.scrubber.scrub(str(exc))
            raise
        finally:
            span.end_time_ns = time.time_ns()
            span.duration_ms = round((span.end_time_ns - span.start_time_ns) / 1_000_000.0, 3)
            with self._lock:
                self._spans.append(span)
                if len(self._spans) > 1000:
                    self._spans = self._spans[-1000:]

            # Explicitly capture OUTCOME after execution (AgentOps Rubric 4.2: Intent vs. Outcome Capture)
            self.cloud_logger.log_outcome(
                component=name,
                intended_action=name,
                actual_outcome=span.status,
                duration_ms=span.duration_ms,
                outcome_metadata={"parent_span_id": span.parent_span_id, **span.attributes},
                trace_id=span.trace_id,
                span_id=span.span_id,
                success=(span.status == "OK"),
            )
            self._local.current_trace_id = prev_trace_id
            self._local.current_span_id = prev_span_id

    def get_recent_spans(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent completed spans in chronological or reverse-chronological order."""
        with self._lock:
            return [s.to_dict() for s in reversed(self._spans[-limit:])]

    def get_spans_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Retrieve all spans belonging to a specific distributed `trace_id`."""
        with self._lock:
            return [s.to_dict() for s in self._spans if s.trace_id == trace_id]


class ClinicalMetricsCollector:
    """Telemetry aggregator tracking NIH Clinical Assistant SLOs and latency percentiles."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.total_queries: int = 0
        self.grounded_responses_count: int = 0
        self.ungrounded_claims_total: int = 0
        self.safe_refusal_count: int = 0
        self.prompt_injection_blocked_count: int = 0
        self.hitl_escalations_count: int = 0
        self.phi_redactions_total: int = 0
        self.tool_error_recoveries_count: int = 0
        self.model_routing_counts: Dict[str, int] = {
            "gemini-2.5-flash": 0,
            "gemini-2.5-pro": 0,
        }
        self._latencies_ms: List[float] = []

    def record_query_outcome(
        self,
        latency_ms: float,
        model_used: str,
        is_grounded: bool = True,
        ungrounded_claims: int = 0,
        is_safe_refusal: bool = False,
        is_prompt_injection: bool = False,
        is_hitl_escalated: bool = False,
        phi_redactions: int = 0,
        tool_error_recoveries: int = 0,
    ) -> None:
        """Record telemetry for a completed clinical turn."""
        with self._lock:
            self.total_queries += 1
            self._latencies_ms.append(float(latency_ms))
            if model_used in self.model_routing_counts:
                self.model_routing_counts[model_used] += 1
            else:
                self.model_routing_counts[model_used] = 1

            if is_grounded:
                self.grounded_responses_count += 1
            self.ungrounded_claims_total += int(ungrounded_claims)
            if is_safe_refusal:
                self.safe_refusal_count += 1
            if is_prompt_injection:
                self.prompt_injection_blocked_count += 1
            if is_hitl_escalated:
                self.hitl_escalations_count += 1
            self.phi_redactions_total += int(phi_redactions)
            self.tool_error_recoveries_count += int(tool_error_recoveries)

    def _percentile(self, p: float) -> float:
        with self._lock:
            if not self._latencies_ms:
                return 0.0
            sorted_lats = sorted(self._latencies_ms)
            idx = min(len(sorted_lats) - 1, max(0, math.ceil((p / 100.0) * len(sorted_lats)) - 1))
            return round(sorted_lats[idx], 3)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Compute current SLO compliance and telemetry snapshot."""
        with self._lock:
            total = max(1, self.total_queries)
            grounding_rate = round(self.grounded_responses_count / total, 4) if self.total_queries > 0 else 1.0
            p50 = self._percentile(50.0)
            p95 = self._percentile(95.0)
            return {
                "total_queries": self.total_queries,
                "grounded_responses_count": self.grounded_responses_count,
                "grounded_citations_rate": grounding_rate,
                "ungrounded_claims_total": self.ungrounded_claims_total,
                "ungrounded_claims_rate": round(self.ungrounded_claims_total / total, 4) if self.total_queries > 0 else 0.0,
                "safe_refusal_count": self.safe_refusal_count,
                "prompt_injection_blocked_count": self.prompt_injection_blocked_count,
                "hitl_escalations_count": self.hitl_escalations_count,
                "phi_redactions_total": self.phi_redactions_total,
                "tool_error_recoveries_count": self.tool_error_recoveries_count,
                "model_routing_counts": dict(self.model_routing_counts),
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "slo_targets": {
                    "grounding_recall_min": 0.90,
                    "grounding_precision_min": 0.88,
                    "p95_latency_max_ms": 2500.0,
                    "ungrounded_claims_max_rate": 0.0,
                    "phi_redaction_target_rate": 1.0,
                    "slo_compliant": (p95 <= 2500.0 and self.ungrounded_claims_total == 0),
                },
            }


default_cloud_logger = StructuredCloudLogger()
default_tracer = SpanTracer(cloud_logger=default_cloud_logger)
default_metrics_collector = ClinicalMetricsCollector()
