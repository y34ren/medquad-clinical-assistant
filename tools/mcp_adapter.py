"""Model Context Protocol (MCP) Tool Registry Adapter for NIH MedQuAD Clinical Tools.

Implements the Google "AI in 5 Days" Day 2 Whitepaper ("Agent Tools & Interoperability
with Model Context Protocol (MCP)") best practices:
- Wraps task-focused, granular clinical tools in standardized MCP JSON-RPC 2.0 tool
  descriptors (`name`, `description`, `inputSchema`, `annotations`).
- Validates tool invocation payloads against strict Pydantic JSON schemas prior to
  execution and formats outputs into MCP `CallToolResult` content blocks.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from tools.clinical_tools import (
    CLINICAL_TOOLS,
    TOOL_JSON_SCHEMAS,
    escalate_to_attending_physician_hitl,
    retrieve_medquad_clinical_evidence,
    screen_drug_contraindications,
    verify_clinical_citation_grounding,
)


class MCPToolAdapter:
    """Adapter exposing a Python clinical tool function in MCP-compliant descriptor format."""

    def __init__(
        self,
        func: Callable[..., Dict[str, Any]],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        read_only_hint: bool = True,
        requires_hitl_hint: bool = False,
    ) -> None:
        self.func = func
        self.name = func.__name__
        self.description = (func.__doc__ or "").strip().split("\n")[0]
        self.full_docstring = (func.__doc__ or "").strip()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.read_only_hint = read_only_hint
        self.requires_hitl_hint = requires_hitl_hint

    def to_mcp_descriptor(self) -> Dict[str, Any]:
        """Return an MCP `tools/list` specification object."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "annotations": {
                "title": self.name.replace("_", " ").title(),
                "readOnlyHint": self.read_only_hint,
                "destructiveHint": False,
                "idempotentHint": self.read_only_hint,
                "openWorldHint": False,
                "clinicalGovernance": {
                    "hipaaCompliant": True,
                    "requiresHitl": self.requires_hitl_hint,
                    "nihCorpusGrounded": True,
                },
            },
        }

    def invoke(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the underlying clinical tool and wrap the response in MCP CallToolResult format."""
        raw_result = self.func(**arguments)
        is_error = raw_result.get("status") == "error"
        return {
            "jsonrpc": "2.0",
            "tool": self.name,
            "isError": is_error,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(raw_result, ensure_ascii=False),
                }
            ],
            "structuredContent": raw_result,
        }


class MCPServerRegistry:
    """MCP Server Registry exposing NIH MedQuAD clinical tools via JSON-RPC 2.0 semantics."""

    def __init__(self, server_name: str = "nih-medquad-clinical-mcp-server", version: str = "1.0.0") -> None:
        self.server_name = server_name
        self.version = version
        self._adapters: Dict[str, MCPToolAdapter] = {}
        self._register_default_clinical_tools()

    def _register_default_clinical_tools(self) -> None:
        """Register the 4 core NIH MedQuAD clinical tools with MCP annotations."""
        tool_map = {
            "retrieve_medquad_clinical_evidence": (retrieve_medquad_clinical_evidence, True, False),
            "screen_drug_contraindications": (screen_drug_contraindications, True, False),
            "verify_clinical_citation_grounding": (verify_clinical_citation_grounding, True, False),
            "escalate_to_attending_physician_hitl": (escalate_to_attending_physician_hitl, False, True),
        }
        for name, (fn, read_only, hitl_hint) in tool_map.items():
            schema_bundle = TOOL_JSON_SCHEMAS[name]
            self._adapters[name] = MCPToolAdapter(
                func=fn,
                input_schema=schema_bundle["input_schema"],
                output_schema=schema_bundle["output_schema"],
                read_only_hint=read_only,
                requires_hitl_hint=hitl_hint,
            )

    def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """Return MCP `tools/list` descriptors for all registered clinical tools."""
        return [adapter.to_mcp_descriptor() for adapter in self._adapters.values()]

    def get_mcp_tool_descriptor(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific MCP tool descriptor by name."""
        adapter = self._adapters.get(tool_name)
        return adapter.to_mcp_descriptor() if adapter else None

    def call_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute an MCP `tools/call` invocation with structured recovery on unknown tools."""
        adapter = self._adapters.get(tool_name)
        if adapter is None:
            return {
                "jsonrpc": "2.0",
                "tool": tool_name,
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "error",
                                "error_code": "UNKNOWN_MCP_TOOL",
                                "message": f"Tool '{tool_name}' is not registered in {self.server_name}.",
                                "recovery_suggestion": f"Select one of the registered MCP tools: {list(self._adapters.keys())}.",
                            }
                        ),
                    }
                ],
            }
        return adapter.invoke(arguments or {})

    def export_mcp_manifest(self) -> Dict[str, Any]:
        """Export the complete MCP Server capability manifest."""
        return {
            "protocolVersion": "2025-03-26",
            "serverInfo": {
                "name": self.server_name,
                "version": self.version,
                "organization": "National Institutes of Health (NIH) Clinical Center",
            },
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "tools": self.list_mcp_tools(),
        }


_DEFAULT_MCP_REGISTRY: Optional[MCPServerRegistry] = None


def get_default_mcp_registry() -> MCPServerRegistry:
    """Return the singleton MCPServerRegistry instance."""
    global _DEFAULT_MCP_REGISTRY
    if _DEFAULT_MCP_REGISTRY is None:
        _DEFAULT_MCP_REGISTRY = MCPServerRegistry()
    return _DEFAULT_MCP_REGISTRY
