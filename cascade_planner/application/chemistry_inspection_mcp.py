"""Single-purpose stdio MCP server for local mapped-SMILES inspection."""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from chemistry_inspection import inspect_mapped_smiles


TOOL_NAME = "inspect_mapped_smiles"
SERVER_NAME = "autoplanner-chemistry-inspection"
SERVER_VERSION = "1.0.0"


def _write(message: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(message), ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _result(request_id: Any, result: Any) -> None:
    _write({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: Any, code: int, message: str) -> None:
    _write(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": int(code), "message": str(message)},
        }
    )


def _tool_definition() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": (
            "Inspect mapped SMILES with local RDKit and return mapped atom facts, "
            "local adjacency and bond orders, ring paths, CIP centers, stereo "
            "bonds, and optionally a bounded enumeration of unassigned "
            "stereoisomers."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "smiles": {"type": "string"},
                "map_ids": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "maxItems": 16,
                },
                "enumerate_unassigned": {"type": "boolean"},
                "max_isomers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 32,
                },
            },
            "required": ["smiles"],
        },
    }


def _call_tool(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise ValueError("arguments must be an object")
    smiles = str(arguments.get("smiles") or "")
    if not smiles:
        raise ValueError("smiles is required")
    raw_map_ids = arguments.get("map_ids") or []
    if not isinstance(raw_map_ids, list):
        raise ValueError("map_ids must be an array")
    result = inspect_mapped_smiles(
        smiles,
        map_ids=[int(value) for value in raw_map_ids],
        enumerate_unassigned=bool(arguments.get("enumerate_unassigned", False)),
        max_isomers=int(arguments.get("max_isomers") or 8),
    )
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ],
        "structuredContent": result,
        "isError": result.get("ok") is not True,
    }


def _handle(message: Any) -> bool:
    if not isinstance(message, Mapping):
        return True
    method = str(message.get("method") or "")
    request_id = message.get("id")
    if not method:
        if request_id is not None:
            _error(request_id, -32600, "invalid request")
        return True
    if method.startswith("notifications/"):
        return True
    if method == "initialize":
        params = message.get("params")
        requested = (
            str(params.get("protocolVersion") or "")
            if isinstance(params, Mapping)
            else ""
        )
        _result(
            request_id,
            {
                "protocolVersion": requested or "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
        return True
    if method == "ping":
        _result(request_id, {})
        return True
    if method == "tools/list":
        _result(request_id, {"tools": [_tool_definition()]})
        return True
    if method == "tools/call":
        params = message.get("params")
        if not isinstance(params, Mapping) or str(params.get("name") or "") != TOOL_NAME:
            _error(request_id, -32601, "unknown tool")
            return True
        try:
            _result(request_id, _call_tool(params.get("arguments")))
        except (TypeError, ValueError) as exc:
            _error(request_id, -32602, str(exc))
        except Exception as exc:  # pragma: no cover - defensive protocol boundary
            _error(request_id, -32603, f"{type(exc).__name__}: {exc}")
        return True
    if method in {"resources/list", "prompts/list"}:
        key = "resources" if method.startswith("resources/") else "prompts"
        _result(request_id, {key: []})
        return True
    if method == "shutdown":
        _result(request_id, None)
        return False
    _error(request_id, -32601, f"method not found: {method}")
    return True


def main() -> int:
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            _error(None, -32700, "parse error")
            continue
        if not _handle(message):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
