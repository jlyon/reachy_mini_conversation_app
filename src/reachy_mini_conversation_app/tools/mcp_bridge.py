"""Bridge Model Context Protocol (streamable HTTP) servers into OpenAI tool calling."""

from __future__ import annotations
import re
import json
import asyncio
import logging
from typing import Any

from mcp import ClientSession
from mcp.types import Tool as MCPTool
from mcp.types import TextContent, CallToolResult, PaginatedRequestParams
from mcp.client.streamable_http import streamable_http_client


logger = logging.getLogger(__name__)

_MCP_LIST_TIMEOUT_S = 20.0
_MCP_CALL_TIMEOUT_S = 120.0


def openai_function_name(server_idx: int, tool_idx: int, raw_name: str) -> str:
    """Build a unique OpenAI-safe function name for an MCP tool."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
    if not safe:
        safe = "tool"
    out = f"mcp{server_idx}_{tool_idx}_{safe}"
    return out[:64]


def _normalize_parameters_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure JSON Schema is object-shaped for OpenAI function parameters."""
    if not schema:
        return {"type": "object", "properties": {}}
    if schema.get("type") == "object":
        return schema
    # Some servers omit type; wrap defensively
    if "properties" in schema:
        return {"type": "object", **schema}
    return {"type": "object", "properties": {}, "additionalProperties": True}


def mcp_tool_to_openai_spec(prefixed_name: str, tool: MCPTool) -> dict[str, Any]:
    """Convert an MCP Tool to OpenAI Realtime function tool spec."""
    desc = (tool.description or "").strip() or f"MCP tool `{tool.name}`"
    return {
        "type": "function",
        "name": prefixed_name,
        "description": desc[:4096],
        "parameters": _normalize_parameters_schema(tool.inputSchema),
    }


async def _list_tools_paginated(session: ClientSession) -> list[MCPTool]:
    collected: list[MCPTool] = []
    cursor: str | None = None
    while True:
        if cursor is None:
            result = await session.list_tools()
        else:
            result = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
        collected.extend(result.tools)
        next_c = getattr(result, "nextCursor", None)
        if not next_c:
            break
        cursor = str(next_c)
    return collected


async def collect_mcp_tool_specs(
    urls: list[str],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    """Fetch tool definitions from MCP HTTP endpoints and build OpenAI specs + routing table.

    Returns:
        OpenAI function tool specs and a map from prefixed name to ``(mcp_url, original_tool_name)``.

    """
    specs: list[dict[str, Any]] = []
    router: dict[str, tuple[str, str]] = {}

    async def _one_server(u: str) -> list[MCPTool]:
        async with streamable_http_client(u) as (read_stream, write_stream, _get_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await _list_tools_paginated(session)

    for server_idx, url in enumerate(urls):
        u = url.strip()
        if not u:
            continue
        try:
            mcp_tools = await asyncio.wait_for(_one_server(u), timeout=_MCP_LIST_TIMEOUT_S)
        except Exception as e:
            logger.warning("MCP list_tools failed for %r: %s", u, e)
            continue

        for tool_idx, tool in enumerate(mcp_tools):
            prefixed = openai_function_name(server_idx, tool_idx, tool.name)
            if prefixed in router:
                prefixed = openai_function_name(server_idx, tool_idx, f"{tool.name}_{tool_idx}")
            specs.append(mcp_tool_to_openai_spec(prefixed, tool))
            router[prefixed] = (u, tool.name)

    return specs, router


def _call_tool_result_to_dict(result: CallToolResult) -> dict[str, Any]:
    """Serialize MCP CallToolResult for the conversation / model."""
    texts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            texts.append(block.text)
        else:
            texts.append(f"[{type(block).__name__}]")

    combined = "\n".join(t for t in texts if t)
    if result.isError:
        return {"error": combined or "MCP tool returned an error"}

    out: dict[str, Any] = {"text": combined}
    if result.structuredContent is not None:
        out["structured"] = result.structuredContent
    return out


async def call_mcp_tool(url: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Connect to one MCP server and invoke a tool by its original MCP name."""

    async def _call() -> dict[str, Any]:
        async with streamable_http_client(url) as (read_stream, write_stream, _get_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments or {})
                return _call_tool_result_to_dict(result)

    try:
        return await asyncio.wait_for(_call(), timeout=_MCP_CALL_TIMEOUT_S)
    except Exception as e:
        logger.exception("MCP call_tool failed url=%r tool=%r", url, tool_name)
        return {"error": f"{type(e).__name__}: {e}"}


def format_mcp_urls_for_env(urls: list[str]) -> str:
    """Serialize URL list for REACHY_MINI_MCP_URLS (single-line JSON array)."""
    cleaned = [u.strip() for u in urls if u and u.strip()]
    return json.dumps(cleaned)


def parse_mcp_urls_value(raw: str | None) -> list[str]:
    """Parse REACHY_MINI_MCP_URLS from env or persisted file (JSON array or delimited text)."""
    if raw is None:
        return []
    s = raw.strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    out: list[str] = []
    for part in re.split(r"[\n\r;|]+", s):
        u = part.strip()
        if u:
            out.append(u)
    return out
