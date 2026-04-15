"""Tests for MCP URL parsing helpers."""

from reachy_mini_conversation_app.tools.mcp_bridge import (
    openai_function_name,
    parse_mcp_urls_value,
    format_mcp_urls_for_env,
)


def test_parse_mcp_urls_json_array() -> None:
    """JSON array form in REACHY_MINI_MCP_URLS."""
    raw = '["http://localhost:4020/mcp", "https://x.example/mcp"]'
    assert parse_mcp_urls_value(raw) == ["http://localhost:4020/mcp", "https://x.example/mcp"]


def test_parse_mcp_urls_delimited() -> None:
    """Newline, semicolon, and pipe delimiters."""
    assert parse_mcp_urls_value("http://a/mcp\nhttp://b/mcp") == ["http://a/mcp", "http://b/mcp"]
    assert parse_mcp_urls_value("http://a/mcp;http://b/mcp") == ["http://a/mcp", "http://b/mcp"]
    assert parse_mcp_urls_value("http://a/mcp|http://b/mcp") == ["http://a/mcp", "http://b/mcp"]


def test_parse_mcp_urls_empty() -> None:
    """Empty and None input."""
    assert parse_mcp_urls_value("") == []
    assert parse_mcp_urls_value(None) == []


def test_format_round_trip() -> None:
    """format_mcp_urls_for_env round-trips through parse."""
    urls = ["http://h/mcp", "https://z/mcp"]
    assert parse_mcp_urls_value(format_mcp_urls_for_env(urls)) == urls


def test_openai_function_name() -> None:
    """Prefixed OpenAI names stay within length limits."""
    n = openai_function_name(0, 1, "my-tool")
    assert n.startswith("mcp0_1_")
    assert len(n) <= 64
