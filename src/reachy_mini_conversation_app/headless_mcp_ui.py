"""Admin UI routes for MCP server URL configuration (headless settings app)."""

from __future__ import annotations
import logging
from typing import Any, Callable

from fastapi import FastAPI


logger = logging.getLogger(__name__)


def mount_mcp_routes(
    app: FastAPI,
    *,
    get_urls: Callable[[], list[str]],
    set_urls: Callable[[list[str]], None],
) -> None:
    """Register GET/POST /mcp_urls on the settings app."""
    try:
        from pydantic import Field, BaseModel
        from fastapi.responses import JSONResponse
    except Exception:  # pragma: no cover
        return

    class McpUrlsPayload(BaseModel):
        urls: list[str] = Field(default_factory=list)

    @app.get("/mcp_urls")
    def _get_mcp_urls() -> dict[str, Any]:
        return {"urls": get_urls()}

    @app.post("/mcp_urls")
    def _post_mcp_urls(payload: McpUrlsPayload) -> JSONResponse:
        cleaned: list[str] = []
        for u in payload.urls:
            s = str(u).strip()
            if not s:
                continue
            if not (s.startswith("http://") or s.startswith("https://")):
                return JSONResponse(
                    {"ok": False, "error": "invalid_url", "detail": f"URL must start with http:// or https://: {s!r}"},
                    status_code=400,
                )
            cleaned.append(s)
        try:
            set_urls(cleaned)
        except Exception as e:
            logger.warning("Failed to persist MCP URLs: %s", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        return JSONResponse({"ok": True, "urls": cleaned})
