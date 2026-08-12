from __future__ import annotations

import json
from typing import Any

import httpx


class McpError(RuntimeError):
    """Raised for MCP transport or JSON-RPC failures."""


class McpClient:
    def __init__(
        self, url: str, token: str | None, *, timeout: float = 120, verify_tls: bool = True
    ) -> None:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=url, headers=headers, timeout=timeout, verify=verify_tls
        )
        self._request_id = 0
        self._session_id: str | None = None

    def __enter__(self) -> McpClient:
        self.initialize()
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        if "text/event-stream" in response.headers.get("content-type", ""):
            data_lines = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
            if not data_lines:
                raise McpError("MCP SSE response did not contain a data event")
            return json.loads(data_lines[-1])
        return response.json()

    def _send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._request_id += 1
        headers = {"Mcp-Session-Id": self._session_id} if self._session_id else None
        response = self._client.post(
            "",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params or {},
            },
        )
        payload = self._decode(response)
        self._session_id = response.headers.get("Mcp-Session-Id", self._session_id)
        if "error" in payload:
            raise McpError(f"{method} failed: {payload['error']}")
        return payload.get("result")

    def initialize(self) -> None:
        self._send(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "industrial-video-detection", "version": "0.1.0"},
            },
        )
        headers = {"Mcp-Session-Id": self._session_id} if self._session_id else None
        response = self._client.post(
            "",
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        response.raise_for_status()

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._send("tools/list") or {}
        return list(result.get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._send("tools/call", {"name": name, "arguments": arguments})


