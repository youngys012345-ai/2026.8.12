from __future__ import annotations

import os
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class RemoteConfig:
    url_env: str
    token_env: str | None = None
    timeout_seconds: float = 120
    verify_tls: bool = True

    def resolve(self) -> tuple[str, str | None]:
        url = os.getenv(self.url_env)
        if not url:
            raise ConfigError(f"environment variable {self.url_env!r} is required")
        token = os.getenv(self.token_env) if self.token_env else None
        if self.token_env and not token:
            raise ConfigError(f"environment variable {self.token_env!r} is required")
        return url, token


@dataclass(frozen=True)
class PipelineConfig:
    remote: RemoteConfig
    tools: dict[str, str]
    models: list[dict[str, Any]] = field(default_factory=list)
    compositions: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)

    def validate(self, *, require_environment: bool = False) -> None:
        required_tools = {"download", "train", "deploy", "compose", "process_media"}
        missing = sorted(required_tools - self.tools.keys())
        if missing:
            raise ConfigError(f"missing tool mappings: {', '.join(missing)}")

        model_ids = [str(item.get("id", "")) for item in self.models]
        if not model_ids or any(not item for item in model_ids):
            raise ConfigError("at least one model with a non-empty id is required")
        if len(model_ids) != len(set(model_ids)):
            raise ConfigError("model ids must be unique")

        composition_ids = {str(item.get("id", "")) for item in self.compositions}
        for item in self.compositions:
            unknown = set(item.get("models", [])) - set(model_ids)
            if unknown:
                raise ConfigError(f"composition {item.get('id')!r} uses unknown models: {unknown}")
        for item in self.datasets:
            if item.get("composition") not in composition_ids:
                raise ConfigError(
                    f"dataset {item.get('id')!r} references unknown composition "
                    f"{item.get('composition')!r}"
                )
        if require_environment:
            self.remote.resolve()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    remote = raw.get("remote", {})
    config = PipelineConfig(
        remote=RemoteConfig(
            url_env=str(remote.get("url_env", "INTRANET_MCP_URL")),
            token_env=remote.get("token_env"),
            timeout_seconds=float(remote.get("timeout_seconds", 120)),
            verify_tls=bool(remote.get("verify_tls", True)),
        ),
        tools={str(k): str(v) for k, v in raw.get("tools", {}).items()},
        models=list(raw.get("models", [])),
        compositions=list(raw.get("compositions", [])),
        datasets=list(raw.get("datasets", [])),
    )
    config.validate()
    return config
