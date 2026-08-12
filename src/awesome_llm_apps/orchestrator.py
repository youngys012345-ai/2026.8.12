from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import PipelineConfig


class ToolCaller(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class Step:
    kind: str
    target: str
    tool: str
    arguments: dict[str, Any]


def build_plan(config: PipelineConfig) -> list[Step]:
    steps: list[Step] = []
    for model in config.models:
        common = {"model_id": model["id"]}
        steps.append(
            Step(
                "download",
                model["id"],
                config.tools["download"],
                common
                | {"source": model.get("source"), "revision": model.get("revision", "main")}
                | dict(model.get("download_args", {})),
            )
        )
        if model.get("train", False):
            steps.append(
                Step(
                    "train",
                    model["id"],
                    config.tools["train"],
                    common | dict(model.get("train_args", {})),
                )
            )
        if model.get("deploy", False):
            steps.append(
                Step(
                    "deploy",
                    model["id"],
                    config.tools["deploy"],
                    common | dict(model.get("deploy_args", {})),
                )
            )
    for composition in config.compositions:
        steps.append(
            Step("compose", composition["id"], config.tools["compose"], dict(composition))
        )
    for dataset in config.datasets:
        steps.append(
            Step(
                "process_media", dataset["id"], config.tools["process_media"], dict(dataset)
            )
        )
    return steps


def execute(config: PipelineConfig, client: ToolCaller) -> list[Any]:
    return [client.call_tool(step.tool, step.arguments) for step in build_plan(config)]

