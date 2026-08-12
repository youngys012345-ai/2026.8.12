import unittest
from pathlib import Path

from awesome_llm_apps.config import PipelineConfig, RemoteConfig
from awesome_llm_apps.inference import load_inference_config
from awesome_llm_apps.orchestrator import build_plan, execute


def sample_config() -> PipelineConfig:
    return PipelineConfig(
        remote=RemoteConfig("URL"),
        tools={
            "download": "download",
            "train": "train",
            "deploy": "deploy",
            "compose": "compose",
            "process_media": "process",
        },
        models=[{"id": "m1", "source": "source", "train": True, "deploy": True}],
        compositions=[{"id": "c1", "models": ["m1"], "strategy": "sequential"}],
        datasets=[{"id": "d1", "composition": "c1", "uri": "/data"}],
    )


class PipelineTests(unittest.TestCase):
    def test_inference_configs_have_serial_stages(self) -> None:
        root = Path(__file__).parents[1]
        for name in (
            "inference.example.toml",
            "inference.production.example.toml",
            "model_params.example.json",
        ):
            config = load_inference_config(root / "configs" / name)
            self.assertGreaterEqual(len(config["stages"]), 1)

    def test_build_plan_orders_lifecycle_before_processing(self) -> None:
        self.assertEqual(
            [step.kind for step in build_plan(sample_config())],
            ["download", "train", "deploy", "compose", "process_media"],
        )

    def test_build_plan_skips_disabled_models(self) -> None:
        config = PipelineConfig(
            remote=RemoteConfig("URL"),
            tools=sample_config().tools,
            models=[
                {"id": "enabled-model", "source": "source"},
                {"id": "disabled-model", "source": "source", "enabled": False},
            ],
            compositions=[],
            datasets=[],
        )
        self.assertEqual([step.target for step in build_plan(config)], ["enabled-model"])

    def test_execute_passes_tool_and_arguments(self) -> None:
        calls = []

        class Client:
            def call_tool(self, name, arguments):
                calls.append((name, arguments))
                return {"ok": True}

        results = execute(sample_config(), Client())
        self.assertEqual(len(results), 5)
        self.assertEqual(calls[0][0], "download")
        self.assertEqual(calls[-1][1]["id"], "d1")


if __name__ == "__main__":
    unittest.main()
