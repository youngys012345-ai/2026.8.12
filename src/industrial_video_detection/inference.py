from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class InferenceError(RuntimeError):
    """Raised when media inference cannot continue safely."""


class Stage(Protocol):
    def predict(self, image: Any, context: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class GroundingDinoStage:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise InferenceError(
                "inference dependencies are missing; install the project with [inference]"
            ) from exc

        device_name = self.config.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise InferenceError("CUDA was requested but torch.cuda.is_available() is false")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if device_name == "cuda" else device_name)
        dtype_name = self.config.get("dtype", "float16")
        self.dtype = getattr(torch, dtype_name)
        model_id = self.config["model_id"]
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        if self.config.get("compile", False):
            self.model = torch.compile(self.model)

    def predict(self, image: Any, context: dict[str, Any]) -> dict[str, Any]:
        import torch

        prompts = list(self.config.get("prompts", []))
        if not prompts:
            raise InferenceError("grounding_dino stage requires at least one prompt")
        # Grounding DINO expects a period-separated lowercase caption.
        text = ". ".join(item.strip().lower().rstrip(".") for item in prompts) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=float(self.config.get("box_threshold", 0.3)),
            text_threshold=float(self.config.get("text_threshold", 0.25)),
            target_sizes=[image.size[::-1]],
        )[0]
        detections = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            detections.append(
                {
                    "box": [round(float(value), 2) for value in box.tolist()],
                    "score": round(float(score), 5),
                    "label": str(label),
                }
            )
        return context | {"detections": detections}


@dataclass
class Owlv2Stage:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise InferenceError(
                "OWLv2 inference dependencies are missing; install torch and transformers"
            ) from exc

        device_name = self.config.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise InferenceError("CUDA was requested but torch.cuda.is_available() is false")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if device_name == "cuda" else device_name)
        model_id = self.config.get("model_id", "google/owlv2-base-patch16-ensemble")
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        if self.config.get("compile", False):
            self.model = torch.compile(self.model)

    def predict(self, image: Any, context: dict[str, Any]) -> dict[str, Any]:
        import torch

        prompts = list(self.config.get("prompts", []))
        if not prompts:
            raise InferenceError("owlv2 stage requires at least one prompt")
        text_labels = [[item.strip() for item in prompts]]
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)
        result = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=float(self.config.get("threshold", 0.1)),
            text_labels=text_labels,
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        detections = []
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            detections.append(
                {
                    "box": [round(float(value), 2) for value in box.tolist()],
                    "score": round(float(score), 5),
                    "label": str(label),
                }
            )
        return context | {"detections": detections}


@dataclass
class TorchScriptDetectorStage:
    """Stable production adapter for a detector exported with an Nx6 output contract."""

    config: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise InferenceError("PyTorch is required for TorchScript inference") from exc
        device_name = self.config.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise InferenceError("CUDA was requested but torch.cuda.is_available() is false")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if device_name == "cuda" else device_name)
        self.dtype = getattr(torch, self.config.get("dtype", "float16"))
        self.input_size = tuple(int(item) for item in self.config.get("input_size", [640, 640]))
        self.threshold = float(self.config.get("confidence_threshold", 0.3))
        self.labels = Path(self.config["labels_path"]).read_text(encoding="utf-8").splitlines()
        self.model = torch.jit.load(self.config["model_path"], map_location=self.device).eval()

    def predict(self, image: Any, context: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        source_width, source_height = image.size
        target_width, target_height = self.input_size
        scale = min(target_width / source_width, target_height / source_height)
        resized = image.resize((round(source_width * scale), round(source_height * scale)))
        canvas = Image.new("RGB", self.input_size, (114, 114, 114))
        offset_x = (target_width - resized.width) // 2
        offset_y = (target_height - resized.height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        rows = output.detach().float().cpu().reshape(-1, 6)
        detections = []
        for x1, y1, x2, y2, score, class_id in rows.tolist():
            if score < self.threshold:
                continue
            class_index = int(class_id)
            detections.append(
                {
                    "box": [
                        round(max(0.0, (x1 - offset_x) / scale), 2),
                        round(max(0.0, (y1 - offset_y) / scale), 2),
                        round(min(float(source_width), (x2 - offset_x) / scale), 2),
                        round(min(float(source_height), (y2 - offset_y) / scale), 2),
                    ],
                    "score": round(float(score), 5),
                    "class_id": class_index,
                    "label": self.labels[class_index]
                    if 0 <= class_index < len(self.labels)
                    else str(class_index),
                }
            )
        return context | {"detections": detections}


def load_inference_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
    stages = config.get("stages", [])
    if not stages:
        raise InferenceError("configuration requires at least one [[stages]] entry")
    return config


def build_stages(config: dict[str, Any]) -> list[Stage]:
    runtime = dict(config.get("runtime", {}))
    stages: list[Stage] = []
    for stage in config["stages"]:
        merged = runtime | dict(stage)
        if stage.get("type") == "grounding_dino":
            stages.append(GroundingDinoStage(merged))
        elif stage.get("type") == "owlv2":
            stages.append(Owlv2Stage(merged))
        elif stage.get("type") == "torchscript_detector":
            stages.append(TorchScriptDetectorStage(merged))
        else:
            raise InferenceError(f"unsupported stage type: {stage.get('type')!r}")
    return stages


def run_pipeline(image: Any, stages: list[Stage]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for stage in stages:
        context = stage.predict(image, context)
    return context


def annotate(image: Any, result: dict[str, Any]) -> Any:
    from PIL import ImageDraw

    output = image.copy()
    draw = ImageDraw.Draw(output)
    for detection in result.get("detections", []):
        box = detection["box"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), f"{detection['label']} {detection['score']:.2f}", fill="red")
    return output


def infer_image(config_path: str, input_path: str, output_dir: str) -> Path:
    from PIL import Image

    config = load_inference_config(config_path)
    stages = build_stages(config)
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    started = time.perf_counter()
    result = run_pipeline(image, stages) | {
        "source": str(source),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    result_path = destination / f"{source.stem}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if config.get("output", {}).get("save_annotated", True):
        annotate(image, result).save(destination / f"{source.stem}.annotated.jpg")
    return result_path


def _distributed_rank() -> tuple[int, int]:
    return int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))


def infer_video(config_path: str, input_path: str, output_dir: str, frame_stride: int) -> Path:
    import cv2
    from PIL import Image

    if frame_stride < 1:
        raise InferenceError("frame_stride must be at least 1")
    config = load_inference_config(config_path)
    stages = build_stages(config)
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rank, world_size = _distributed_rank()
    result_path = destination / f"{source.stem}.rank-{rank:02d}.jsonl"
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise InferenceError(f"cannot open video: {source}")

    frame_index = -1
    writer = None
    if world_size == 1 and config.get("output", {}).get("save_annotated", True):
        fps = capture.get(cv2.CAP_PROP_FPS) / frame_stride
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(destination / f"{source.stem}.annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    with result_path.open("w", encoding="utf-8") as stream:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                sampled_index = frame_index // frame_stride
                if frame_index % frame_stride or sampled_index % world_size != rank:
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                started = time.perf_counter()
                result = run_pipeline(image, stages) | {
                    "source": str(source),
                    "frame_index": frame_index,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                if writer is not None:
                    rendered = annotate(image, result)
                    writer.write(cv2.cvtColor(__import__("numpy").asarray(rendered), cv2.COLOR_RGB2BGR))
        finally:
            capture.release()
            if writer is not None:
                writer.release()
    return result_path


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="artifacts/inference")
    return parser


def main_image() -> int:
    parser = _common_parser("Run an image through the configured serial model pipeline")
    args = parser.parse_args()
    print(infer_image(args.config, args.input, args.output_dir))
    return 0


def main_video() -> int:
    parser = _common_parser("Run sampled video frames through the configured serial pipeline")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    print(infer_video(args.config, args.input, args.output_dir, args.frame_stride))
    return 0
