from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from guanwu.video.model_backend.config import VLMProviderConfig
from guanwu.video.model_backend.schemas import FrameDetectionsModel
from guanwu.video.model_backend.providers.base import post_json


MOVABLE_DISCOVERY_SYS = (
    "You are a vision assistant for a physical simulation. "
    "Given the first frame of a video, list ALL movable objects (things a human can pick up or push). "
    "Exclude: walls, floor, ceiling, fixed furniture that cannot be moved (e.g. built-in cabinets). "
    "Return strictly a JSON array of lowercase English category names."
)


class DisabledVLMProvider:
    mode = "disabled"

    def infer_object_physics_priors(
        self,
        detections: FrameDetectionsModel,
        objects: list[dict[str, Any]],
    ) -> dict[str, dict]:
        _ = (detections, objects)
        return {}

    def discover_movable_object_categories(self, image_b64: str) -> list[str]:
        _ = image_b64
        return []


class EmbeddedVLMProvider:
    mode = "embedded"

    def __init__(self, cfg: VLMProviderConfig) -> None:
        self.backend = cfg.backend.strip().lower()
        self.api_key = cfg.api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = cfg.base_url
        self.model = cfg.model
        self.command_template = (cfg.command_template or "").strip()

    def infer_object_physics_priors(
        self,
        detections: FrameDetectionsModel,
        objects: list[dict[str, Any]],
    ) -> dict[str, dict]:
        if self.backend == "command" and self.command_template:
            return self._command_infer(detections, objects)
        if self.backend == "api" and self.api_key:
            data = self._api_infer(objects)
            if data is not None:
                return data
        raise NotImplementedError("VLM embedded backend requires 'api' or 'command' mode. 'mock' mode has been removed.")

    def discover_movable_object_categories(self, image_b64: str) -> list[str]:
        if self.backend == "api" and self.api_key:
            found = self._api_discover(image_b64)
            if found:
                return found
        raise NotImplementedError("VLM discovery requires valid 'api' backend mode with reachable endpoint. Default mock values have been removed.")



    def _command_infer(
        self,
        detections: FrameDetectionsModel,
        objects: list[dict[str, Any]],
    ) -> dict[str, dict]:
        payload = {
            "frame_idx": detections.frame_idx,
            "timestamp": detections.timestamp,
            "image_b64": detections.image_b64,
            "objects": objects,
        }
        proc = subprocess.run(
            shlex.split(self.command_template),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"VLM command failed: {proc.stderr.strip()}")
        data = json.loads(proc.stdout)
        if not isinstance(data, dict):
            raise RuntimeError("VLM command output must be JSON object")
        return data

    def _api_infer(self, objects: list[dict[str, Any]]) -> dict[str, dict] | None:
        try:
            from openai import OpenAI
        except Exception:
            return None
        try:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            objects_desc = [
                {"object_id": str(obj.get("object_id", "obj_unknown")), "label": str(obj.get("label", "unknown"))}
                for obj in objects
            ]
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return JSON physics priors keyed by object_id."},
                    {"role": "user", "content": json.dumps(objects_desc)},
                ],
                temperature=0.1,
            )
            raw = (completion.choices[0].message.content or "{}").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(raw)
            if isinstance(data, dict):
                # Cannot merge with defaults since defaults are removed
                return data
        except Exception:
            return None
        return None

    def _api_discover(self, image_b64: str) -> list[str]:
        image_data = (image_b64 or "").strip()
        if not image_data:
            return []
        try:
            from openai import OpenAI
        except Exception:
            return []
        try:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            # Support both raw base64 and data URI schemes
            if not image_data.startswith("data:"):
                data_url = f"data:image/jpeg;base64,{image_data}"
            else:
                data_url = image_data
                
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MOVABLE_DISCOVERY_SYS},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Return ONLY a JSON array of movable object categories.",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.1,
            )
            raw = (completion.choices[0].message.content or "[]").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(x).strip().lower() for x in arr if str(x).strip()]
        except Exception:
            return []
        return []


class HttpVLMProvider:
    def __init__(
        self,
        base_url: str,
        timeout_sec: float,
        mode: str,
    ) -> None:
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.mode = mode

    def infer_object_physics_priors(
        self,
        detections: FrameDetectionsModel,
        objects: list[dict[str, Any]],
    ) -> dict[str, dict]:
        payload = {
            "detections": detections.model_dump(),
            "objects": objects,
        }
        data = post_json(self.base_url, "/v1/tasks/infer-object-physics-priors", payload, self.timeout_sec)
        out = data.get("priors")
        return out if isinstance(out, dict) else {}

    def discover_movable_object_categories(self, image_b64: str) -> list[str]:
        data = post_json(self.base_url, "/v1/tasks/discover-movable-object-categories", {"image_b64": image_b64}, self.timeout_sec)
        categories = data.get("categories")
        return [str(x) for x in categories] if isinstance(categories, list) else []


def build_vlm_provider(cfg: VLMProviderConfig):
    mode = cfg.mode.strip().lower()
    if mode == "http":
        return HttpVLMProvider(cfg.service.base_url, cfg.service.timeout_sec, mode="http")
    if mode == "embedded":
        return EmbeddedVLMProvider(cfg)
    return DisabledVLMProvider()
