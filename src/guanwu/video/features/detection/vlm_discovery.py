from __future__ import annotations

import json
import os
import time
from typing import Any

from guanwu.video.core.config import VLMConfig
from guanwu.video.core.json_repair import parse_vlm_json
from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)

DISCOVERY_SYS = (
    "You are a vision assistant for a physical simulation. "
    "Given a image, list ALL distinct objects visible in this image. "
    "Include every identifiable object, for example: furniture, appliances, tools, containers, "
    "decorations, electronics, food items, toys, people, animals, vehicles, etc. "
    "Exclude only: walls, floor, ceiling, and the room/space itself. "
    "Return strictly a JSON array of lowercase English category names."
)

INCREMENTAL_DISCOVERY_SYS = (
    "You are a vision assistant for a physical simulation. "
    "You are given a image and a list of object categories that have "
    "already been detected. Your job is to find NEW objects that are NOT in the "
    "already-known list. Focus on objects that may have entered the scene or were "
    "previously missed. "
    "Return strictly a JSON array of lowercase English category names for newly "
    "discovered objects only. Return an empty array [] if no new objects are found."
)


class VLMDiscoveryAgent:
    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = config.base_url
        self.model = config.model
        self.max_retries = config.max_retries
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except ImportError:
                logger.warning("VLM Discovery Agent disabled: openai package is not installed.")

    def _call_vlm_with_retry(self, messages: list[dict[str, Any]]) -> str:
        """Call the VLM API with exponential backoff retry. Returns raw response text."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                )
                raw = completion.choices[0].message.content or "[]"
                parsed = parse_vlm_json(raw, fallback="[]")
                return json.dumps(parsed)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"[VLMDiscoveryAgent] Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
        logger.error(
            f"[VLMDiscoveryAgent] All {self.max_retries} attempts failed. Last error: {last_exc}"
        )
        raise last_exc  # type: ignore[misc]

    def discover_objects(self, image_b64: str) -> list[str]:
        """Discover ALL objects visible in the frame."""
        if not self.client:
            return []

        image_data = (image_b64 or "").strip()
        if not image_data:
            return []

        if not image_data.startswith("data:"):
            data_url = f"data:image/jpeg;base64,{image_data}"
        else:
            data_url = image_data

        logger.info(f"[VLMDiscoveryAgent] Running full discovery with model {self.model}...")
        try:
            raw = self._call_vlm_with_retry([
                {"role": "system", "content": DISCOVERY_SYS},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Return ONLY a JSON array of all object categories visible in this frame."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ])
            arr = json.loads(raw)
            if isinstance(arr, list):
                result = [str(x).strip().lower() for x in arr if str(x).strip()]
                logger.info(f"[VLMDiscoveryAgent] Discovered: {result}")
                return result
        except Exception as e:
            logger.error(f"[VLMDiscoveryAgent] Discovery failed: {e}")
            return []
        return []

    def discover_incremental(self, image_b64: str, existing_categories: list[str]) -> list[str]:
        """Discover only NEW objects not already in existing_categories."""
        if not self.client:
            return []

        image_data = (image_b64 or "").strip()
        if not image_data:
            return []

        if not image_data.startswith("data:"):
            data_url = f"data:image/jpeg;base64,{image_data}"
        else:
            data_url = image_data

        known_str = ", ".join(existing_categories) if existing_categories else "(none)"
        logger.info(f"[VLMDiscoveryAgent] Running incremental discovery (known: {known_str})...")
        try:
            raw = self._call_vlm_with_retry([
                {"role": "system", "content": INCREMENTAL_DISCOVERY_SYS},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Already known objects: [{known_str}]\n\n"
                                "Return ONLY a JSON array of NEW object categories "
                                "that are NOT in the list above."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ])
            arr = json.loads(raw)
            if isinstance(arr, list):
                # Filter out anything that's already known
                existing_lower = {c.lower() for c in existing_categories}
                result = [
                    str(x).strip().lower()
                    for x in arr
                    if str(x).strip() and str(x).strip().lower() not in existing_lower
                ]
                if result:
                    logger.info(f"[VLMDiscoveryAgent] Incremental discovery: {result}")
                else:
                    logger.info("[VLMDiscoveryAgent] Incremental discovery: no new objects found.")
                return result
        except Exception as e:
            logger.error(f"[VLMDiscoveryAgent] Incremental discovery failed: {e}")
            return []
        return []

    # Keep backward-compatible alias
    def discover_movable_object_categories(self, image_b64: str) -> list[str]:
        return self.discover_objects(image_b64)
