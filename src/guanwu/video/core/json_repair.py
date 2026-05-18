"""Lightweight JSON repair for VLM outputs.

Handles common LLM JSON mistakes: trailing commas, single quotes,
unquoted keys, <think> blocks, and markdown code fences.
"""
from __future__ import annotations

import json
import re


def parse_vlm_json(raw: str, fallback: str = "{}") -> object:
    """Extract and parse JSON from raw VLM output.

    Strips <think> blocks, code fences, then attempts ``json.loads``.
    On failure, applies lightweight repairs and retries.
    """
    text = _strip_wrappers(raw)
    # Fast path: valid JSON as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Repair and retry
    text = _repair(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: extract first JSON object/array substring
    extracted = _extract_json_substring(text)
    if extracted is not None:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    return json.loads(fallback)


def _strip_wrappers(raw: str) -> str:
    text = raw.strip()
    # Strip <think>...</think>
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


def _repair(text: str) -> str:
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single-quoted strings with double-quoted
    # Only do this if there are no double quotes (to avoid breaking valid JSON)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    # Quote unquoted keys:  { key: "value" } -> { "key": "value" }
    text = re.sub(r'(?<=[\{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', text)
    return text


def _extract_json_substring(text: str) -> str | None:
    """Find the first balanced {...} or [...] in text."""
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    candidate = _repair(candidate)
                    return candidate
    return None
