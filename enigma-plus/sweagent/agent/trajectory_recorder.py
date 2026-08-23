"""Atomic trajectory persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrajectoryRecorder:
    def __init__(self, *, enable_thought_recording: bool = True) -> None:
        self.enable_thought_recording = enable_thought_recording

    def normalize_step(self, step: dict[str, Any]) -> dict[str, Any]:
        result = dict(step)
        if not self.enable_thought_recording:
            result["thought"] = ""
        blocks = result.get("content_blocks")
        result["content_blocks"] = blocks if isinstance(blocks, list) else []
        raw_response = result.get("raw_response")
        result["raw_response"] = raw_response if isinstance(raw_response, dict) else {}
        usage = result.get("usage")
        result["usage"] = usage if isinstance(usage, dict) else {}
        result.setdefault("content_text", result.get("response", "") or "")
        result.setdefault("stop_reason", None)
        result.setdefault("context_compressed", False)
        return result

    def save(self, path: Path, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["trajectory"] = [self.normalize_step(s) for s in payload.get("trajectory", [])]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

