"""Token-aware conversation compression for long-running CTF agents."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


def count_tokens(messages: list[dict[str, Any]], tokenizer: Any | None = None) -> int:
    """Count message tokens using tiktoken when available, with a deterministic fallback."""
    text = "\n".join(str(m.get("content", "")) for m in messages)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


SUMMARY_PROMPT = """You are a CTF expert maintaining memory for an agent. Summarize the complete solving process below as structured JSON. Preserve:
1. current task objective
2. every action already executed
3. important observations from each step
4. discovered vulnerability information
5. current environment state and known file paths
6. tested methods and failed methods
7. the most likely successful method
8. concrete next-stage recommendations
Do not restart the task. The same container, terminal state, workspace, and flag objective remain active.

Conversation:
{conversation}

Return JSON with keys: task, progress, actions, important_observations, vulnerabilities, current_state, known_files, tested_methods, failed_methods, likely_solution, next_steps."""


@dataclass
class CompressionResult:
    messages: list[dict[str, Any]]
    old_token_count: int
    new_token_count: int
    summary: str
    compressed: bool = False


class ContextCompressionManager:
    def __init__(
        self,
        *,
        enabled: bool = False,
        max_context_tokens: int = 128000,
        trigger_ratio: float = 0.95,
        summary_model: Callable[[list[dict[str, Any]]], Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_context_tokens = max_context_tokens
        self.trigger_ratio = trigger_ratio
        self.summary_model = summary_model
        self.events: list[dict[str, Any]] = []

    def maybe_compress(self, messages: list[dict[str, Any]]) -> CompressionResult:
        old_count = count_tokens(messages)
        threshold = int(self.max_context_tokens * self.trigger_ratio)
        if not self.enabled or old_count < threshold or self.summary_model is None:
            return CompressionResult(messages, old_count, old_count, "", False)

        system = [m for m in messages if m.get("role") == "system"]
        body = [m for m in messages if m.get("role") != "system"]
        conversation = json.dumps(body, ensure_ascii=False, indent=2)
        prompt = SUMMARY_PROMPT.format(conversation=conversation)
        try:
            response = self.summary_model(
                system + [{"role": "user", "content": prompt, "agent": "context-compressor"}],
            )
            summary = str(response)
        except Exception as exc:
            summary = json.dumps({"compression_error": str(exc)}, ensure_ascii=False)

        # Keep the system contract and a small recent tail so the next action has local detail.
        tail = body[-4:]
        summary_message = {
            "role": "user",
            "agent": "context-compressor",
            "content": "Previous solving context (continue the same task and environment):\n" + summary,
            "context_summary": summary,
        }
        compact = system + [summary_message] + tail
        new_count = count_tokens(compact)
        event = {
            "type": "compression",
            "old_token_count": old_count,
            "new_token_count": new_count,
            "summary": summary,
        }
        self.events.append(event)
        return CompressionResult(compact, old_count, new_count, summary, True)

