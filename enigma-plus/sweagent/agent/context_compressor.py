"""Token-aware conversation compression for long-running CTF agents."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable


def count_tokens(messages: list[dict[str, Any]], tokenizer: Any | None = None) -> int:
    """Count message tokens using tiktoken when available, with a deterministic fallback."""
    text = "\n".join(
        json.dumps(m.get("content", ""), ensure_ascii=False, default=str)
        for m in messages
    )
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
        max_summary_input_chars: int | None = None,
        max_summary_output_chars: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_context_tokens = max_context_tokens
        self.trigger_ratio = trigger_ratio
        self.summary_model = summary_model
        self.max_summary_input_chars = max_summary_input_chars or int(
            os.environ.get("SWE_AGENT_CONTEXT_SUMMARY_INPUT_CHARS", "120000")
        )
        self.max_summary_output_chars = max_summary_output_chars or int(
            os.environ.get("SWE_AGENT_CONTEXT_SUMMARY_OUTPUT_CHARS", "24000")
        )
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _truncate(value: Any, limit: int) -> str:
        text = value if isinstance(value, str) else str(value)
        if len(text) <= limit:
            return text
        head = max(1, limit // 3)
        tail = max(1, limit - head)
        return text[:head] + "\n...[context truncated]...\n" + text[-tail:]

    def _bounded_conversation(self, messages: list[dict[str, Any]]) -> str:
        """Keep summary requests bounded while retaining the newest evidence."""
        bounded: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            if "content" in item:
                item["content"] = self._truncate(item["content"], 24000)
            for key in ("thought", "reasoning_content"):
                if key in item:
                    item[key] = self._truncate(item[key], 8000)
            bounded.append(item)
        serialized = json.dumps(bounded, ensure_ascii=False, indent=2, default=str)
        if len(serialized) <= self.max_summary_input_chars:
            return serialized
        # The first messages contain the task contract; the tail contains the
        # latest state. Drop only the middle, which is normally repetitive
        # command output.
        keep_head = bounded[:2]
        keep_tail = bounded[-6:]
        middle = {"role": "user", "content": "[middle history omitted during compression]"}
        serialized = json.dumps(keep_head + [middle] + keep_tail, ensure_ascii=False, indent=2, default=str)
        return self._truncate(serialized, self.max_summary_input_chars)

    def _fit_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure the compact history remains below the configured token budget."""
        compact = [dict(message) for message in messages]
        target_chars = max(4000, self.max_context_tokens * 3)
        serialized = json.dumps(compact, ensure_ascii=False, default=str)
        if len(serialized) <= target_chars and count_tokens(compact) <= self.max_context_tokens:
            return compact
        for message in compact:
            if "content" in message:
                message["content"] = self._truncate(message["content"], 12000)
            for key in ("thought", "reasoning_content"):
                if key in message:
                    message[key] = self._truncate(message[key], 4000)
        serialized = json.dumps(compact, ensure_ascii=False, default=str)
        if len(serialized) <= target_chars and count_tokens(compact) <= self.max_context_tokens:
            return compact
        system = [m for m in compact if m.get("role") == "system"]
        tail = [m for m in compact if m.get("role") != "system"][-6:]
        compact = system + [{"role": "user", "content": "[older context omitted; continue from the recent history]"}] + tail
        for message in compact:
            if "content" in message:
                message["content"] = self._truncate(message["content"], 4000)
            for key in ("thought", "reasoning_content"):
                if key in message:
                    message[key] = self._truncate(message[key], 1000)
        while count_tokens(compact) > self.max_context_tokens and len(tail) > 2:
            tail = tail[1:]
            compact = system + [{"role": "user", "content": "[older context omitted; continue from the recent history]"}] + tail
        if count_tokens(compact) > self.max_context_tokens:
            for message in compact:
                if "content" in message:
                    message["content"] = self._truncate(message["content"], 1000)
                for key in ("thought", "reasoning_content"):
                    if key in message:
                        message[key] = self._truncate(message[key], 256)
        return compact

    def maybe_compress(self, messages: list[dict[str, Any]]) -> CompressionResult:
        old_count = count_tokens(messages)
        threshold = int(self.max_context_tokens * self.trigger_ratio)
        if not self.enabled or old_count < threshold or self.summary_model is None:
            return CompressionResult(messages, old_count, old_count, "", False)

        system = [m for m in messages if m.get("role") == "system"]
        body = [m for m in messages if m.get("role") != "system"]
        conversation = self._bounded_conversation(body)
        prompt = SUMMARY_PROMPT.format(conversation=conversation)
        try:
            response = self.summary_model(
                system + [{"role": "user", "content": prompt, "agent": "context-compressor"}],
            )
            summary = self._truncate(response, self.max_summary_output_chars)
        except Exception as exc:
            summary = json.dumps({"compression_error": self._truncate(exc, 4000)}, ensure_ascii=False)

        # Keep the system contract and a small recent tail so the next action has local detail.
        tail = body[-4:]
        summary_message = {
            "role": "user",
            "agent": "context-compressor",
            "content": "Previous solving context (continue the same task and environment):\n" + summary,
            "context_summary": summary,
        }
        compact = self._fit_messages(system + [summary_message] + tail)
        new_count = count_tokens(compact)
        event = {
            "type": "compression",
            "old_token_count": old_count,
            "new_token_count": new_count,
            "summary": summary,
        }
        self.events.append(event)
        return CompressionResult(compact, old_count, new_count, summary, True)

