# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


#
from __future__ import annotations

import copy
import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


def content_text_projection(content: Any) -> str:
    """Return text only where a history processor needs text semantics.

    Structured Anthropic content must remain structured in history.  This
    projection is deliberately limited to statistics and window matching; it
    is never written back for entries that are retained in the history.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            text for text in (content_text_projection(item) for item in content) if text
        )
    if isinstance(content, dict):
        parts: list[str] = []
        for key in ("text", "thinking", "reasoning_content", "content", "data"):
            value = content.get(key)
            if isinstance(value, (str, list, dict)):
                projected = content_text_projection(value)
                if projected:
                    parts.append(projected)
        return "\n".join(parts)
    return str(content)


def _replace_text_blocks(content: Any, replacement: str) -> Any:
    """Replace text-bearing block payloads while preserving block structure."""
    if isinstance(content, str):
        return replacement
    if isinstance(content, list):
        copied = [dict(item) if isinstance(item, dict) else item for item in content]
        for index, item in enumerate(copied):
            if isinstance(item, dict) and str(item.get("type", "")).lower() in {"text", "thinking", "reasoning"}:
                key = "text" if "text" in item else "thinking" if "thinking" in item else "reasoning_content"
                if key in item:
                    item[key] = replacement
                    return copied
            copied[index] = _replace_text_blocks(item, replacement)
        copied.append({"type": "text", "text": replacement})
        return copied
    if isinstance(content, dict):
        copied = dict(content)
        block_type = str(copied.get("type", "")).lower()
        # Tool results/calls and media blocks are structured payloads; their
        # ``content``/``input`` fields are not history text windows.
        if block_type in {"tool_result", "tool_use", "tool_call", "function", "image", "document"}:
            return copied
        for key in ("text", "thinking", "reasoning_content"):
            if isinstance(copied.get(key), str):
                copied[key] = replacement
                return copied
        if isinstance(copied.get("content"), (str, list, dict)):
            copied["content"] = _replace_text_blocks(copied["content"], replacement)
        return copied
    return replacement


def _omitted_content(content: Any) -> Any:
    """Create an omitted marker without flattening provider-native blocks."""
    line_count = len(content_text_projection(content).splitlines())
    replacement = f"Old output omitted ({line_count} lines)"
    if isinstance(content, str) or content is None:
        return replacement
    copied = copy.deepcopy(content)
    if isinstance(copied, list):
        for item in copied:
            if isinstance(item, dict):
                block_type = str(item.get("type", "")).lower()
                if block_type in {"text", "thinking", "reasoning"}:
                    key = "text" if "text" in item else "thinking" if "thinking" in item else "reasoning_content"
                    item[key] = replacement
                    return copied
        copied.append({"type": "text", "text": replacement})
        return copied
    if isinstance(copied, dict):
        block_type = str(copied.get("type", "")).lower()
        if block_type in {"text", "thinking", "reasoning"}:
            key = "text" if "text" in copied else "thinking" if "thinking" in copied else "reasoning_content"
            copied[key] = replacement
            return copied
        if isinstance(copied.get("content"), (str, list, dict)):
            copied["content"] = _omitted_content(copied["content"])
            return copied
    return [{"type": "text", "text": replacement}]


class FormatError(Exception):
    pass


# ABSTRACT BASE CLASSES


class HistoryProcessorMeta(type):
    _registry = {}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "HistoryProcessor":
            cls._registry[name] = new_cls
        return new_cls


@dataclass
class HistoryProcessor(metaclass=HistoryProcessorMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, history: list[str]) -> list[str]:
        raise NotImplementedError

    @classmethod
    def get(cls, name, *args, **kwargs):
        try:
            return cls._registry[name](*args, **kwargs)
        except KeyError:
            msg = f"Model output parser ({name}) not found."
            raise ValueError(msg)


# DEFINE NEW PARSING FUNCTIONS BELOW THIS LINE
class DefaultHistoryProcessor(HistoryProcessor):
    def __call__(self, history):
        return history


def last_n_history(history, n):
    if n <= 0:
        msg = "n must be a positive integer"
        raise ValueError(msg)
    new_history = list()
    user_messages = len([entry for entry in history if (entry["role"] == "user" and not entry.get("is_demo", False))])
    user_msg_idx = 0
    for entry in history:
        data = entry.copy()
        if data["role"] != "user":
            new_history.append(entry)
            continue
        if data.get("is_demo", False):
            new_history.append(entry)
            continue
        else:
            user_msg_idx += 1
        if user_msg_idx == 1 or user_msg_idx in range(user_messages - n + 1, user_messages + 1):
            new_history.append(entry)
        else:
            data["content"] = _omitted_content(entry.get("content"))
            new_history.append(data)
    return new_history


class LastNObservations(HistoryProcessor):
    def __init__(self, n):
        self.n = n

    def __call__(self, history):
        return last_n_history(history, self.n)


class Last2Observations(HistoryProcessor):
    def __call__(self, history):
        return last_n_history(history, 2)


class Last5Observations(HistoryProcessor):
    def __call__(self, history):
        return last_n_history(history, 5)


class ClosedWindowHistoryProcessor(HistoryProcessor):
    pattern = re.compile(r"^(\d+)\:.*?(\n|$)", re.MULTILINE)
    file_pattern = re.compile(r"\[File:\s+(.*)\s+\(\d+\s+lines\ total\)\]")

    def __call__(self, history):
        new_history = list()
        # For each value in history, keep track of which windows have been shown.
        # We want to mark windows that should stay open (they're the last window for a particular file)
        # Then we'll replace all other windows with a simple summary of the window (i.e. number of lines)
        windows = set()
        for entry in reversed(history):
            data = entry.copy()
            if data["role"] != "user":
                new_history.append(entry)
                continue
            if data.get("is_demo", False):
                new_history.append(entry)
                continue
            projected = content_text_projection(entry.get("content"))
            matches = list(self.pattern.finditer(projected))
            if len(matches) >= 1:
                file_match = self.file_pattern.search(projected)
                if file_match:
                    file = file_match.group(1)
                else:
                    continue
                if file in windows:
                    start = matches[0].start()
                    end = matches[-1].end()
                    replacement = f"Outdated window with {len(matches)} lines omitted..."
                    if isinstance(entry.get("content"), str):
                        data["content"] = projected[:start] + replacement + "\n" + projected[end:]
                    else:
                        data["content"] = _replace_text_blocks(entry.get("content"), replacement)
                windows.add(file)
            new_history.append(data)
        return list(reversed(new_history))
