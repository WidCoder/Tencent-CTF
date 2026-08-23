# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Data models for Cyber-Zero framework."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union


ContentValue = Union[str, List[Dict[str, Any]]]


def block_to_dict(block: Any) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic SDK block or mapping to a JSON-safe mapping."""
    if isinstance(block, dict):
        return deepcopy(block)
    if hasattr(block, "model_dump"):
        value = block.model_dump()
        return deepcopy(value) if isinstance(value, dict) else None
    if hasattr(block, "dict"):
        value = block.dict()
        return deepcopy(value) if isinstance(value, dict) else None
    if hasattr(block, "__dict__"):
        return {
            key: deepcopy(value)
            for key, value in vars(block).items()
            if not key.startswith("_")
        }
    return None


def project_content_text(
    content: Any,
    content_blocks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Project structured content to text for legacy parsing and validation."""
    blocks = content_blocks
    if blocks is None and isinstance(content, list):
        blocks = [
            mapped
            for item in content
            if (mapped := block_to_dict(item)) is not None
        ]
    if blocks is not None:
        parts: List[str] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type in {"text", "output_text"}:
                text = block.get("text", "")
                if text:
                    parts.append(str(text))
            elif block_type in {"tool_use", "tool_call"}:
                arguments = block.get("input", block.get("arguments", {}))
                if isinstance(arguments, dict):
                    command = arguments.get("command") or arguments.get("cmd")
                    if command:
                        parts.append(
                            f"{chr(96)*3}bash\n{command}\n{chr(96)*3}"
                        )
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn with dual content storage."""

    role: str
    content: ContentValue
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    content_blocks: Optional[List[Dict[str, Any]]] = None
    content_text: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.role not in ["system", "user", "assistant", "tool"]:
            raise ValueError(
                f"Invalid role: {self.role}. Must be 'system', 'user', 'assistant', or 'tool'"
            )
        if not isinstance(self.content, (str, list)):
            raise ValueError("Message content must be a string or Anthropic block list")
        if isinstance(self.content, list):
            mapped_blocks = [block_to_dict(item) for item in self.content]
            if any(item is None for item in mapped_blocks):
                raise ValueError("Every content block must be a mapping or Anthropic SDK block")
            if self.content_blocks is None:
                self.content_blocks = [item for item in mapped_blocks if item is not None]
        elif self.content_blocks is not None:
            self.content_blocks = deepcopy(self.content_blocks)
        if self.content_text is None:
            self.content_text = project_content_text(self.content, self.content_blocks)
        if self.tool_calls is not None and not isinstance(self.tool_calls, list):
            raise ValueError("tool_calls must be a list of dictionaries")
        if self.tool_calls is not None and not all(isinstance(call, dict) for call in self.tool_calls):
            raise ValueError("Each tool_call must be a dictionary")


@dataclass
class TaskMeta:
    """Metadata for a CTF task."""

    task_name: str
    task_tag: str
    task_points: str
    task_description: str
    solution: str
    task_files: List[str]
    server_description: str
    writeup_path: str
    task_writeup: Optional[str] = None
    trajectory_id: int = 0

    def __post_init__(self):
        if not self.task_name:
            raise ValueError("task_name is required")
        if not self.solution:
            raise ValueError("solution is required")


@dataclass
class TrajectoryData:
    """Trajectory data supporting intermediate and public representations."""

    id: str
    sample_type: str
    messages: List[ConversationTurn]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryData":
        raw_messages = data.get("messages", data.get("trajectory", []))
        messages = [
            ConversationTurn(
                role=message["role"],
                content=message.get("content", ""),
                reasoning_content=message.get("reasoning_content"),
                tool_calls=message.get("tool_calls"),
                content_blocks=message.get("content_blocks"),
                content_text=message.get("content_text"),
                raw_response=message.get("raw_response"),
            )
            for message in raw_messages
        ]
        return cls(
            id=str(data.get("id", data.get("trajectory_id", ""))),
            sample_type=data.get("sample_type", "main"),
            messages=messages,
        )

    def _message_dict(self, message: ConversationTurn, *, preserve_blocks: bool) -> Dict[str, Any]:
        text = message.content_text or project_content_text(
            message.content, message.content_blocks
        )
        item: Dict[str, Any] = {
            "role": message.role,
            "content": deepcopy(message.content) if preserve_blocks else text,
        }
        if message.content_blocks:
            item["content_blocks"] = deepcopy(message.content_blocks)
        if message.content_text is not None:
            item["content_text"] = text
        if message.reasoning_content:
            item["reasoning_content"] = message.reasoning_content
        if message.raw_response is not None:
            item["raw_response"] = deepcopy(message.raw_response)
        if message.tool_calls:
            item["tool_calls"] = deepcopy(message.tool_calls)
        return item

    def to_intermediate_dict(self) -> Dict[str, Any]:
        """Serialize the internal artifact with structured content primary."""
        return {
            "id": self.id,
            "sample_type": self.sample_type,
            "messages": [
                self._message_dict(message, preserve_blocks=True)
                for message in self.messages
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the public target format with text plus raw blocks."""
        return {
            "id": self.id,
            "sample_type": self.sample_type,
            "messages": [
                self._message_dict(message, preserve_blocks=False)
                for message in self.messages
            ],
        }


@dataclass
class EvaluationResult:
    """Result of trajectory quality evaluation."""

    trajectory_id: str
    is_high_quality: bool
    evaluation_details: Optional[str] = None
    model_used: Optional[str] = None
    num_evaluations: int = 1
