# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


#
"""
Data models for Cyber-Zero framework.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # 'system', 'user', 'assistant', or 'tool'
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        if self.role not in ['system', 'user', 'assistant', 'tool']:
            raise ValueError(
                f"Invalid role: {self.role}. Must be 'system', 'user', 'assistant', or 'tool'"
            )
        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string")
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
        # Validate required fields
        if not self.task_name:
            raise ValueError("task_name is required")
        if not self.solution:
            raise ValueError("solution is required")


@dataclass
class TrajectoryData:
    """Trajectory data serialized in the agent-event JSONL format."""
    id: str
    sample_type: str
    messages: List[ConversationTurn]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryData':
        """Create TrajectoryData from either the new or legacy representation."""
        raw_messages = data.get('messages', data.get('trajectory', []))
        messages = [
            ConversationTurn(
                role=message['role'],
                content=message.get('content', ''),
                reasoning_content=message.get('reasoning_content'),
                tool_calls=message.get('tool_calls'),
            )
            for message in raw_messages
        ]
        return cls(
            id=str(data.get('id', data.get('trajectory_id', ''))),
            sample_type=data.get('sample_type', 'main'),
            messages=messages,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert TrajectoryData to the target JSONL representation."""
        messages = []
        for message in self.messages:
            item = {
                'role': message.role,
                'content': message.content,
            }
            if message.reasoning_content:
                item['reasoning_content'] = message.reasoning_content
            if message.tool_calls:
                item['tool_calls'] = message.tool_calls
            messages.append(item)
        return {
            'id': self.id,
            'sample_type': self.sample_type,
            'messages': messages,
        }


@dataclass
class EvaluationResult:
    """Result of trajectory quality evaluation."""
    trajectory_id: str
    is_high_quality: bool
    evaluation_details: Optional[str] = None
    model_used: Optional[str] = None
    num_evaluations: int = 1 