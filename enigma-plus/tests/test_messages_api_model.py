from __future__ import annotations

import unittest
from unittest.mock import patch

from sweagent.agent.models import (
    EmptyModelResponseError,
    MessagesAPIModel,
    ModelArguments,
    extract_messages_api_response,
    extract_messages_api_response_details,
)


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def fenced(command: str) -> str:
    return chr(96) * 3 + chr(10) + command + chr(10) + chr(96) * 3


class MessagesAPIResponseTests(unittest.TestCase):
    def test_structured_blocks_are_preserved_with_text_projection(self):
        details = extract_messages_api_response_details(
            {
                "content": [
                    {"type": "thinking", "thinking": "inspect"},
                    {"type": "text", "text": "DISCUSSION\n" + fenced("ls")},
                    {"type": "tool_use", "id": "call-1", "name": "Bash", "input": {"command": "ls"}},
                ],
                "stop_reason": "tool_use",
            }
        )
        self.assertEqual([block["type"] for block in details["content_blocks"]], ["thinking", "text", "tool_use"])
        self.assertIn("DISCUSSION", details["text_content"])
        self.assertEqual(details["tool_calls"][0]["id"], "call-1")

    def test_anthropic_text_content(self):
        result, reasoning = extract_messages_api_response(
            {"content": [{"type": "text", "text": "DISCUSSION" + chr(10) + fenced("ls")}]}
        )
        self.assertIn("ls", result)
        self.assertEqual(reasoning, "")

    def test_openai_compatible_content_and_reasoning(self):
        result, reasoning = extract_messages_api_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "DISCUSSION" + chr(10) + fenced("pwd"),
                            "reasoning_content": "inspect the working directory",
                        }
                    }
                ]
            }
        )
        self.assertIn("pwd", result)
        self.assertEqual(reasoning, "inspect the working directory")

    def test_structured_tool_call_accepts_dict_arguments(self):
        result, _ = extract_messages_api_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"function": {"name": "Bash", "arguments": {"command": "ls -la"}}}
                            ],
                        }
                    }
                ]
            }
        )
        self.assertEqual(result, fenced("ls -la"))

        result, _ = extract_messages_api_response(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "cat README.md"},
                    }
                ]
            }
        )
        self.assertEqual(result, fenced("cat README.md"))

    def test_empty_and_reasoning_only_responses_are_rejected(self):
        with self.assertRaises(EmptyModelResponseError):
            extract_messages_api_response({"content": []})
        with self.assertRaises(EmptyModelResponseError):
            extract_messages_api_response({"choices": [{"message": {"content": None, "reasoning": "thinking"}}]})

    def test_query_accepts_openai_compatible_gateway_response(self):
        model = MessagesAPIModel(
            ModelArguments(model_name="glm52_10", messages_api_url="http://127.0.0.1:23106/v1/messages"),
            [],
        )
        response = _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "DISCUSSION" + chr(10) + fenced("ls"),
                            "reasoning_content": "inspect",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 6},
            }
        )
        with patch("sweagent.agent.models.requests.post", return_value=response):
            result = model.query(
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                ]
            )
        self.assertIn("ls", result)
        self.assertEqual(model.stats.tokens_sent, 5)
        self.assertEqual(model.stats.tokens_received, 6)
        self.assertEqual(model.last_thought, "inspect")


if __name__ == "__main__":
    unittest.main()
