# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


#
from __future__ import annotations

import copy
import json
import logging
import time
import yaml
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import boto3
from botocore.config import Config
import together
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic, AnthropicBedrock
from groq import Groq
from openai import AzureOpenAI, BadRequestError, OpenAI
from simple_parsing.helpers.serialization.serializable import FrozenSerializable, Serializable
from tenacity import (
    retry,
    retry_any,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sweagent.agent.commands import Command
from sweagent.utils.config import keys_config
from sweagent.utils.log import get_logger
import requests  # Add this import for HTTP requests
import re

logger = get_logger("api_models")

# Messages requests run inside Agent._query_model_with_timeout and cannot be
# force-killed when the helper thread expires. Keep transport retries bounded
# so their total budget remains below the agent-side timeout.
_MAX_RETRIES = max(1, int(keys_config.get("SWE_AGENT_MODEL_MAX_RETRIES", 10)))
_MESSAGES_API_MAX_RETRIES = max(1, min(_MAX_RETRIES, 2))
_MODEL_TIMEOUT = float(keys_config.get("SWE_AGENT_MODEL_TIMEOUT", 300))
_CONFIGURED_MESSAGES_API_TIMEOUT = float(
    keys_config.get("SWE_AGENT_MESSAGES_API_TIMEOUT", max(30.0, _MODEL_TIMEOUT - 15.0))
)
_MODEL_TIMEOUT_MARGIN = min(30.0, max(5.0, _MODEL_TIMEOUT * 0.1))
_MESSAGES_API_TIMEOUT = max(
    5.0,
    min(
        _CONFIGURED_MESSAGES_API_TIMEOUT,
        # Transport timeouts are not retried (see the retry predicate below),
        # so reserve only the model timeout margin rather than splitting the
        # budget in half. This avoids the historical 277.5s cap when the
        # model-level timeout is 600s.
        max(5.0, _MODEL_TIMEOUT - _MODEL_TIMEOUT_MARGIN),
    ),
)

# Load model configurations from YAML
def load_model_configs():
    """Load model configurations from YAML file"""
    config_path = Path(__file__).parent.parent / "models_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model_metadata(model_name: str, provider_configs: dict, shortcuts: dict, defaults: dict) -> dict:
    """Get model metadata with default values for missing fields"""
    # Check shortcuts first
    actual_model = shortcuts.get(model_name, model_name)
    
    # Get model config
    model_config = provider_configs.get(actual_model, {})
    
    # Apply defaults for missing values
    metadata = {
        'max_context': model_config.get('max_context', defaults['max_context']),
        'cost_per_input_token': model_config.get('cost_per_input_token', defaults['cost_per_input_token']),
        'cost_per_output_token': model_config.get('cost_per_output_token', defaults['cost_per_output_token']),
    }
    
    # Add optional fields if present
    if 'max_tokens' in model_config:
        metadata['max_tokens'] = model_config['max_tokens']
    elif 'max_tokens' in defaults:
        metadata['max_tokens'] = defaults['max_tokens']
    
    return metadata

def extract_thought(result: str, reasoning_content: str | None = None) -> str:
    """Normalize reasoning fields from provider responses and <think> tags."""
    import re
    if reasoning_content:
        return str(reasoning_content).strip()
    match = re.search(r"<think>(.*?)</think>", result or "", flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

class EmptyModelResponseError(RuntimeError):
    """Raised when a model gateway returns no usable assistant response."""


def _tool_call_parts(tool_call: dict[str, Any]) -> tuple[str, Any]:
    """Return a provider-independent tool name and arguments."""
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name") or tool_call.get("name") or ""
        arguments = function.get("arguments")
    else:
        name = tool_call.get("name") or ""
        arguments = tool_call.get("arguments")
    if arguments is None:
        arguments = tool_call.get("input", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"command": arguments}
    return str(name), arguments


def _tool_call_to_agent_response(tool_call: dict[str, Any]) -> str:
    """Convert a structured tool call to the CTF agent's text action format."""
    name, arguments = _tool_call_parts(tool_call)
    normalized_name = name.lower()
    if isinstance(arguments, dict):
        command = arguments.get("command") or arguments.get("cmd") or arguments.get("script")
        if normalized_name == "submit":
            command = command or arguments.get("flag") or arguments.get("submission") or arguments.get("answer")
        if command is None and normalized_name in {"bash", "shell", "run_command", "execute", "execute_command"}:
            command = arguments.get("input")
        if command is None and name:
            command = " ".join(str(value) for value in arguments.values() if value is not None)
    else:
        command = arguments
    if not name and not command:
        return ""
    if command is None or not str(command).strip():
        command = name
    elif normalized_name == "submit":
        command = f"submit {command}"
    elif normalized_name not in {"bash", "shell", "run_command", "execute", "execute_command", "command", "tool"}:
        command = f"{name} {command}"
    fence = chr(96) * 3
    return fence + chr(10) + str(command).strip() + chr(10) + fence


def _tool_call_to_shell_action(tool_call: dict[str, Any]) -> str:
    """Extract the executable shell action from a native Anthropic tool_use."""
    name, arguments = _tool_call_parts(tool_call)
    if isinstance(arguments, dict):
        command = arguments.get("command") or arguments.get("cmd") or arguments.get("script")
        if command is None and name.lower() == "submit":
            command = arguments.get("flag") or arguments.get("submission") or arguments.get("answer")
    else:
        command = arguments
    command = str(command or "").strip()
    if name.lower() == "submit" and command and not command.startswith("submit "):
        command = f"submit {command}"
    return command


def _request_assistant_blocks(blocks: Any, fallback: Any = "") -> Any:
    """Project a provider response to blocks safe to replay on the next turn.

    GLM-compatible Messages gateways commonly return hidden ``thinking``
    blocks.  Those blocks are response metadata, not a portable assistant
    message: replaying them together with ``tool_use`` can make the gateway
    wait indefinitely on the next request.  Keep them in the trajectory, but
    omit them from the request history.  If no executable/text block remains,
    use the compatibility text projection instead.
    """
    if not isinstance(blocks, list):
        return fallback
    kept: list[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            kept.append(copy.deepcopy(block))
            continue
        block_type = str(block.get("type", "")).lower()
        if block_type in {"thinking", "reasoning"}:
            continue
        kept.append(copy.deepcopy(block))
    return kept if kept else fallback


def _messages_payload_shape(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-sensitive diagnostics for a Messages payload."""
    shape: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            block_types = [
                str(block.get("type", "unknown"))
                for block in content
                if isinstance(block, dict)
            ]
            chars = sum(
                len(str(block.get(key, "")))
                for block in content
                if isinstance(block, dict)
                for key in ("text", "thinking", "content")
                if block.get(key) is not None
            )
            shape.append({"role": message.get("role"), "blocks": block_types, "chars": chars})
        else:
            shape.append({"role": message.get("role"), "type": type(content).__name__, "chars": len(str(content))})
    return shape


def extract_messages_api_response_details(data: dict[str, Any]) -> dict[str, Any]:
    """Extract a Messages response without discarding its structured blocks.

    ``content_text`` is the compatibility projection consumed by the existing
    action parser. ``content_blocks`` and ``raw_response`` remain structured
    for trajectory persistence.
    """
    if not isinstance(data, dict):
        raise EmptyModelResponseError("Messages API returned a non-object JSON response")

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    content_blocks: list[dict[str, Any]] = []

    def consume_content(content: Any) -> None:
        if isinstance(content, str):
            text_parts.append(content)
            content_blocks.append({"type": "text", "text": content})
        elif isinstance(content, dict):
            block = copy.deepcopy(content)
            block_type = str(block.get("type", "")).lower()
            if block_type in {"thinking", "reasoning"}:
                value = block.get("thinking") or block.get("reasoning_content") or block.get("text")
                if value:
                    reasoning_parts.append(str(value))
                content_blocks.append(block)
            elif block_type in {"tool_use", "tool_call", "function"}:
                content_blocks.append(block)
                tool_calls.append(block)
            else:
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    text_parts.append(value)
                content_blocks.append(block)
        elif isinstance(content, list):
            for block in content:
                consume_content(block)

    content = data.get("content")
    if content is not None:
        consume_content(content)

    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        if data.get("stop_reason") is None and choice.get("stop_reason") is not None:
            data = dict(data)
            data["stop_reason"] = choice.get("stop_reason")
        message = choice.get("message") or choice.get("delta") or {}
        if isinstance(message, dict):
            consume_content(message.get("content"))
            message_reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("thinking")
            if message_reasoning:
                reasoning_parts.append(str(message_reasoning))
                content_blocks.append({"type": "thinking", "thinking": str(message_reasoning)})
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        call_copy = copy.deepcopy(call)
                        tool_calls.append(call_copy)
                        content_blocks.append(call_copy)
        top_level_calls = choice.get("tool_calls")
        if isinstance(top_level_calls, list):
            for call in top_level_calls:
                if isinstance(call, dict):
                    call_copy = copy.deepcopy(call)
                    tool_calls.append(call_copy)
                    content_blocks.append(call_copy)

    output = data.get("output")
    if output is not None:
        consume_content(output)
    for key in ("output_text", "text"):
        value = data.get(key)
        if isinstance(value, str):
            text_parts.append(value)
            content_blocks.append({"type": "text", "text": value})
    top_level_reasoning = data.get("reasoning_content") or data.get("reasoning") or data.get("thinking")
    if top_level_reasoning:
        reasoning_parts.append(str(top_level_reasoning))
        content_blocks.append({"type": "thinking", "thinking": str(top_level_reasoning)})

    text_content = chr(10).join(part for part in text_parts if part)
    action_text = text_content
    if tool_calls:
        if len(tool_calls) > 1:
            logger.warning("Messages API returned %d tool calls; executing only the first", len(tool_calls))
        tool_response = _tool_call_to_agent_response(tool_calls[0])
        if tool_response:
            action_text = chr(10).join(part for part in [text_content, tool_response] if part)

    result = action_text
    if not result.strip():
        keys = sorted(str(key) for key in data.keys())
        block_types = [str(block.get("type", "unknown")) for block in content_blocks]
        logger.warning(
            "Messages API empty response: keys=%s block_types=%s stop_reason=%s usage=%s",
            keys,
            block_types,
            data.get("stop_reason"),
            data.get("usage", {}),
        )
        raise EmptyModelResponseError("Messages API returned no usable text or tool call")
    return {
        "content_blocks": content_blocks,
        "content_text": result,
        "text_content": text_content,
        "tool_calls": copy.deepcopy(tool_calls),
        "reasoning": chr(10).join(reasoning_parts).strip(),
        "stop_reason": data.get("stop_reason"),
        "usage": copy.deepcopy(data.get("usage")) if isinstance(data.get("usage"), dict) else {},
        "raw_response": copy.deepcopy(data),
    }


def _is_thinking_only_truncated_response(data: Any) -> bool:
    """Identify a response exhausted by hidden reasoning without an action."""
    if not isinstance(data, dict):
        return False
    stop_reason = data.get("stop_reason")
    if stop_reason is None and isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            stop_reason = choice.get("stop_reason")
    if stop_reason != "max_tokens":
        return False
    has_thinking = False
    has_action = False

    def inspect(content: Any) -> None:
        nonlocal has_thinking, has_action
        if isinstance(content, list):
            for block in content:
                inspect(block)
        elif isinstance(content, dict):
            block_type = str(content.get("type", "")).lower()
            if block_type in {"thinking", "reasoning"} or any(
                content.get(key) for key in ("thinking", "reasoning", "reasoning_content")
            ):
                has_thinking = True
            if block_type in {"text", "tool_use", "tool_call", "function"}:
                value = content.get("text") or content.get("input") or content.get("arguments")
                has_action = bool(value)
        elif isinstance(content, str) and content.strip():
            has_action = True

    inspect(data.get("content"))
    for choice in data.get("choices", []) if isinstance(data.get("choices"), list) else []:
        if isinstance(choice, dict):
            message = choice.get("message") or choice.get("delta") or {}
            if isinstance(message, dict):
                inspect(message.get("content"))
                has_action = has_action or bool(message.get("tool_calls"))
                has_thinking = has_thinking or bool(
                    message.get("thinking") or message.get("reasoning") or message.get("reasoning_content")
                )
    return has_thinking and not has_action


def extract_messages_api_response(data: dict[str, Any]) -> tuple[str, str]:
    """Backward-compatible text/reasoning projection of a Messages response."""
    details = extract_messages_api_response_details(data)
    return details["content_text"], details["reasoning"]


def clean_result(result):
    # First, split on </think> and take everything after the first one (if any)
    if "</think>" in result:
        content = " ".join(result.split("</think>")[1:])
    else:
        content = result
    content = content.split("<|im_end|>")[0]
    
    # print(f"Content: {result}")
    # exit()
    # # Now, remove all <|...|> patterns including Unicode variants
    import re
    # # Remove all <|...|> patterns - this pattern matches < followed by any pipe-like character, then any content, then pipe-like character and >
    
    # Also remove specific tool call patterns
    tool_patterns = [
        r"<锝渢ool鈻乧all鈻乥egin锝?.*?<锝渢ool鈻乧all鈻乪nd锝?",
        r"<锝渢ool鈻乧alls鈻乥egin锝?.*?<锝渢ool鈻乧alls鈻乪nd锝?",
    ]
    # Use a loop to handle nested patterns
    for pattern in tool_patterns:
        while re.search(pattern, content, flags=re.DOTALL):
            content = re.sub(pattern, "", content, flags=re.DOTALL)

    content = content.replace("<锝渢ool鈻乧all鈻乥egin锝?", "").replace("<锝渢ool鈻乧all鈻乪nd锝?", "").replace("<锝渢ool鈻乧alls鈻乥egin锝?", "").replace("<锝渢ool鈻乧alls鈻乪nd锝?", "")
    
    return content.strip()

@dataclass(frozen=True)
class ModelArguments(FrozenSerializable):
    """Arguments configuring the model and its behavior."""

    # Name of the model to use
    model_name: str
    # Cost limit for every instance (task)
    per_instance_cost_limit: float = 0.0
    # Total cost limit
    total_cost_limit: float = 0.0
    # Sampling temperature
    temperature: float = 0.0
    # Sampling top-p
    top_p: float = 1.0
    # Sampling top-k
    top_k: int = 20
    # Path to replay file when using the replay model
    replay_path: str | None = None
    # Host URL when using Ollama model
    host_url: str = "localhost:11434"
    # Anthropic Messages-compatible endpoint for local gateways.
    messages_api_url: str = ""
    messages_api_key: str = "EMPTY"
    # Maximum number of steps (environment interactions) per instance (0 = unlimited)
    per_instance_step_limit: int = 0


@dataclass
class APIStats(Serializable):
    total_cost: float = 0
    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other):
        if not isinstance(other, APIStats):
            msg = "Can only add APIStats with APIStats"
            raise TypeError(msg)

        return APIStats(
            **{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)},
        )

    def replace(self, other):
        if not isinstance(other, APIStats):
            msg = "Can only replace APIStats with APIStats"
            raise TypeError(msg)

        return APIStats(**{field.name: getattr(other, field.name) for field in fields(self)})


class ContextWindowExceededError(Exception):
    pass


class CostLimitExceededError(Exception):
    pass


class BaseModel:
    def __init__(self, args: ModelArguments, commands: list[Command]):
        self.args = args
        self.commands = commands
        self.model_metadata = {}
        self.stats = APIStats()
        self.last_thought = ""
        # Structured response metadata for trajectory persistence.  The model
        # still returns a text projection because the existing parser expects it.
        self.last_content_blocks: list[dict[str, Any]] = []
        self.last_content_text = ""
        self.last_text_content = ""
        self.last_raw_response: dict[str, Any] = {}
        self.last_stop_reason: str | None = None
        self.last_usage: dict[str, Any] = {}
        self.last_tool_calls: list[dict[str, Any]] = []
        # Load configurations from YAML
        configs = load_model_configs()
        defaults = configs['defaults']
        
        # Get provider-specific configs and shortcuts
        provider_configs, shortcuts = self._get_provider_configs(configs)
        
        # Map `model_name` to API-compatible name `api_model`
        self.api_model = shortcuts.get(self.args.model_name, self.args.model_name)

        # Handle special model name prefixes
        if args.model_name.startswith("ft:"):
            ft_model = args.model_name.split(":")[1]
            self.model_metadata = get_model_metadata(ft_model, provider_configs, shortcuts, defaults)
        elif args.model_name.startswith("ollama:"):
            self.api_model = args.model_name.split("ollama:", 1)[1]
            # Ollama models use default metadata
            self.model_metadata = get_model_metadata(self.api_model, {}, {}, defaults)
        elif args.model_name.startswith("azure:"):
            azure_model = args.model_name.split("azure:", 1)[1]
            self.model_metadata = get_model_metadata(azure_model, provider_configs, shortcuts, defaults)
        elif args.model_name.startswith("bedrock:"):
            self.api_model = args.model_name.split("bedrock:", 1)[1]
            bedrock_configs = configs.get('bedrock_models', {})
            self.model_metadata = get_model_metadata(self.api_model, bedrock_configs, {}, defaults)
        elif args.model_name.startswith("groq:"):
            self.api_model = args.model_name.split("groq:", 1)[1]
            groq_configs = configs.get('groq_models', {})
            groq_shortcuts = configs.get('groq_shortcuts', {})
            self.model_metadata = get_model_metadata(self.api_model, groq_configs, groq_shortcuts, defaults)
        elif args.model_name.startswith("vllm:"):
            # VLLM models use default metadata
            self.model_metadata = get_model_metadata(self.args.model_name, {}, {}, defaults)
        else:
            # Try to find model in any provider configs
            self.model_metadata = get_model_metadata(args.model_name, provider_configs, shortcuts, defaults)
            
            # If model not found anywhere, check special models
            if not any(key in self.model_metadata for key in ['max_context']) or self.model_metadata.get('max_context') == defaults['max_context']:
                special_configs = configs.get('special_models', {})
                if args.model_name in special_configs:
                    self.model_metadata = get_model_metadata(args.model_name, special_configs, {}, defaults)
                elif self.api_model not in provider_configs and args.model_name not in shortcuts:
                    msg = f"Unregistered model ({args.model_name}). Add model to models_config.yaml"
                    logger.warning(msg)
                    # Use defaults for unknown models
                    self.model_metadata = defaults.copy()

    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        """Get the appropriate provider configs and shortcuts based on model class"""
        # This method should be overridden by subclasses to return the right configs
        return {}, {}

    def reset_stats(self, other: APIStats | None = None):
        if other is None:
            self.stats = APIStats(total_cost=self.stats.total_cost)
            logger.info("Resetting model stats")
        else:
            # Make sure to copy the stats to avoid modifying the original
            self.stats = copy.deepcopy(other)

    def update_stats(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the cost of a response from the openai API.

        Args:
        input_tokens (int): The number of tokens in the prompt.
        output_tokens (int): The number of tokens in the response.

        Returns:
        float: The cost of the response.
        """
        # Calculate cost and update cost related fields
        cost = (
            self.model_metadata.get("cost_per_input_token", 0.0) * input_tokens
            + self.model_metadata.get("cost_per_output_token", 0.0) * output_tokens
        )
        self.stats.total_cost += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        # Log updated cost values to std. err
        logger.debug(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.instance_cost:.2f}, "
            f"cost={cost:.2f}",
        )
        logger.debug(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={self.stats.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )

        # Check whether total cost or instance cost limits have been exceeded
        if 0 < self.args.total_cost_limit <= self.stats.total_cost:
            logger.warning(f"Cost {self.stats.total_cost:.2f} exceeds limit {self.args.total_cost_limit:.2f}")
            msg = "Total cost limit exceeded"
            raise CostLimitExceededError(msg)

        if 0 < self.args.per_instance_cost_limit <= self.stats.instance_cost:
            logger.warning(f"Cost {self.stats.instance_cost:.2f} exceeds limit {self.args.per_instance_cost_limit:.2f}")
            msg = "Instance cost limit exceeded"
            raise CostLimitExceededError(msg)
        return cost

    def query(self, history: list[dict[str, str]]) -> str:
        msg = "Use a subclass of BaseModel"
        raise NotImplementedError(msg)


class OpenAIModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return configs.get('openai_models', {}), configs.get('openai_shortcuts', {})

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self._setup_client()
        # Track all previous responses to detect duplicates
        self.previous_responses = []

    def _setup_client(self):
        if self.args.model_name.startswith("azure"):
            logger.warning(
                "The --model CLI argument is ignored when using the Azure GPT endpoint. "
                "The model is determined by the AZURE_OPENAI_DEPLOYMENT key/"
                "environment variable (this might change in the future).",
            )
            self.api_model = keys_config["AZURE_OPENAI_DEPLOYMENT"]
            self.client = AzureOpenAI(
                api_key=keys_config["AZURE_OPENAI_API_KEY"],
                azure_endpoint=keys_config["AZURE_OPENAI_ENDPOINT"],
                api_version=keys_config.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
        else:
            api_base_url: str | None = keys_config.get("OPENAI_API_BASE_URL", None)
            self.client = OpenAI(api_key=keys_config["OPENAI_API_KEY"], base_url=api_base_url)

    def history_to_messages(
        self,
        history: list[dict[str, str]],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        """
        Create `messages` by filtering out all keys except for role/content per `history` turn
        """
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return "\n".join([entry["content"] for entry in history])
        # Return history components with just role, content fields
        return [{k: v for k, v in entry.items() if k in ["role", "content"]} for entry in history]

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type((CostLimitExceededError, RuntimeError)),
    )
    def query(self, history: list[dict[str, str]]) -> str:
        """
        Query the OpenAI API with the given `history` and return the response.
        """
        max_resample_attempts = 10
        resample_count = 0
        
        while resample_count < max_resample_attempts:
            try:
                # Perform OpenAI API call
                response = self.client.chat.completions.create(
                    messages=self.history_to_messages(history),
                    model=self.api_model,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                )
                break
            except BadRequestError as e:
                logger.exception("BadRequestError")
                if "context window" in str(e) or getattr(e, "error", {}).get("code") == "context_length_exceeded":
                    msg = f"Context window ({self.model_metadata.get('max_context', 'unknown')} tokens) exceeded"
                    raise ContextWindowExceededError(msg) from e
                else:
                    raise e
            
        # Calculate + update costs, get response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        self.update_stats(input_tokens, output_tokens)
        raw_content = response.choices[0].message.content or ""
        reasoning = getattr(response.choices[0].message, "reasoning_content", None) or getattr(response.choices[0].message, "thinking", None)
        self.last_thought = extract_thought(raw_content, reasoning)
        current_response = clean_result(raw_content)
        
        # Store this response for future comparison
        self.previous_responses.append(current_response.strip())
        return current_response


class DeepSeekModel(OpenAIModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return configs.get('deepseek_models', {}), {}

    def _setup_client(self) -> None:
        api_base_url: str = keys_config["DEEPSEEK_API_BASE_URL"]
        self.client = OpenAI(api_key=keys_config["DEEPSEEK_API_KEY"], base_url=api_base_url)


class GroqModel(OpenAIModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return configs.get('groq_models', {}), configs.get('groq_shortcuts', {})

    def _setup_client(self) -> None:
        self.client = Groq(
            api_key=keys_config["GROQ_API_KEY"],
        )


class MessagesAPIModel(BaseModel):
    """Local Anthropic Messages-compatible model, such as a GLM gateway."""

    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return configs.get("messages_api_models", {}), {}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)
        self.api_url = args.messages_api_url.rstrip("/")
        if not self.api_url:
            raise ValueError("--messages_api_url is required for glm52_* models")
        self.api_key = args.messages_api_key or "EMPTY"
        self.request_timeout = _MESSAGES_API_TIMEOUT
        self.max_output_tokens = max(256, int(keys_config.get("SWE_AGENT_MESSAGES_MAX_TOKENS", 8192)))
        # The upstream Cyber-Zero request path uses plain role/content
        # messages. Native tool blocks are opt-in because some gateways accept
        # the request but stall while reasoning over them.
        self.native_tools = str(
            keys_config.get("SWE_AGENT_MESSAGES_NATIVE_TOOLS", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.thinking_mode = str(
            keys_config.get("SWE_AGENT_MESSAGES_THINKING", "auto")
        ).strip().lower()

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        reraise=True,
        stop=stop_after_attempt(_MESSAGES_API_MAX_RETRIES),
        retry=retry_any(
            retry_if_exception_type(EmptyModelResponseError),
        # A transport timeout means the gateway did not finish the request;
        # retrying immediately can duplicate an expensive GLM generation and
        # make the next request time out as well. Surface it as a terminal
        # model_timeout instead.
        retry_if_not_exception_type((CostLimitExceededError, RuntimeError, TimeoutError)),
        ),
    )
    def query(self, history: list[dict[str, str]]) -> str:
        system = chr(10).join(entry["content"] for entry in history if entry["role"] == "system")
        if self.native_tools:
            # Preserve provider-native blocks only when explicitly requested.
            messages = []
            for entry in history:
                role = entry.get("role")
                if role == "system":
                    continue
                blocks = entry.get("content_blocks")
                if role == "assistant" and isinstance(blocks, list) and blocks:
                    content = _request_assistant_blocks(blocks, entry.get("content", ""))
                else:
                    content = entry.get("content", "")
                messages.append({"role": role, "content": content})
        else:
            # Match the upstream request shape.  Structured blocks are still
            # persisted in trajectories, but are projected to text for the
            # model gateway so old GLM deployments see the same protocol.
            flattened: list[dict[str, str]] = []
            for entry in history:
                role = entry.get("role")
                if role == "system":
                    continue
                content = entry.get("content", "")
                if isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if not isinstance(block, dict):
                            parts.append(str(block))
                            continue
                        block_type = str(block.get("type", "")).lower()
                        if block_type == "tool_result":
                            parts.append(str(block.get("content", "")))
                        elif block_type == "text":
                            parts.append(str(block.get("text", "")))
                        elif block_type in {"thinking", "reasoning"}:
                            parts.append(str(block.get("thinking") or block.get("reasoning_content") or ""))
                        else:
                            parts.append(str(block.get("text") or block.get("content") or ""))
                    content = "\n".join(part for part in parts if part)
                flattened.append({"role": str(role), "content": str(content)})
            messages = anthropic_history_to_messages(self, [
                {"role": "system", "content": system}, *flattened
            ]) if system else anthropic_history_to_messages(self, flattened)
        payload = {
            "model": self.api_model,
            "messages": messages,
            "max_tokens": min(self.model_metadata.get("max_tokens", 8192), self.max_output_tokens),
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        if self.native_tools:
            payload["tools"] = [{
                "name": "Bash",
                "description": "Execute a shell command in the task container.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }]
        # Some gateways emit hidden reasoning by default and then reject or
        # stall when that response metadata is replayed.  Keep the default
        # provider behavior (``auto``), but allow a deployment-specific,
        # explicit switch for A/B tests and production runs.
        if self.thinking_mode in {"disabled", "disable", "off", "false", "0"}:
            payload["thinking"] = {"type": "disabled"}
        elif self.thinking_mode.startswith("budget:"):
            try:
                budget = max(16, int(self.thinking_mode.split(":", 1)[1]))
            except (TypeError, ValueError):
                budget = 256
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if system:
            payload["system"] = system
        request_started = time.monotonic()
        logger.info(
            "Messages API request start: model=%s messages=%d timeout=%.1fs max_tokens=%s native_tools=%s thinking=%s shape=%s",
            self.api_model, len(messages), self.request_timeout, payload["max_tokens"],
            self.native_tools, self.thinking_mode, _messages_payload_shape(messages),
        )
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "x-api-key": self.api_key,
                    "Authorization": f"Bearer {self.api_key}",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=self.request_timeout,
            )
        except requests.exceptions.Timeout as error:
            # Normalize urllib3/requests read and connect timeouts so the
            # Agent can persist a model_timeout terminal state instead of
            # misclassifying the task as runner_exception.
            raise TimeoutError(
                f"Messages API request timed out after {self.request_timeout:.1f}s"
            ) from error
        response.raise_for_status()
        logger.info(
            "Messages API response received: status=%s elapsed=%.1fs",
            getattr(response, "status_code", "unknown"), time.monotonic() - request_started,
        )
        data = response.json()
        self.last_raw_response = copy.deepcopy(data) if isinstance(data, dict) else {}
        self.last_stop_reason = data.get("stop_reason") if isinstance(data, dict) else None
        try:
            details = extract_messages_api_response_details(data)
        except EmptyModelResponseError:
            # Some gateways spend the entire budget in hidden reasoning and
            # return ``max_tokens`` without an executable block. Retry once
            # with reasoning disabled so the step cannot get stuck in a
            # thinking-only loop.
            if _is_thinking_only_truncated_response(data) and payload.get("thinking", {}).get("type") != "disabled":
                retry_payload = dict(payload)
                retry_payload["thinking"] = {"type": "disabled"}
                logger.warning("Messages API returned thinking-only max_tokens; retrying with thinking disabled")
                retry_response = requests.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "Authorization": f"Bearer {self.api_key}",
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=retry_payload,
                    timeout=self.request_timeout,
                )
                retry_response.raise_for_status()
                data = retry_response.json()
                self.last_raw_response = copy.deepcopy(data) if isinstance(data, dict) else {}
                self.last_stop_reason = data.get("stop_reason") if isinstance(data, dict) else None
                details = extract_messages_api_response_details(data)
            else:
                raise
        usage = details["usage"]
        self.update_stats(
            int(usage.get("input_tokens", usage.get("prompt_tokens", 0))),
            int(usage.get("output_tokens", usage.get("completion_tokens", 0))),
        )
        self.last_content_blocks = copy.deepcopy(details["content_blocks"])
        self.last_raw_response = copy.deepcopy(details["raw_response"])
        self.last_stop_reason = details["stop_reason"]
        self.last_usage = copy.deepcopy(usage)
        self.last_tool_calls = copy.deepcopy(details.get("tool_calls", []))
        self.last_thought = details["reasoning"] or extract_thought(details["content_text"])
        cleaned = clean_result(details["content_text"])
        self.last_content_text = cleaned
        self.last_text_content = details.get("text_content", "")
        if not cleaned.strip():
            raise EmptyModelResponseError("Messages API response became empty after cleanup")
        return cleaned

class AnthropicModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return configs.get('anthropic_models', {}), configs.get('anthropic_shortcuts', {})

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        # Set Anthropic key
        self.api = Anthropic(api_key=keys_config["ANTHROPIC_API_KEY"])

    def history_to_messages(
        self,
        history: list[dict[str, str]],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        """
        Create `prompt` by filtering out all keys except for role/content per `history` turn
        Reference: https://docs.anthropic.com/claude/reference/complete_post
        """
        return anthropic_history_to_messages(self, history, is_demonstration)

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type((CostLimitExceededError, RuntimeError)),
    )
    def query(self, history: list[dict[str, str]]) -> str:
        """
        Query the Anthropic API with the given `history` and return the response.
        """
        return anthropic_query(self, history)


class BedrockModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return {}, {}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        # Extract provider from model ID
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
        self.model_provider = self.api_model.split(".")[0]
        if self.model_provider == "anthropic":
            # Note: this assumes AWS credentials are already configured.
            # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
            self.api = AnthropicBedrock()
        elif self.model_provider == "us":
            # For DeepSeek models, use native Bedrock client
            config = Config(
                retries={
                    "max_attempts": 100,
                    "mode": "standard"
                }
            )
            self.api = boto3.client('bedrock-runtime', config=config, region_name='us-west-2')
        elif self.model_provider in ["ai21", "amazon", "cohere", "meta", "mistral"]:
            msg = f"{self.api_model} is not supported!"
            raise NotImplementedError(msg)
        else:
            msg = f"Provider {self.model_provider} is not supported by Amazon Bedrock!"
            raise ValueError(msg)

    def history_to_messages(
        self,
        history: list[dict[str, str]],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        """
        Create `prompt` from the history of messages
        """
        if self.model_provider == "anthropic":
            return anthropic_history_to_messages(self, history, is_demonstration)
        elif self.model_provider == "us":
            # For DeepSeek models, return messages in standard format
            if is_demonstration:
                history = [entry for entry in history if entry["role"] != "system"]
                return "\n".join([entry["content"] for entry in history])
            return [{k: v for k, v in entry.items() if k in ["role", "content"]} for entry in history]
        else:
            msg = f"{self.api_model} is not supported!"
            raise NotImplementedError(msg)

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type((CostLimitExceededError, RuntimeError)),
    )
    def query(self, history: list[dict[str, str]]) -> str:
        """
        Query Amazon Bedrock with the given `history` and return the response.
        """
        if self.model_provider == "anthropic":
            return anthropic_query(self, history)
        elif self.model_provider == "us":
            for _ in range(5):
                response = deepseek_query(self, history)
                if response:
                    return response
            
            msg = f"{self.api_model} is not supported!"
            raise NotImplementedError(msg)


def deepseek_query(model: BedrockModel, history: list[dict[str, str]]) -> str:
    """
    Query DeepSeek models via Amazon Bedrock with the given `history` and return the response.
    """
    # Get system message(s) and user messages
    system_message = "\n".join([entry["content"] for entry in history if entry["role"] == "system"])
    
    # Convert messages to Bedrock format
    messages = []
    for entry in history:
        if entry["role"] != "system":  # Skip system messages as they're handled separately
            # Ensure content is not empty
            content = entry.get("content", "").strip()
            if content:  # Only add non-empty messages
                messages.append({
                    "role": entry["role"],
                    "content": [{"text": content}]
                })
    
    # Ensure we have at least one message
    if not messages:
        # If no messages, add a default user message
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    
    # Prepare system prompts - only include if there's a system message
    system_prompts = [{"text": system_message}] if system_message.strip() else None

    # Configure inference parameters
    inference_config = {
        "temperature": max(0.0, min(1.0, model.args.temperature)),  # Clamp temperature between 0 and 1
        "maxTokens": model.model_metadata.get("max_tokens", 4096),  # Ensure maxTokens doesn't exceed limits
    }
    
    # Add top_p if it's not the default value
    if model.args.top_p != 1.0:
        inference_config["topP"] = max(0.0, min(1.0, model.args.top_p))  # Clamp topP between 0 and 1
    
    # Prepare converse parameters
    converse_params = {
        "modelId": model.api_model,
        "messages": messages,
        "inferenceConfig": inference_config,
    }
    
    # Only add system prompts if they exist
    if system_prompts:
        converse_params["system"] = system_prompts
    
    # Perform Bedrock API call using converse method
    response = model.api.converse(**converse_params)
    
    # Extract the response content
    output_message = response["output"]["message"]
    response_text = ""
    reasoning_parts = []
    # Handle reasoning content and regular content
    for content in output_message["content"]:
        if content.get("reasoningContent"):
            reasoning = content.get("reasoningContent")
            if isinstance(reasoning, dict):
                reasoning_parts.append(reasoning.get("text", reasoning.get("data", "")))
            else:
                reasoning_parts.append(str(reasoning))
            continue
        else:
            response_text = content["text"].split("(Open file:")[0].strip()
            break

    model.last_thought = "\n".join(str(x) for x in reasoning_parts if x).strip()

    # Calculate token usage for cost tracking
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    
    # Update stats and return response
    if response_text:
        model.update_stats(input_tokens, output_tokens)
        return response_text


class OllamaModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return configs.get('ollama_models', {}), configs.get('ollama_shortcuts', {})

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)
        from ollama import Client

        self.client = Client(host=args.host_url)

    def history_to_messages(
        self,
        history: list[dict[str, str]],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        """
        Create `messages` by filtering out all keys except for role/content per `history` turn
        """
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return "\n".join([entry["content"] for entry in history])
        # Return history components with just role, content fields
        return [{k: v for k, v in entry.items() if k in ["role", "content"]} for entry in history]

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type((CostLimitExceededError, RuntimeError)),
    )
    def query(self, history: list[dict[str, str]]) -> str:
        """
        Query the Ollama API with the given `history` and return the response.
        """
        response = self.client.chat(
            model=self.api_model,
            messages=self.history_to_messages(history),
            options={
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
            },
        )
        # Calculate + update costs, return response
        if "prompt_eval_count" in response:
            input_tokens = response["prompt_eval_count"]
        else:
            logger.warning(
                "Prompt eval count not found in response. Using 0. "
                "This might be because the prompt has been cached. "
                "See https://github.com/swe-agent/SWE-agent/issues/44 "
                "and https://github.com/ollama/ollama/issues/3427.",
            )
            input_tokens = 0
        output_tokens = response["eval_count"]
        self.update_stats(input_tokens, output_tokens)
        return response["message"]["content"]


class TogetherModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return configs.get('together_models', {}), configs.get('together_shortcuts', {})

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)
        assert together.version >= "1.1.0", "Please upgrade to Together SDK v1.1.0 or later."

        # Set Together key
        together.api_key = keys_config["TOGETHER_API_KEY"]

    def history_to_messages(self, history: list[dict[str, str]], is_demonstration: bool = False) -> str:
        """
        Create `prompt` by filtering out all keys except for role/content per `history` turn
        """
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
        # Map history to TogetherAI format
        mapping = {"user": "human", "assistant": "bot", "system": "bot"}
        prompt = [f'<{mapping[d["role"]]}>: {d["content"]}' for d in history]
        prompt = "\n".join(prompt)
        return f"{prompt}\n<bot>:"

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type((CostLimitExceededError, RuntimeError)),
    )
    def query(self, history: list[dict[str, str]]) -> str:
        """
        Query the Together API with the given `history` and return the response.
        """
        # Perform Together API call
        prompt = self.history_to_messages(history)
        # Anthropic's count_tokens is convenient because it caches and utilizes huggingface/tokenizers, so we will use.
        max_tokens_to_sample = self.model_metadata.get("max_context", 32768) - Anthropic().count_tokens(prompt)
        completion = together.Complete.create(
            model=self.api_model,
            prompt=prompt,
            max_tokens=max_tokens_to_sample,
            stop=["<human>"],
            temperature=self.args.temperature,
            top_p=self.args.top_p,
        )
        # Calculate + update costs, return response
        response = completion["choices"][0]["text"].split("<human>")[0]
        input_tokens = completion["usage"]["prompt_tokens"]
        output_tokens = completion["usage"]["completion_tokens"]
        self.update_stats(input_tokens, output_tokens)
        return response


class HumanModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return {}, {}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        # Determine which commands require multi-line input
        self.multi_line_command_endings = {
            command.name: command.end_name for command in commands if command.end_name is not None
        }

    def history_to_messages(
        self,
        history: list[dict[str, str]],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        """
        Create `messages` by filtering out all keys except for role/content per `history` turn
        """
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return "\n".join([entry["content"] for entry in history])
        # Return history components with just role, content fields
        return [{k: v for k, v in entry.items() if k in ["role", "content"]} for entry in history]

    def query(self, history: list[dict[str, str]], action_prompt: str = "> ") -> str:
        """
        Logic for handling user input to pass to SWEEnv
        """
        action = input(action_prompt)
        command_name = action.split()[0] if action.strip() else ""

        # Special handling for multi-line input actions (i.e. edit)
        if command_name in self.multi_line_command_endings:
            buffer = [action]
            end_keyword = self.multi_line_command_endings[command_name]
            while True:
                action = input("... ")
                buffer.append(action)
                if action.rstrip() == end_keyword:
                    # Continue reading input until terminating keyword inputted
                    break
            action = "\n".join(buffer)
        elif action.strip() == "start_multiline_command":  # do arbitrary multi-line input
            buffer = []
            while True:
                action = input("... ")
                if action.rstrip() == "end_multiline_command":
                    break
                buffer.append(action)
            action = "\n".join(buffer)
        return action


class HumanThoughtModel(HumanModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return {}, {}

    def query(self, history: list[dict[str, str]]) -> str:
        """
        Logic for handling user input (both thought + action) to pass to SWEEnv
        """
        thought_all = ""
        thought = input("Thought (end w/ END_THOUGHT): ")
        while True:
            if "END_THOUGHT" in thought:
                thought = thought.split("END_THOUGHT")[0]
                thought_all += thought
                break
            thought_all += thought
            thought = input("... ")

        action = super().query(history, action_prompt="Action: ")

        return f"{thought_all}\n```\n{action}\n```"


class ReplayModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return {}, {}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        if self.args.replay_path is None or not Path(self.args.replay_path).exists():
            msg = "--replay_path must point to a file that exists to run a replay policy"
            raise ValueError(msg)

        self.replays = [
            list(json.loads(x).values())[0] for x in Path(self.args.replay_path).read_text().splitlines(keepends=True)
        ]
        self.replay_idx = 0
        self.action_idx = 0

    def _next_replay(self) -> None:
        """Called after last action"""
        self.replay_idx += 1
        self.action_idx = 0

    def query(self, history: list[dict[str, str]]) -> str:
        """
        Logic for tracking which replay action to pass to SWEEnv
        """
        actions = self.replays[self.replay_idx]
        try:
            action = actions[self.action_idx]
        except IndexError:
            msg = (
                "This seems to be an incomplete trajectory. "
                "We reached the end of it, but `submit` was not called. "
                "Calling it now."
            )
            logger.warning(msg)
            action = "```\nsubmit\n```"

        self.action_idx += 1

        # Assuming `submit` is always last action of replay trajectory
        if action == "submit":
            self._next_replay()

        return action


class InstantEmptySubmitTestModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return {}, {}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        """This model immediately submits. Useful for testing purposes"""
        super().__init__(args, commands)
        self._action_idx = 0

    def query(self, history: list[dict[str, str]]) -> str:
        # Need to at least do _something_ to submit
        if self._action_idx == 0:
            self._action_idx = 1
            action = "DISCUSSION\nLet's reproduce the bug by creating a `reproduce.py` file.\n\n```\ncreate reproduce.py\n```\n"
        elif self._action_idx == 1:
            self._action_idx = 0
            action = "DISCUSSION\nThe task should be resolved, so let's submit the patch.\n\n```\nsubmit\n```\n"
        self.update_stats(0, 0)
        return action


class VLLMModel(BaseModel):
    def _get_provider_configs(self, configs: dict) -> tuple[dict, dict]:
        return {}, {}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        # Parse model name and host
        if ":" in args.model_name:
            # e.g. vllm:Qwen/Qwen3-32B
            _, model_name = args.model_name.split(":", 1)
        else:
            model_name = args.model_name
        
        # Create a new ModelArguments with the correct model_name, preserving other fields
        new_args = ModelArguments(
            model_name=model_name,
            per_instance_cost_limit=args.per_instance_cost_limit,
            total_cost_limit=args.total_cost_limit,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            replay_path=args.replay_path,
            host_url=args.host_url,
            per_instance_step_limit=args.per_instance_step_limit,
        )
        super().__init__(new_args, commands)
        self.vllm_model = model_name
        self.host_url = getattr(args, "host_url", "http://localhost:8000")
        if not self.host_url.startswith("http"):
            self.host_url = f"http://{self.host_url}"
        self.api_url = f"{self.host_url}/v1/chat/completions"

    def history_to_messages(self, history: list[dict[str, str]], is_demonstration: bool = False) -> list[dict[str, str]]:
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return [{"role": entry["role"], "content": entry["content"]} for entry in history]
        return [{"role": entry["role"], "content": entry["content"]} for entry in history]

    def query(self, history: list[dict[str, str]]) -> str:
        payload = {
            "model": self.vllm_model,
            "messages": self.history_to_messages(history),
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=3600)
            response.raise_for_status()
            data = response.json()
            # vLLM returns choices[0].message.content
            result = data["choices"][0]["message"]["content"]
            # Use token usage if available
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            self.update_stats(input_tokens, output_tokens)
            return clean_result(result)
        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            raise


def anthropic_history_to_messages(
    model: AnthropicModel | BedrockModel,
    history: list[dict[str, str]],
    is_demonstration: bool = False,
) -> str | list[dict[str, str]]:
    """
    Create `prompt` by filtering out all keys except for role/content per `history` turn
    Reference: https://docs.anthropic.com/claude/reference/complete_post
    """
    # Preserve behavior for older models
    if model.api_model in ["claude-instant", "claude-2.0"] or (
        isinstance(model, BedrockModel) and model.api_model in ["anthropic.claude-instant-v1", "anthropic.claude-v2"]
    ):
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
        # Map history to Claude format
        prompt = "\n\n"
        for entry in history:
            if entry["role"] in {"user", "system"}:
                prompt += f'{HUMAN_PROMPT} {entry["content"]}\n\n'
            elif entry["role"] == "assistant":
                prompt += f'{AI_PROMPT} {entry["content"]}\n\n'
        prompt += AI_PROMPT
        return prompt

    # Remove system messages if it is a demonstration
    if is_demonstration:
        history = [entry for entry in history if entry["role"] != "system"]
        return "\n".join([entry["content"] for entry in history])

    # Return history components with just role, content fields (no system message)
    messages = [
        {k: v for k, v in entry.items() if k in ["role", "content"]} for entry in history if entry["role"] != "system"
    ]
    compiled_messages = []  # Combine messages from the same role
    last_role = None
    for message in reversed(messages):
        if last_role == message["role"]:
            compiled_messages[-1]["content"] = message["content"] + "\n" + compiled_messages[-1]["content"]
        else:
            compiled_messages.append(message)
        last_role = message["role"]
    compiled_messages = list(reversed(compiled_messages))
    # Replace any empty content values with a "(No output)"
    for message in compiled_messages:
        if message["content"].strip() == "":
            message["content"] = "(No output)"
    return compiled_messages


def extract_anthropic_response_parts(response: Any) -> tuple[str, str]:
    """Separate Anthropic thinking/reasoning blocks from executable text."""
    text_parts: list[str] = []
    thought_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        block_type = str(getattr(block, "type", "") or "").lower()
        if block_type in {"thinking", "reasoning"}:
            value = getattr(block, "thinking", None) or getattr(block, "reasoning_content", None)
            if value:
                thought_parts.append(str(value))
        elif block_type == "text":
            value = getattr(block, "text", None)
            if value:
                text_parts.append(str(value))
        else:
            value = getattr(block, "text", None)
            if value:
                text_parts.append(str(value))
    response_text = chr(10).join(text_parts).split("(Open file:")[0].strip()
    thought = chr(10).join(thought_parts).strip()
    return response_text, thought

def anthropic_query(model: AnthropicModel | BedrockModel, history: list[dict[str, str]]) -> str:
    """
    Query the Anthropic API with the given `history` and return the response.
    """
    # Preserve behavior for older models
    if model.api_model in ["claude-instant", "claude-2.0", "claude-2.1"] or (
        isinstance(model, BedrockModel) and model.model_provider == "anthropic" and model.api_model in ["anthropic.claude-instant-v1", "anthropic.claude-v2"]
    ):
        # Perform Anthropic API call
        prompt = anthropic_history_to_messages(model, history)
        if isinstance(model, BedrockModel):
            # Use a dummy Anthropic client since count_tokens
            # is not available in AnthropicBedrock
            # https://github.com/anthropics/anthropic-sdk-python/issues/353
            input_tokens = Anthropic().count_tokens(prompt)
        else:
            input_tokens = model.api.count_tokens(prompt)
        completion = model.api.completions.create(
            model=model.api_model,
            prompt=prompt,
            max_tokens_to_sample=model.model_metadata["max_context"] - input_tokens
            if isinstance(model, Anthropic)
            else model.model_metadata["max_tokens_to_sample"],
            temperature=model.args.temperature,
            top_p=model.args.top_p,
            top_k=model.args.top_k,
        )
        # Calculate + update costs, return response
        response = completion.completion
        if isinstance(model, BedrockModel):
            output_tokens = Anthropic().count_tokens(response)
        else:
            output_tokens = model.api.count_tokens(response)
        model.update_stats(input_tokens, output_tokens)
        return response

    # Get system message(s)
    system_message = "\n".join([entry["content"] for entry in history if entry["role"] == "system"])
    messages = anthropic_history_to_messages(model, history)

    # Perform Anthropic API call
    response = model.api.messages.create(
        messages=messages,
        max_tokens=model.model_metadata["max_tokens"],
        model=model.api_model,
        temperature=model.args.temperature,
        top_p=model.args.top_p,
        system=system_message,
    )
    # Calculate + update costs, return response
    model.update_stats(response.usage.input_tokens, response.usage.output_tokens)
    response_text, reasoning = extract_anthropic_response_parts(response)
    model.last_thought = reasoning or extract_thought(response_text)
    if not response_text:
        raise EmptyModelResponseError("Anthropic API returned no usable text content")
    return response_text


def get_model(args: ModelArguments, commands: list[Command] | None = None):
    """
    Returns correct model object given arguments and commands
    """
    if commands is None:
        commands = []
    
    # Load configurations to check shortcuts
    configs = load_model_configs()
    
    # Special models first
    if args.model_name == "instant_empty_submit":
        return InstantEmptySubmitTestModel(args, commands)
    if args.model_name == "human":
        return HumanModel(args, commands)
    if args.model_name == "human_thought":
        return HumanThoughtModel(args, commands)
    if args.model_name == "replay":
        return ReplayModel(args, commands)
    
    # Check model prefixes
    if re.fullmatch(r"glm52_(?:[1-9]|10)", args.model_name):
        return MessagesAPIModel(args, commands)
    if (args.model_name.startswith("gpt") or 
        args.model_name.startswith("ft:gpt") or 
        args.model_name.startswith("azure:gpt") or 
        args.model_name.startswith("o1") or
        args.model_name.startswith("deepseek-r") or
        args.model_name in configs.get('openai_shortcuts', {}) or
        args.model_name in configs.get('openai_models', {})):
        return OpenAIModel(args, commands)
    elif args.model_name.startswith("claude") or args.model_name in configs.get('anthropic_shortcuts', {}):
        return AnthropicModel(args, commands)
    elif args.model_name.startswith("bedrock"):
        return BedrockModel(args, commands)
    elif args.model_name.startswith("ollama"):
        return OllamaModel(args, commands)
    elif args.model_name.startswith("deepseek") and not args.model_name.startswith("deepseek-r"):
        return DeepSeekModel(args, commands)
    elif (args.model_name.startswith("groq") or 
          args.model_name in configs.get('groq_shortcuts', {}) or
          args.model_name in configs.get('groq_models', {})):
        return GroqModel(args, commands)
    elif args.model_name in configs.get('together_shortcuts', {}) or args.model_name in configs.get('together_models', {}):
        return TogetherModel(args, commands)
    elif args.model_name.startswith("vllm:"):
        return VLLMModel(args, commands)
    else:
        # Try to determine model type from configurations
        if args.model_name in configs.get('openai_models', {}):
            return OpenAIModel(args, commands)
        elif args.model_name in configs.get('anthropic_models', {}):
            return AnthropicModel(args, commands)
        elif args.model_name in configs.get('groq_models', {}):
            return GroqModel(args, commands)
        elif args.model_name in configs.get('together_models', {}):
            return TogetherModel(args, commands)
        elif args.model_name in configs.get('deepseek_models', {}):
            return DeepSeekModel(args, commands)
        elif args.model_name in configs.get('special_models', {}):
            # Default to OpenAI-compatible for unknown special models
            return OpenAIModel(args, commands)
        else:
            # Default to OpenAI model for unknown models (with warning in BaseModel)
            return OpenAIModel(args, commands)
