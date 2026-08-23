# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


#
"""
LLM client for interacting with various language models.
"""

from typing import List, Dict, Any, Optional
import json
import time

import litellm
import requests
from litellm import completion

from .config import Config
from .validation import ResponseValidator
from .models import ConversationTurn

# Suppress debug info from litellm
litellm.suppress_debug_info = True


class LLMClient:
    """Client for interacting with language models through litellm."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.validator = ResponseValidator(config)
    
    def call_model_response(
        self,
        messages: List[Dict[str, Any]],
        role: str,
        model_id: str = "deepseek-v3-0324",
        max_retries: int = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call a model and preserve text, reasoning, and structured tool calls."""
        max_retries = max_retries or self.config.MAX_RETRIES
        model_full_id = self.config.models.get_model_id(model_id)
        effective_temperature = temperature if temperature is not None else self.config.temperature
        effective_top_p = top_p if top_p is not None else self.config.top_p

        for attempt in range(max_retries):
            try:
                if self.config.messages_api_base_url:
                    response = self._call_messages_api(
                        messages=messages,
                        model=model_full_id,
                        temperature=effective_temperature,
                        top_p=effective_top_p,
                    )
                else:
                    raw = completion(
                        model=model_full_id,
                        messages=messages,
                        temperature=effective_temperature,
                        top_p=effective_top_p,
                    )
                    response = self._normalize_model_message(
                        raw.get('choices', [{}])[0].get('message', {})
                    )

                if not response.get('content') and not response.get('tool_calls'):
                    raise Exception("No response from model")

                content = response.get('content', '')
                if content and not self._validate_model_response(content, role):
                    raise Exception("Response validation failed")
                return response
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if "long" in str(e) or attempt >= max_retries - 1:
                    return None
                time.sleep(min(2 ** attempt, 10))
        return None

    def call_model(
        self,
        messages: List[Dict[str, Any]],
        role: str,
        model_id: str = "deepseek-v3-0324",
        max_retries: int = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Optional[str]:
        """Call a model and return only text for legacy callers."""
        response = self.call_model_response(
            messages=messages,
            role=role,
            model_id=model_id,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
        )
        return response.get('content', '') if response else None

    def _normalize_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Normalize provider-specific tool calls to name/arguments dictionaries."""
        normalized = []
        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue
            function = call.get('function') if isinstance(call.get('function'), dict) else call
            name = function.get('name') or call.get('name')
            arguments = function.get('arguments', function.get('input', call.get('arguments', {})))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {'command': arguments}
            if not isinstance(arguments, dict):
                arguments = {'value': arguments}
            if name:
                normalized.append({'name': name, 'arguments': arguments})
        return normalized

    def _normalize_model_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenAI-compatible message responses."""
        content = message.get('content', '')
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        tool_calls = self._normalize_tool_calls(message.get('tool_calls'))
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = block.get('type')
                if block_type in ('text', 'output_text'):
                    text_parts.append(block.get('text', ''))
                elif block_type in ('thinking', 'reasoning'):
                    reasoning += block.get('thinking', block.get('text', ''))
                elif block_type in ('tool_use', 'tool_call'):
                    tool_calls.extend(self._normalize_tool_calls([block]))
            content = ''.join(text_parts)
        return {
            'content': content if isinstance(content, str) else ('' if content is None else str(content)),
            'reasoning_content': reasoning,
            'tool_calls': tool_calls,
        }

    def _call_messages_api(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        """Call a Messages-compatible endpoint and normalize its response."""
        system_messages = [item["content"] for item in messages if item.get("role") == "system"]
        body_messages = [
            {"role": item["role"], "content": item["content"]}
            for item in messages
            if item.get("role") != "system"
        ]
        if not body_messages:
            raise ValueError("Messages API requires at least one non-system message")

        payload = {
            "model": model,
            "messages": body_messages,
            "max_tokens": self.config.messages_api_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)

        response = requests.post(
            self.config.messages_api_base_url,
            headers={
                "x-api-key": self.config.messages_api_key,
                "Authorization": f"Bearer {self.config.messages_api_key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        if 'choices' in data:
            return self._normalize_model_message(
                data.get('choices', [{}])[0].get('message', {})
            )
        return self._normalize_model_message(data)

    def _validate_model_response(self, response: str, role: str) -> bool:
        """Validate model response according to framework rules."""
        # Check colon patterns
        if not self.validator.check_colon_patterns(response, role):
            print("Colon pattern validation failed")
            return False
        
        # Check markdown balance
        if not self.validator.check_markdown_balance(response, role):
            print("Markdown balance check failed")
            return False
        
        # Check hint format
        if not self.validator.check_hint_format(response):
            print("Hint format check failed")
            return False
        
        # Validate response content
        if not self.validator.validate_response(response, role):
            print("Response content validation failed")
            return False
        
        return True
    
    def prepare_assistant_messages(
        self,
        conversation: List[ConversationTurn],
        system_prompt: str
    ) -> List[Dict[str, str]]:
        """Prepare messages for assistant model call."""
        messages = [{"role": "system", "content": system_prompt}]
        
        for turn in conversation:
            if turn.role == "system":
                continue
            role = "user" if turn.role == "tool" else turn.role
            messages.append({
                "role": role,
                "content": turn.content
            })
        
        # Add instruction for single code block
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += "\nMAKE SURE YOU ONLY INCLUDE ONE BASH MARKDOWN CODE BLOCK IN YOUR RESPONSE."
        
        return messages
    
    def prepare_user_messages(
        self,
        conversation: List[ConversationTurn],
        system_prompt: str,
        current_command: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Prepare messages for user (terminal) model call."""
        # Build user model context: only code blocks from assistant turns, full user turns
        user_context = []
        
        for turn in conversation:
            if turn.role == "assistant":
                import re
                commands = [
                    call.get("arguments", {}).get("command", "")
                    for call in (turn.tool_calls or [])
                    if call.get("name") == "Bash"
                    and isinstance(call.get("arguments"), dict)
                ]
                if not commands:
                    code_blocks = re.findall(r"```bash\n([\s\S]*?)\n```", turn.content)
                    commands = [code_blocks[0].strip()] if code_blocks else []
                for command in commands:
                    user_context.append({
                        "role": "user",
                        "content": f"```bash\n{command}\n```",
                    })
            elif turn.role in ("user", "tool"):
                user_context.append({"role": "assistant", "content": turn.content})
        
        messages = [{"role": "system", "content": system_prompt}] + user_context
        if current_command:
            messages.append({
                "role": "user",
                "content": f"```bash\n{current_command.strip()}\n```",
            })
        return messages


class QualityEvaluator:
    """Evaluates trajectory quality using LLM."""
    
    QUALITY_PROMPT = """You are an expert and strict CTF security researcher and educator. Your task is to evaluate the quality of a CTF solving trajectory.

You will be given:
1. A CTF task with its description, writeup, and expected solution
2. A conversation trajectory showing how an AI agent solved the task step by step

Please evaluate whether this trajectory demonstrates HIGH QUALITY by checking if it aligns with:
1. **Task Writeup Alignment**: Does the trajectory workflow match the intended approach described in the writeup? If the writeup does not provide useful information, you should judge based on your own knowledge and experience.
2. **Real-world CTF Practice**: Does the solving approach follow realistic CTF methodologies and best practices?

NOT HIGH QUALITY indicators:
- Deviates the method and scenario from writeup solution
- Uses unrealistic or impossible techniques
- Shows inconsistent or illogical progression
- Execution behavior of Linux system has flaws
- Relies on lucky guesses without proper analysis
- Contains errors in technical understanding
- Has unclear or confusing methodology
- Not be able to use as the learning material for the intermediate level CTF players
- The approach cannot generalize to other CTF tasks

Think step by step in plain text and then respond with a Markdown code block (``` and ```) ONLY containing "true" if the trajectory demonstrates VERY HIGH QUALITY, or "false" if it does not meet the very high quality standards."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm_client = LLMClient(config)
    
    def evaluate_trajectory(
        self,
        trajectory_data: Dict[str, Any],
        model_id: str = None,
        num_evaluations: int = None
    ) -> bool:
        """
        Evaluate trajectory quality.
        
        Returns True if high quality, False otherwise.
        """
        model_id = model_id or self.config.QUALITY_EVALUATION_MODEL
        num_evaluations = num_evaluations or self.config.NUM_EVALUATIONS
        
        # Build evaluation prompt
        evaluation_content = self._build_evaluation_content(trajectory_data)
        messages = [
            {"role": "system", "content": self.QUALITY_PROMPT},
            {"role": "user", "content": evaluation_content}
        ]
        
        # Perform multiple evaluations
        for _ in range(num_evaluations):
            response = self.llm_client.call_model(
                messages=messages,
                role="assistant",
                model_id=model_id,
                max_retries=3
            )
            
            if response is None:
                return False
            
            # Extract evaluation result
            import re
            code_block_match = re.search(r'```\s*(true|false)\s*```', response, re.IGNORECASE)
            if code_block_match:
                result = code_block_match.group(1).lower() == 'true'
                if not result:  # If any evaluation is false, consider it low quality
                    return False
            else:
                return False  # Invalid response format
        
        return True
    
    def _build_evaluation_content(self, trajectory_data: Dict[str, Any]) -> str:
        """Build the content for quality evaluation."""
        task_info = f"""**Task Information:**
- Name: {trajectory_data.get('task_name', '')}
- Category: {trajectory_data.get('task_tag', '')}
- Points: {trajectory_data.get('task_points', '')}
- Description: {trajectory_data.get('task_description', '')}
- Expected Solution: {trajectory_data.get('solution', '')}

**Trajectory:**
"""
        
        # Add conversation turns
        trajectory = trajectory_data.get('trajectory', [])
        for i, turn in enumerate(trajectory):
            role = turn.get('role', '')
            content = turn.get('content', '')
            task_info += f"\n**Turn {i+1} ({role.title()}):**\n{content}\n"
        
        return task_info 