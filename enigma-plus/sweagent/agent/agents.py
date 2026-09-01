# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


#
from __future__ import annotations

import copy
import json
import re
import time
import traceback
import threading
import queue
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from tenacity import RetryError

from sweagent.agent.commands import Command, ParseCommand
from sweagent.agent.history_processors import HistoryProcessor
from sweagent.agent.models import (
    APIStats,
    ContextWindowExceededError,
    CostLimitExceededError,
    EmptyModelResponseError,
    ModelArguments,
    _tool_call_to_shell_action,
    get_model,
)
from sweagent.agent.parsing import FormatError, ParseFunction
from sweagent.agent.summarizer import SummarizerConfig
from sweagent.agent.context_compressor import ContextCompressionManager
from sweagent.agent.trajectory_recorder import TrajectoryRecorder
from sweagent.environment.swe_env import CANONICAL_STATE_DEFAULTS, SWEEnv, normalize_state
from sweagent.types import AgentInfo, History, HistoryItem, Trajectory, TrajectoryStep
from sweagent.utils.config import convert_paths_to_abspath, keys_config
from sweagent.utils.log import get_logger

# Import the task timeout constant
from sweagent.environment.swe_env import TASK_EXECUTION_TIMEOUT, MODEL_GENERATION_TIMEOUT

# A malformed response should not trigger an unbounded series of expensive
# gateway calls.  Two corrective requests are enough to recover ordinary
# formatting mistakes; repeated identical output is stopped immediately.
FORMAT_RETRY_LIMIT = max(0, int(keys_config.get("SWE_AGENT_FORMAT_RETRY_LIMIT", 2)))


def native_tool_result_blocks(tool_use_id: str, observation: str, status_text: str) -> list[dict[str, Any]]:
    """Build the user content after a native tool call without duplicating output.

    The complete command output belongs in ``tool_result``.  The companion
    text block is reserved for prompt state (working directory, open file,
    interactive session, and the shell marker); callers must pass a
    status-only rendering there.
    """
    return [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": observation,
        },
        {"type": "text", "text": status_text},
    ]


@dataclass(frozen=True)
class Subroutine(FrozenSerializable):
    name: str
    agent_file: str
    # one of "action", "observation", "response", "state", "thought"
    return_type: str = None  # type: ignore
    init_observation: str | None = None
    end_name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    model: ModelArguments | None = None
    agent_args: Any | None = None


@dataclass(frozen=True)
class AgentConfig(FrozenSerializable):
    system_template: str
    instance_template: str
    next_step_template: str | None = None  # defaults to instance_template
    next_step_no_output_template: str | None = None  # defaults to next_step_template
    strategy_template: str | None = None
    demonstration_template: str | None = None
    # Paths to demonstrations. If path is not absolute, it is assumed to be
    # relative to the SWE_AGENT_CONFIG_ROOT (if set) or the SWE-agent repository root
    demonstrations: list[str | Path] = field(default_factory=list)
    put_demos_in_history: bool = False  # if True, add demonstration to history instead of as a single message
    # defaults to format_error_template in ParseFunction
    format_error_template: str = None  # type: ignore
    # Paths to command files. If path is not absolute, it is assumed to be
    # relative to the SWE_AGENT_CONFIG_ROOT (if set) or the SWE-agent repository root
    command_files: list[str | Path] = field(default_factory=list)
    env_variables: dict[str, str] = field(default_factory=dict)
    util_functions: list[str] = field(default_factory=list)
    submit_command: str = "submit"
    parse_function: str = "ThoughtActionParser"
    parse_command: str = "ParseCommandBash"
    history_processor: str = "DefaultHistoryProcessor"
    history_processor_args: dict[str, Any] = field(default_factory=dict)
    command_docs: str = None  # type: ignore
    summarizer_config: SummarizerConfig = field(default_factory=SummarizerConfig)
    enable_context_compression: bool = False
    enable_thought_recording: bool = True
    context_compression: dict[str, Any] = field(default_factory=dict)
    blocklist_error_template: str = "Interactive operation '{name}' is not supported by this environment"
    blocklist: tuple[str, ...] = (
        "vim",
        "vi",
        "emacs",
        "nano",
        "nohup",
        "git",
        "gdb",
    )
    blocklist_standalone: tuple[str, ...] = (
        "python",
        "python3",
        "ipython",
        "bash",
        "sh",
        "exit",
        "/bin/bash",
        "/bin/sh",
        "nohup",
        "vi",
        "vim",
        "emacs",
        "nano",
        "su",
    )
    block_unless_regex: dict[str, str] = field(default_factory=dict)
    # Should extract environment state in a json readable form
    state_command: Command = Command(
        name="state",
        code="""state() {
            # More robust state command that doesn't depend on $ROOT
            local current_dir=$(pwd)
            local relative_dir="."
            
            # Try to get relative path from repository root if possible
            if [[ -n "$ROOT" ]] && [[ -d "$ROOT" ]]; then
                # If ROOT is set and exists, use it
                relative_dir=$(realpath --relative-to="$ROOT/.." "$current_dir" 2>/dev/null || echo ".")
            elif [[ -d ".git" ]]; then
                # If we're in a git repository, use git root
                local git_root=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
                if [[ -n "$git_root" ]]; then
                    relative_dir=$(realpath --relative-to="$git_root/.." "$current_dir" 2>/dev/null || echo ".")
                fi
            fi
            
            echo '{"working_dir": "'$relative_dir'"}';
        };""",
    )
    _commands: list[Command] = field(default_factory=list)
    _subroutines: dict[str, Subroutine] = field(default_factory=dict)
    subroutine_types: list[Subroutine] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "command_files", convert_paths_to_abspath(self.command_files))
        object.__setattr__(self, "demonstrations", convert_paths_to_abspath(self.demonstrations))

        if self.next_step_template is None:
            object.__setattr__(self, "next_step_template", self.instance_template)
        if self.next_step_no_output_template is None:
            object.__setattr__(self, "next_step_no_output_template", self.next_step_template)

        object.__setattr__(self, "parse_command", ParseCommand.get(self.parse_command))
        for file in self.command_files:
            commands = self.parse_command.parse_command_file(file)

            util_functions = [command for command in commands if command.name.startswith("_")]
            commands = [command for command in commands if not command.name.startswith("_")]

            object.__setattr__(self, "util_functions", self.util_functions + util_functions)
            object.__setattr__(self, "_commands", self._commands + commands)

        for subroutine in self.subroutine_types:
            if subroutine.name == "submit":
                msg = "Cannot use 'submit' as a subroutine name"
                raise ValueError(msg)
            agent_args = AgentArguments(
                model=subroutine.model,
                config_file=subroutine.agent_file,
            )
            object.__setattr__(subroutine, "agent_args", agent_args)
            object.__setattr__(self, "_subroutines", {**self._subroutines, subroutine.name: subroutine})

        multi_line_command_endings = {
            command.name: command.end_name
            for command in [*self._commands, *self._subroutines.values()]
            if command.end_name is not None
        }
        object.__setattr__(self, "multi_line_command_endings", multi_line_command_endings)
        object.__setattr__(
            self,
            "command_docs",
            self.parse_command.generate_command_docs(
                self._commands,
                self.subroutine_types,
                **self.env_variables,
            ),
        )
        object.__setattr__(self, "parse_function", ParseFunction.get(self.parse_function))
        if self.format_error_template is None:
            object.__setattr__(
                self,
                "format_error_template",
                self.parse_function.format_error_template,
            )
        object.__setattr__(
            self,
            "format_error_template",
            self.format_error_template.format(**self.__dict__),
        )
        for command in self._commands:
            if command.name == self.submit_command:
                object.__setattr__(self, "submit_command_end_name", command.end_name)
                break
        object.__setattr__(
            self,
            "history_processor",
            HistoryProcessor.get(self.history_processor, **self.history_processor_args),
        )
        if "WINDOW" in self.env_variables:
            window_size = self.env_variables["WINDOW"]
            if self.summarizer_config.window_length < int(window_size):
                msg = f"Summarizer window length is set to {self.summarizer_config.window_length} which is less than the window length {window_size}"
                raise ValueError(msg)
        object.__setattr__(
            self,
            "block_unless_regex",
            {"radare2": r"\b(?:radare2)\b.*\s+-c\s+.*", "r2": r"\b(?:radare2)\b.*\s+-c\s+.*"},
        )


@dataclass(frozen=True)
class AgentArguments(FlattenedAccess, FrozenSerializable):
    """Configure the agent's behaviour (templates, parse functions, blocklists, ...)."""

    model: ModelArguments = None

    # Policy can only be set via config yaml file from command line
    config_file: Path | None = None
    config: AgentConfig | None = field(default=None, cmd=False)

    def __post_init__(self):
        if self.config is None and self.config_file is not None:
            # If unassigned, we load the config from the file to store its contents with the overall arguments
            config = AgentConfig.load_yaml(self.config_file)
            object.__setattr__(self, "config", config)
        assert self.config is not None  # mypy
        for subroutine in getattr(self.config, "subroutines", {}).values():
            model_args = subroutine.model
            object.__setattr__(
                model_args,
                "per_instance_cost_limit",
                self.model.per_instance_cost_limit,
            )
            object.__setattr__(model_args, "total_cost_limit", self.model.total_cost_limit)


class AgentHook:
    def on_init(self, *, agent: Agent):
        """Note: Depending on the internals of `Agent` should be done with care,
        it's best to use this as little as possible.
        """

    def on_run_start(
        self,
    ): ...

    def on_step_start(self): ...

    def on_actions_generated(self, *, thought: str, action: str, output: str): ...

    def on_sub_action_started(self, *, sub_action: str): ...

    def on_sub_action_executed(self, *, obs: str, done: bool): ...

    def on_step_done(self, *, trajectory_step: TrajectoryStep, model_stats: APIStats): ...

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo): ...

    def on_model_query(self, *, query: str, agent: str):
        """Actually query the model with the complete history."""

    def on_query_message_added(
        self,
        *,
        role: str,
        content: str,
        agent: str,
        is_demo: bool = False,
        thought: str = "",
        action: str = "",
    ): ...

    def on_setup_done(self): ...


class SubAction(TypedDict):
    agent: str
    action: str
    cmd_name: str | None
    args: str


class Agent:
    """Agent handles the behaviour of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: AgentArguments):
        self.name = name
        # todo: currently only used to get the model name, so might remove this later
        self._args = args
        self.model = get_model(args.model, args.config._commands + args.config.subroutine_types)
        self.summarizer_model = get_model(
            args.config.summarizer_config.model if args.config.summarizer_config.model is not None else args.model
        )
        self.config = args.config
        assert self.config is not None  # mypy
        self.system_args = {
            "command_docs": self.config.command_docs,
            **self.config.env_variables,
        }
        self.instance_args = None
        self._parse_command_patterns()
        self.last_container_id = None
        self.hooks = []
        self.logger = get_logger("agent")
        # Requires instance, so is set in `setup` methods
        self._rloop = None
        compression = dict(getattr(self.config, "context_compression", {}) or {})
        self.context_compressor = ContextCompressionManager(
            enabled=bool(getattr(self.config, "enable_context_compression", False) or compression.get("enabled", False)),
            max_context_tokens=int(compression.get("max_context_tokens", 128000)),
            trigger_ratio=float(compression.get("trigger_ratio", 0.95)),
            summary_model=self._summarize_context,
            max_summary_input_chars=int(compression.get("max_summary_input_chars", 120000)),
            max_summary_output_chars=int(compression.get("max_summary_output_chars", 24000)),
        )
        self.trajectory_recorder = TrajectoryRecorder(enable_thought_recording=bool(getattr(self.config, "enable_thought_recording", True)))
        self._last_context_compressed = False

        # Set in run method
        self._env: SWEEnv | None = None
        self.traj_dir: None | Path = None

        #: Number of attempts to solve the issue when using a review loop
        self._i_attempt: int = 0

        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self._history_by_attempt: dict[int, list] = defaultdict(list)
        self._trajectory_by_attempt: dict[int, Trajectory] = defaultdict(list)
        self._info_by_attempt: dict[int, AgentInfo] = defaultdict(dict)

        #: Variables to be referenced in the templates that are forwarded from one
        #: solution attempt to the next
        self._forwarded_vars: dict[str, Any] = {}

    @property
    def history(self) -> History:
        """History that is passed on to the model.
        Use `_append_history` to modify.
        """
        return self._history_by_attempt[self._i_attempt]

    @history.setter
    def history(self, value: History):
        self._history_by_attempt[self._i_attempt] = value

    @property
    def trajectory(self) -> Trajectory:
        """Trajectory of the agent for the current instance. In contrast to `history`,
        this is mostly for the informational value of how the agent interacted with
        the environment and is also what is being used when replaying the trajectory
        """
        return self._trajectory_by_attempt[self._i_attempt]

    @trajectory.setter
    def trajectory(self, value: Trajectory):
        self._trajectory_by_attempt[self._i_attempt] = value

    @property
    def info(self) -> AgentInfo:
        """Information about the agent's run"""
        return self._info_by_attempt[self._i_attempt]

    @info.setter
    def info(self, value: AgentInfo):
        self._info_by_attempt[self._i_attempt] = value

    @property
    def traj_path(self) -> Path | None:
        """Returns path to the trajectory.
        The path is reset for every new instance.
        """
        if self.traj_dir and self._env is not None:
            assert self._env.record
            return self.traj_dir / (self._env.record["instance_id"] + ".traj")
        return None

    def add_hook(self, hook: AgentHook) -> None:
        """Add hook to agent"""
        hook.on_init(agent=self)
        self.hooks.append(hook)

    def _append_history(self, item: HistoryItem) -> None:
        """Adds an item to history while keeping hook calls backward compatible."""
        for hook in self.hooks:
            hook_content = item.get("content", "") or ""
            if not isinstance(hook_content, str):
                hook_content = json.dumps(hook_content, ensure_ascii=False)
            hook.on_query_message_added(
                role=item.get("role", ""),
                content=hook_content,
                agent=item.get("agent", self.name),
                is_demo=item.get("is_demo", False),
                thought=item.get("thought", ""),
                action=item.get("action", "") or "",
            )
        self.history.append(item)

    def _model_response_metadata(self) -> dict[str, Any]:
        """Return JSON-safe structured metadata from the latest model query."""
        return {
            "content_blocks": copy.deepcopy(getattr(self.model, "last_content_blocks", [])),
            "content_text": getattr(self.model, "last_content_text", "") or "",
            "text_content": getattr(self.model, "last_text_content", "") or "",
            "raw_response": copy.deepcopy(getattr(self.model, "last_raw_response", {})),
            "stop_reason": getattr(self.model, "last_stop_reason", None),
            "usage": copy.deepcopy(getattr(self.model, "last_usage", {})),
        }

    def _query_model_with_timeout(self, history: list[dict[str, str]], timeout: float = None) -> str:
        """
        Query the model with a timeout to prevent hanging model generation from blocking task timeout.
        
        Args:
            history: Chat history to send to model
            timeout: Timeout in seconds (default: MODEL_GENERATION_TIMEOUT)
            
        Returns:
            str: Model response
            
        Raises:
            TimeoutError: If model generation exceeds timeout
            RuntimeError: If model query fails
        """
        if timeout is None:
            timeout = MODEL_GENERATION_TIMEOUT
            
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def model_query_worker():
            """Worker function to run model query in separate thread"""
            try:
                response = self.model.query(history)
                result_queue.put(response)
            except Exception as e:
                exception_queue.put(e)
        
        # Start model query in separate thread
        query_thread = threading.Thread(target=model_query_worker, daemon=True)
        query_thread.start()
        
        # Wait for result with timeout
        query_thread.join(timeout)
        
        if query_thread.is_alive():
            # Thread is still running - model query timed out
            transport_timeout = getattr(self.model, "request_timeout", None)
            self.logger.error(
                "Model generation timed out: limit=%.1fs history_entries=%d transport_timeout=%s model=%s; provider request is still running",
                timeout, len(history), transport_timeout, type(self.model).__name__,
            )
            # Note: We can't actually kill the thread, but we can stop waiting for it
            raise TimeoutError(
                f"Model generation exceeded {timeout} second timeout "
                f"(history_entries={len(history)}, transport_timeout={transport_timeout})"
            )
        
        # Check if an exception occurred
        if not exception_queue.empty():
            exception = exception_queue.get()
            self.logger.warning(f"Model query failed: {exception}")
            raise exception
            
        # Check if we have a result
        if not result_queue.empty():
            return result_queue.get()
        else:
            # Thread finished but no result - this shouldn't happen
            raise RuntimeError("Model query thread finished without result or exception")

    # todo: klieret: Long term: Might make more sense to reinitialize the agent class for every instance instead of this
    def setup(self, instance_args: dict[str, Any], init_model_stats: APIStats | None = None) -> None:
        """Setup the agent for a new instance. This includes
        formatting the system message and adding demonstrations to the history.

        Args:
            instance_args: Arguments for the instance
        """
        assert self.config is not None  # mypy
        self.instance_args = instance_args

        self._i_attempt = 0
        self._history_by_attempt = defaultdict(list)
        self._trajectory_by_attempt = defaultdict(list)
        self._info_by_attempt = defaultdict(dict)  # type: ignore
        self._forwarded_vars = {}
        if self._rloop is not None:
            self._forwarded_vars = self._rloop.get_forwarded_vars()

        self.setup_attempt(init_model_stats=init_model_stats)

        for hook in self.hooks:
            hook.on_setup_done()

    def setup_attempt(self, *, init_model_stats: APIStats | None = None) -> None:
        """Setup the agent for a new attempt. This includes resetting the model stats."""
        assert self.config is not None  # mypy
        if self._i_attempt > 0 and init_model_stats is not None:
            msg = (
                "We might be dealing with nested retries, where subroutines are mixed with retries. "
                "Currently, this messes up accounting with init_model_stats."
            )
            raise ValueError(msg)
        if self._i_attempt > 0:
            assert self._env is not None  # mypy
            self._env.reset_for_new_attempt()
        self.model.reset_stats(init_model_stats)
        # self.model = get_model(self._args.model, self.config._commands + self.config.subroutine_types)
        # fixme: This doesn't reset total cost
        system_msg = self.config.system_template.format(**self.system_args, **self.instance_args)
        self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(HistoryItem({"role": "system", "content": system_msg, "agent": self.name}))
        if "history_to_messages" in dir(self.model):
            for demonstration_path in self.config.demonstrations:
                if self.config.demonstration_template is None and not self.config.put_demos_in_history:
                    msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
                    raise ValueError(msg)

                # Load history
                self.logger.info(f"DEMONSTRATION: {demonstration_path}")
                demo_history = json.loads(Path(demonstration_path).read_text())["history"]
                demo_history = [
                    entry
                    for entry in demo_history
                    if ("agent" not in entry) or ("agent" in entry and entry["agent"] == self.name)
                ]

                if self.config.put_demos_in_history:
                    if self.config.demonstration_template is not None:
                        self.logger.warning("Demonstration template is ignored for put_demos_in_history=True")
                    # Add demonstration to history directly as separate messages
                    for entry in demo_history:
                        if entry["role"] != "system":
                            entry["is_demo"] = True
                            self._append_history(entry)
                else:
                    # Add demonstration as single message to history
                    demo_message = self.model.history_to_messages(
                        demo_history,
                        is_demonstration=True,
                    )
                    demonstration = self.config.demonstration_template.format(demonstration=demo_message)
                    self._append_history(
                        {
                            "agent": self.name,
                            "content": demonstration,
                            "is_demo": True,
                            "role": "user",
                        },
                    )

    @property
    def state_command(self) -> str:
        """Return the bash command that will be used to extract the environment state."""
        assert self.config is not None
        return self.config.state_command.name

    @property
    def local_history(self) -> list[dict[str, str]]:
        """Return the history of the agent since the last reset."""
        history = self.config.history_processor([entry for entry in self.history if entry["agent"] == self.name])
        result = self.context_compressor.maybe_compress(history)
        self._last_context_compressed = result.compressed
        if result.compressed:
            for msg in result.messages:
                if msg.get("context_summary") is not None:
                    msg["agent"] = self.name
            self.history = [entry for entry in self.history if entry.get("role") == "system"]
            self.history.extend(result.messages[1:])
            self.info.setdefault("context_events", []).append(self.context_compressor.events[-1])
        return result.messages

    def _summarize_context(self, messages: list[dict[str, Any]]) -> Any:
        return self._query_model_with_timeout(messages)

    def _get_total_stats(self) -> APIStats:
        """Combine model stats of different attempts"""
        total_stats = APIStats()
        for stats in self._info_by_attempt.values():
            assert "model_stats" in stats  # mypy
            attempt_stats = APIStats(**stats["model_stats"])  # type: ignore
            total_stats += attempt_stats
        if self._rloop is not None:
            total_stats += self._rloop.model_stats
        return total_stats

    def save_trajectory(
        self,
    ) -> None:
        """Save the trajectory to disk.
        This includes the history, the environment state, and the model stats.
        """

        def get_attempt_data(attempt_idx: int) -> dict[str, Any]:
            """Get data saved for every attempt"""
            assert self._env is not None
            # The deepcopy here is important because else the
            # data["info"]["model_stats"] update will create havoc!
            return copy.deepcopy(
                {
                    "environment": self._env.name,
                    "trajectory": self._trajectory_by_attempt[attempt_idx],
                    "history": self._history_by_attempt[attempt_idx],
                    "info": self._info_by_attempt[attempt_idx],
                }
            )

        self.info.setdefault("trajectory_schema_version", 2)
        self.info.setdefault("artifact_type", "agent_trace")
        data = {"trajectory_schema_version": 2, **get_attempt_data(0), "context_events": self.info.get("context_events", [])}
        assert self.traj_path is not None
        self.trajectory_recorder.save(self.traj_path, data)

    def finalize_episode(self, exit_status: str, terminal_reason: str, error_type: str | None = None) -> None:
        """Record one canonical terminal state and durably save the partial trajectory."""
        if getattr(self, "_episode_finalized", False):
            return
        self._episode_finalized = True
        self.info["exit_status"] = exit_status
        self.info["terminal_reason"] = terminal_reason
        self.info["error_type"] = error_type or ""
        self.info.setdefault("flag_submitted", False)
        self.info.setdefault("flag_verified", None)
        self.info["episode_end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.trajectory.append(TrajectoryStep({
            "step_id": len(self.trajectory) + 1, "thought": "", "action": exit_status,
            "tool_name": "unknown", "tool_args": "", "observation": terminal_reason,
            "return_code": None, "execution_time": 0.0, "error": error_type or "",
            "terminal": True, "response": "", "state": "",
        }))
        self.info["total_steps"] = len(self.trajectory)
        try:
            self.save_trajectory()
        except Exception:
            self.logger.exception("Failed to save terminal trajectory")

    @staticmethod
    def _structured_action(action: str) -> tuple[str, str]:
        """Extract command metadata without discarding the raw action."""
        parts = action.strip().split(maxsplit=1)
        return (parts[0], parts[1] if len(parts) > 1 else "") if parts else ("unknown", "")

    @staticmethod
    def _strict_action_is_valid(action: str) -> bool:
        """Reject prose/Markdown while allowing one parsed multiline command.

        ``ThoughtActionParser`` removes the outer fenced block before this
        check.  A heredoc or a multiline ``python3 -c`` is therefore a valid
        single action and must not be rejected merely because it contains
        newlines.
        """
        stripped = action.strip()
        if not stripped or "```" in stripped or re.search(r"(?im)^\s*(discussion|analysis|explanation)\s*:", stripped):
            return False
        return True
    def _get_first_match(self, action: str, pattern_type: str) -> re.Match | None:
        """Return the first match of a command pattern in the action string."""
        assert self.config is not None  # mypy
        if pattern_type == "subroutine":
            patterns = {k: v for k, v in self.subroutine_patterns.items()}
        elif pattern_type == "multi_line":
            patterns = {
                k: v
                for k, v in self.command_patterns.items()
                if k in self.config.multi_line_command_endings or k == self.config.submit_command
            }
            patterns += {
                k: v for k, v in self.subroutine_patterns.items() if k in self.config.multi_line_command_endings
            }
        elif pattern_type == "multi_line_no_subroutines":
            patterns = {k: v for k, v in self.command_patterns.items() if k in self.config.multi_line_command_endings}
        else:
            msg = f"Unknown pattern type: {pattern_type}"
            raise ValueError(msg)
        matches = list()
        for _, pat in patterns.items():
            match = pat.search(action)
            if match:
                matches.append(match)
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda x: x.start())
        return matches[0]

    def _guard_multiline_input(self, action: str) -> str:
        """Split action by multiline commands, then append the first line in each multiline command with "<< '{end_name}'".
        Multiline commands (which are specified by an end_name) are commands that span multiple lines and are terminated by a specific end_name.

        Their multi-line argument is sent using a heredoc, which is a way to send a multi-line string to a command in bash.
        """
        parsed_action = list()
        rem_action = action
        while rem_action.strip():
            first_match = self._get_first_match(rem_action, "multi_line_no_subroutines")
            if first_match:
                pre_action = rem_action[: first_match.start()]
                match_action = rem_action[first_match.start() : first_match.end()]
                rem_action = rem_action[first_match.end() :]
                if pre_action.strip():
                    parsed_action.append(pre_action)
                if match_action.strip():
                    eof = first_match.group(3).strip()
                    if not match_action.split("\n")[0].strip().endswith(f"<< '{eof}'"):
                        guarded_command = match_action[first_match.start() :]
                        first_line = guarded_command.split("\n")[0]
                        guarded_command = guarded_command.replace(first_line, first_line + f" << '{eof}'", 1)
                        parsed_action.append(guarded_command)
                    else:
                        parsed_action.append(match_action)
            else:
                parsed_action.append(rem_action)
                rem_action = ""
        return "\n".join(parsed_action)

    def split_actions(self, action: str, pattern_type="subroutine") -> list[SubAction]:
        """Split an action into a list of actions in a greedy manner, each of which is a subroutine call or a single command."""
        parsed_action: list[SubAction] = list()
        rem_action = action
        while rem_action.strip():
            first_match = self._get_first_match(rem_action, pattern_type)
            if first_match:
                pre_action = rem_action[: first_match.start()]
                match_action = rem_action[first_match.start() : first_match.end()]
                rem_action = rem_action[first_match.end() :]
                if pre_action.strip():
                    parsed_action.append({"agent": self.name, "action": pre_action, "cmd_name": None, "args": ""})
                if match_action.strip():
                    if match_action.split()[0] == self.config.submit_command:
                        parsed_action.append(
                            SubAction(
                                {
                                    "agent": self.name,
                                    "action": match_action,
                                    "cmd_name": first_match.group(1),
                                    "args": "",
                                },
                            )
                        )  # submit command is not a subroutine
                    else:
                        parsed_action.append(
                            SubAction(
                                {
                                    "agent": first_match.group(1),
                                    "args": first_match.group(2),
                                    "action": match_action,
                                    "cmd_name": first_match.group(1),
                                },
                            )
                        )
            else:
                parsed_action.append(
                    SubAction({"agent": self.name, "action": rem_action, "cmd_name": None, "args": ""})
                )
                rem_action = ""
        return parsed_action

    def _parse_command_patterns(self) -> None:
        assert self.config is not None  # mypy
        self.command_patterns = dict()
        for command in self.config._commands:
            if command.end_name is not None:
                pat = re.compile(
                    rf"^\s*({command.name})\s*(.*?)^({command.end_name})\s*$",
                    re.DOTALL | re.MULTILINE,
                )
                self.command_patterns[command.name] = pat
            else:
                pat = re.compile(rf"^\s*({command.name})\s*(.*?)$", re.MULTILINE)
                self.command_patterns[command.name] = pat
        self.subroutine_patterns = dict()
        for _, subroutine in self.config._subroutines.items():
            if subroutine.end_name is None:
                pat = re.compile(rf"^\s*({subroutine.name})\s*(.*?)$", re.MULTILINE)
                self.subroutine_patterns[subroutine.name,] = pat
            else:
                pat = re.compile(
                    rf"^\s*({subroutine.name})\s*(.*?)^({subroutine.end_name})\s*$",
                    re.DOTALL | re.MULTILINE,
                )
                self.subroutine_patterns[subroutine.name] = pat
        if hasattr(self.config, "submit_command_end_name"):
            submit_pat = re.compile(
                rf"^\s*({self.config.submit_command})\s*(.*?)^({self.config.submit_command_end_name})\s*$",
                re.DOTALL | re.MULTILINE,
            )
        else:
            submit_pat = re.compile(rf"^\s*({self.config.submit_command})(\s*)$", re.MULTILINE)  # group 2 is nothing
        self.subroutine_patterns[self.config.submit_command] = submit_pat
        self.command_patterns[self.config.submit_command] = submit_pat

    def forward(self, observation: str | None, available_actions: list[str], state: str) -> tuple[str, str, str]:
        """Forwards the model

        Args:
            observation: Observation
            available_actions: Currently not used
            state:

        Returns:
            thought: model reasoning
            action: action that the model proposes
            output: raw model output (not output of the action)
        """
        thought, action, output = self.forward_with_error_check(observation, state)

        history_item = {
            "role": "assistant",
            "content": output,
            "thought": getattr(self.model, "last_thought", "") or thought,
            "context_compressed": self._last_context_compressed,
            "action": action,
            "agent": self.name,
            **self._model_response_metadata(),
        }
        self._append_history(history_item)

        self.logger.info(f"濠电姷顣藉Σ鍛村磻閹捐泛绶ゅù鐘差儏閻ゎ喗銇勯弽銊х焼闁?THOUGHT ({self.name})\n{thought}")
        self.logger.info(f"濠电姷顣藉Σ鍛村磻閹捐泛绶ゅù鐘差儏閻ゎ喗銇勯幇鈺佲偓妤佺▔?ACTION ({self.name})\n{action}")

        return thought, action, output

    def forward_model(self, observation: str | None, state: str) -> str:
        """Query the model with the current state and observation with the appropriate template.

        Returns:
            output: raw model output (not output of the command)
        """
        assert self.config is not None  # mypy
        state_vars = json.loads(normalize_state(state))

        templates: list[str] = []
        # Determine observation template based on what prior observation was
        if self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False):
            # Show instance template if prev. obs. was initial system message
            templates = [self.config.instance_template]
            if self.config.strategy_template is not None:
                templates.append(self.config.strategy_template)
        elif observation is None or observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [self.config.next_step_no_output_template]
        else:
            # Show standard output template if there is observation content
            templates = [self.config.next_step_template]

        # Populate selected template(s) with information (e.g., issue,
        # arguments, state).  Native tool_result messages carry the complete
        # observation separately, so callers can render the same template
        # with an empty observation for the status-only text block.
        def render_messages(observation_text: str) -> str:
            return "\n".join(
                template.format(
                    **self.instance_args,
                    **self.system_args,
                    **state_vars,
                    observation=observation_text,
                    **self._forwarded_vars,
                )
                for template in templates
            )

        message = render_messages(observation if observation is not None else "")

        self.logger.info(f"濠电姷顣藉Σ鍛村磻閹捐泛绶ゅù鐘差儏缁愭鎱ㄥ鍡楀姦?MODEL INPUT\n{message}")
        # For a native Anthropic tool call, return the real environment
        # observation as a tool_result block. The accompanying text block is
        # deliberately status-only: putting ``{observation}`` in both blocks
        # duplicates large outputs (notably decompiler output) in the next
        # model request and can push GLM into a long-thinking timeout.
        previous = self.history[-1] if self.history else {}
        previous_blocks = previous.get("content_blocks", []) if isinstance(previous, dict) else []
        native_block = next((b for b in previous_blocks
                             if isinstance(b, dict) and b.get("type") == "tool_use"), None)
        if (
            native_block
            and observation is not None
            and self.model.args.model_name.startswith("glm52")
            and getattr(self.model, "native_tools", False)
        ):
            status_message = render_messages("")
            history_content: Any = native_tool_result_blocks(
                native_block.get("id", ""), observation, status_message,
            )
            self._append_history({"role": "user", "content": history_content,
                                  "agent": self.name})
        else:
            self._append_history({"role": "user", "content": message, "agent": self.name})

        for hook in self.hooks:
            hook.on_model_query(query=self.local_history, agent=self.name)
        return self._query_model_with_timeout(self.local_history)

    def retry_after_format_fail(self, output: str) -> str:
        """Ask the model to correct (without committing to persistent history) after a malformatted model output"""
        format_error_template = self.config.format_error_template

        self.logger.warning(f"MALFORMED OUTPUT\n{output}")
        self.logger.warning(f"FORMAT ERROR\n{format_error_template}")

        temp_history = self.local_history + [
            {"role": "assistant", "content": output, "agent": self.name},
            {"role": "user", "content": format_error_template, "agent": self.name},
        ]
        return self._query_model_with_timeout(temp_history)

    def retry_after_blocklist_fail(self, output: str, action: str) -> str:
        """Ask the model to correct (without committing to persistent history) after a disallowed command"""
        name = action.strip().split()[0]
        blocklist_error_message = self.config.blocklist_error_template.format(name=name)

        self.logger.warning(f"BLOCKLISTED OUTPUT\n{output}")
        self.logger.warning(f"BLOCKLIST ERROR\n{blocklist_error_message}")

        temp_history = self.local_history + [
            {"role": "assistant", "content": output, "agent": self.name},
            {"role": "user", "content": blocklist_error_message, "agent": self.name},
        ]
        return self._query_model_with_timeout(temp_history)

    def should_block_action(self, action: str) -> bool:
        """Check if the command should be blocked."""
        names = action.strip().split()
        if len(names) == 0:
            return False
        name = names[0]
        if name in self.config.blocklist:
            return True
        if name in self.config.blocklist_standalone and name == action.strip():
            return True
        if name in self.config.block_unless_regex and not re.search(self.config.block_unless_regex[name], action):
            return True
        return False

    def check_format_and_requery(self, output: str) -> tuple[str, str, str]:
        """Parse exactly one shell command; malformed prose is retried, never executed."""
        if self.model.args.model_name == "human":
            if not self._strict_action_is_valid(output):
                raise FormatError("action must be one standalone command")
            return "", output, output
        if self.model.args.model_name == "human_thought":
            thought, action = ParseFunction.get("ThoughtActionParser")(output, self.config._commands + self.config.subroutine_types, strict=True)
            if not self._strict_action_is_valid(action):
                raise FormatError("action must be one standalone command")
            return thought, action, output
        # Native Anthropic Messages response: execute the tool_use directly.
        # This must happen before the legacy Markdown/code-block parser, which
        # otherwise turns a valid structured response into ``exit_format``.
        native_calls = getattr(self.model, "last_tool_calls", [])
        if native_calls:
            native_action = _tool_call_to_shell_action(native_calls[0])
            if native_action and not self.should_block_action(native_action):
                self.info["native_tool_call"] = True
                self.info["native_tool_name"] = native_calls[0].get("name", "")
                return getattr(self.model, "last_thought", "") or "", native_action, output
            self.info["native_tool_call"] = True
            self.info["last_parse_error"] = "invalid_native_tool_call"
            return "Exit due to invalid native tool call", "exit_format", output
        format_fails = blocklist_fails = 0
        previous_format_output: str | None = None
        while format_fails + blocklist_fails <= 10:
            try:
                thought, action = self.config.parse_function(output, self.config._commands + self.config.subroutine_types, strict=True)
                if not self._strict_action_is_valid(action):
                    raise FormatError("action must be one standalone command")
            except KeyboardInterrupt:
                raise
            except FormatError:
                format_fails += 1
                self.info["parser_retry_count"] = format_fails
                self.info["last_parse_error"] = "parse_error"
                if output == previous_format_output:
                    self.info["parser_stuck"] = True
                    self.info["last_parse_error"] = "repeated_parse_error"
                    break
                previous_format_output = output
                if format_fails > FORMAT_RETRY_LIMIT:
                    break
                output = self.retry_after_format_fail(output)
                continue
            if self.should_block_action(action):
                blocklist_fails += 1
                output = self.retry_after_blocklist_fail(output, action)
            else:
                self.info["parser_retry_count"] = format_fails
                return thought, action, output
        self.logger.warning(f"Malformat limit reached: \n{output}")
        self.info["parser_retry_count"] = format_fails
        self.info["last_parse_error"] = "parse_error"
        return "Exit due to format error", "exit_format", output
    def forward_with_error_check(self, observation: str | None, state: str) -> tuple[str, str, str]:
        """Wrapper around `self.forward_model` that handles errors and retries
        due to format errors or blocked actions.

        Returns:
            thought: model reasoning
            action: action that the model proposes
            output: raw model output
        """
        try:
            return self.check_format_and_requery(self.forward_model(observation, state))
        except KeyboardInterrupt:
            raise
        except EmptyModelResponseError as e:
            self.logger.warning(f"Model API returned no usable response: {e}")
            return (
                f"Exit due to model API response error: {e}",
                "exit_api",
                f"model API response error: {e}",
            )
        except RuntimeError as e:
            self.logger.warning(f"Runtime error: {e}")
            return (
                f"Exit due to runtime error: {e}",
                "exit_error",
                f"exit due to runtime error: {e}",
            )
        except ContextWindowExceededError:
            self.logger.warning("Context window exceeded")
            return "Exit due to context window", "exit_context", "Exit due to context window"
        except CostLimitExceededError:
            self.logger.warning("Cost limit exceeded")
            return "Exit due to cost limit", "exit_cost", "Exit due to cost limit"
        except RetryError as e:
            self.logger.warning(f"Retry error: {e}")
            return (
                f"Exit due to retry error: {e}",
                "exit_api",
                f"exit due to retry error: {e}",
            )

    def init_environment_vars(self, env: SWEEnv):
        assert self.config is not None
        self.set_environment_vars(env, self.config.env_variables)

    def set_environment_vars(self, env: SWEEnv, env_variables: dict[str, Any]) -> None:
        """Sets environment variables in the container and for example makes sure
        that all the commands are available in the PATH on the container.
        """
        assert self.config is not None  # mypy
        commands_to_execute = (
            [self.config.state_command.code]
            +
            # [code for code in self.config.util_functions] +
            # [command.code for command in self.config._commands] +
            [f"{k}={v}" for k, v in env_variables.items()]
        )
        commands = "\n".join(commands_to_execute)
        
        # Log the state command setup for debugging
        self.logger.debug(f"Setting up state command in container: {self.config.state_command.name}")
        self.logger.debug(f"State command code: {self.config.state_command.code}")
        
        try:
            output = env.communicate(commands)
            if env.returncode != 0:
                msg = f"Nonzero return code: {env.returncode}\nOutput: {output}"
                self.logger.error(f"Failed to set up environment variables and state command: {msg}")
                raise RuntimeError(msg)
            else:
                self.logger.debug("Environment variables and state command setup completed successfully")
                
                # Test if the state command is properly defined
                try:
                    test_output = env.communicate("type state")
                    if "state is a function" in test_output or "state ()" in test_output:
                        self.logger.debug("State command successfully defined as a function")
                    else:
                        self.logger.warning(f"State command may not be properly defined: {test_output}")
                except Exception as test_e:
                    self.logger.debug(f"Could not test state command definition: {test_e}")
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Failed to set environment variables: {traceback.format_exc()}")
            raise e
        command_files = list()
        for file in self.config.command_files:
            datum = dict()
            with open(file) as f:
                contents = f.read()
            datum["contents"] = contents
            filename = Path(file).name
            if not contents.strip().startswith("#!"):
                if filename.endswith(".sh"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "source_file"
                elif filename.startswith("_"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "utility"
                else:
                    msg = (
                        f"Non-shell script file {file} does not start with shebang.\n"
                        "Either add a shebang (#!) or change the file extension to .sh if you want to source it.\n"
                        "You can override this behavior by adding an underscore to the file name (e.g. _utils.py)."
                    )
                    raise ValueError(msg)
            else:
                # scripts are made executable
                datum["name"] = Path(file).name.rsplit(".", 1)[0]
                datum["type"] = "script"
            command_files.append(datum)
        env.add_commands(command_files)

    def get_environment_vars(self, env: SWEEnv) -> dict[str, Any]:
        """Get environment variables inside of the container"""
        assert self.config is not None  # mypy
        env_vars = dict()
        for var in self.config.env_variables:
            env_vars[var] = env.communicate(f"echo ${var}").strip()
        return env_vars

    def call_subroutine(self, agent_name: str, sub_action: SubAction, env: SWEEnv):
        """Call subroutine"""
        assert self.config is not None  # mypy
        env_vars = self.get_environment_vars(env)
        cwd = env.communicate("pwd -P").strip()
        init_observation = self.config._subroutines[agent_name].init_observation
        if init_observation is not None:
            obs, _, _, _ = env.step(init_observation.format(args=sub_action["args"]))
        else:
            obs = None
        if env.returncode != 0:
            self._append_history(HistoryItem({"role": "user", "content": obs, "agent": agent_name}))
            msg = f"Nonzero return code: {env.returncode} for init_observation in {agent_name}.\n{obs}"
            raise RuntimeError(msg)
        return_type = self.config._subroutines[agent_name].return_type
        sub_agent = Agent(agent_name, self.config._subroutines[agent_name].agent_args)
        sub_agent_output = sub_agent.run(
            {"issue": sub_action["args"]},
            env,
            observation=obs,
            return_type=return_type,
            init_model_stats=self.model.stats,
        )
        self.history += sub_agent.history
        self.set_environment_vars(env, env_vars)
        env.communicate(f"cd {cwd}")
        self.model.stats.replace(sub_agent.model.stats)
        return sub_agent_output

    def _update_summarizer_stats(self, cost: APIStats):
        """Update stats for summarizer"""
        self.model.stats += cost
        if "summarizer" not in self.info:
            self.info["summarizer"] = {
                "model_stats": APIStats().to_dict(),
                "n_calls": 0,
            }
        total_cost = APIStats(**self.info["summarizer"]["model_stats"])
        total_cost += cost
        self.info["summarizer"]["model_stats"] = total_cost.to_dict()
        self.info["summarizer"]["n_calls"] += 1

    def _run_sub_action(self, sub_action: SubAction) -> tuple[str | None, bool]:
        """Execute a sub-action. If the sub-action is a command, execute it.
        If it is a subroutine, call the subroutine.

        Returns:
            observation: Observation
            done: Whether `submit` or another exit reason was called
        """
        assert self._env is not None
        assert self.config is not None
        if sub_action["agent"] == self.name or sub_action["cmd_name"] == self.config.submit_command:
            # Normal command, not a subroutine
            for hook in self.hooks:
                hook.on_sub_action_started(sub_action=sub_action)
            observation, _, done, _info = self._env.step(sub_action["action"])
            observation, additional_cost = self.config.summarizer_config.function(  # type: ignore
                sub_action["action"], observation, self._env, self.summarizer_model
            )
            self._update_summarizer_stats(additional_cost)
            self.info.update(_info)
            self.info["return_code"] = _info.get("return_code") if isinstance(_info, dict) else None
            for hook in self.hooks:
                hook.on_sub_action_executed(obs=observation, done=done)
        else:
            agent_name = sub_action["agent"]
            sub_agent_output = self.call_subroutine(agent_name, sub_action, self._env)
            observation = sub_agent_output
            assert isinstance(observation, str) or observation is None
            done = False
        return observation, done

    def _run_step(self, observation: str | None) -> tuple[str | None, bool]:
        """Run a step of the agent (forward, execute, and save).

        Returns:
            observation: Observation
            done: Whether `submit` or another exit reason was called
        """

        assert self.config is not None  # mypy
        assert self._env is not None

        for hook in self.hooks:
            hook.on_step_start()

        def environment_alive() -> bool:
            env = self._env
            if env.container is None or env.container.poll() is not None:
                return False
            try:
                if env.container_obj is not None:
                    env.container_obj.reload()
                    if env.container_obj.status in {"dead", "exited", "stopped", "removing"}:
                        return False
            except Exception:
                return False
            return True

        if self.state_command:
            try:
                state = self._env.communicate(self.state_command)
                state_text = (
                    state.decode("utf-8", errors="replace")
                    if isinstance(state, bytes) else state
                )
                # Log if state command returns empty for debugging
                if not state_text or not state_text.strip():
                    self.logger.warning("State command returned empty output. This may indicate an issue with environment setup.")
                    if not environment_alive():
                        reason = "shell/container became unavailable while reading agent state"
                        self.finalize_episode("environment_error", reason, "container_unavailable")
                        return None, True
                    state = normalize_state(None)
                elif not state_text.lstrip().startswith('{'):
                    # Normalize malformed output through the same fallback as
                    # empty, incomplete, and communication-error states.
                    self.logger.warning(f"State command returned non-JSON output: {state_text[:100]}...")
                    if not environment_alive():
                        reason = "shell/container became unavailable while reading agent state"
                        self.finalize_episode("environment_error", reason, "container_unavailable")
                        return None, True
                    state = normalize_state(state_text)
            except Exception as e:
                self.logger.warning(f"Failed to execute state command: {e}")
                if not environment_alive():
                    reason = f"shell/container unavailable while reading agent state: {e}"
                    self.finalize_episode("environment_error", reason, "container_unavailable")
                    return None, True
                state = normalize_state(None)
        else:
            state = normalize_state(None)
        state = normalize_state(state)
        thought, action, output = self.forward(observation, self._env.get_available_actions(), state)
        for hook in self.hooks:
            hook.on_actions_generated(thought=thought, action=action, output=output)
        run_action: str = self._guard_multiline_input(action)

        # Loop over sub-actions (if any)
        done = False
        observations: list[str | None] = list()
        execution_t0 = time.perf_counter()
        for sub_action in self.split_actions(run_action):
            observation, done = self._run_sub_action(sub_action)
            # Terminal observations carry verifier and environment feedback.
            # Preserve them in the trajectory before ending the step.
            observations.append(observation)
            if done:
                break
        observation = "\n".join([obs for obs in observations if obs is not None])
        execution_time = time.perf_counter() - execution_t0

        tool_name, tool_args = self._structured_action(action)
        terminal_error = "parse_error" if action == "exit_format" else ""
        trajectory_step = TrajectoryStep(
            {
                "step_id": len(self.trajectory) + 1,
                "action": action,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "observation": observation,
                "return_code": self.info.get("return_code"),
                "response": output,
                "state": state,
                "thought": getattr(self.model, "last_thought", "") or thought,
                **self._model_response_metadata(),
                "context_compressed": self._last_context_compressed,
                "execution_time": execution_time,
                "error": terminal_error,
                "terminal": done,
            },
        )
        self.trajectory.append(trajectory_step)
        model_stats: APIStats = self.model.stats
        self.info["model_stats"] = model_stats.to_dict()
        for hook in self.hooks:
            hook.on_step_done(trajectory_step=trajectory_step, model_stats=model_stats)
        return observation, done

    def run(
        self,
        setup_args: dict[str, Any],
        env: SWEEnv,
        observation: str | None = None,
        traj_dir: Path | None = None,
        return_type: str = "info_trajectory",
        init_model_stats: APIStats | None = None,
    ):
        """Run an episode and always persist a terminal, analysis-ready trajectory."""
        self._episode_finalized = False
        # Establish a writable trajectory context before container/setup assertions can fail.
        self._env = env
        self.traj_dir = traj_dir
        self.trajectory = Trajectory()
        self.info = AgentInfo()
        try:
            assert env.record is not None
            assert env.container_obj is not None
            if env.container_obj.id != self.last_container_id:
                self.logger.info(f"Initializing agent settings for container {env.container_obj.id}")
                self.init_environment_vars(env)
                self.last_container_id = env.container_obj.id
            self.setup(setup_args, init_model_stats)
            self.config.summarizer_config.function.setup(setup_args, self.config)
            self.trajectory = Trajectory()
            self._env = env
            self.info = AgentInfo()
            self.traj_dir = traj_dir
            self.logger.info("Trajectory will be saved to %s", self.traj_path)
            for hook in self.hooks:
                hook.on_run_start()
            done = False
            task_start_time = time.time()
            while not done:
                elapsed_time = time.time() - task_start_time
                if elapsed_time > TASK_EXECUTION_TIMEOUT:
                    reason = f"Task exceeded {TASK_EXECUTION_TIMEOUT}s timeout"
                    self.finalize_episode("task_timeout", reason, "task_timeout")
                    done = True
                    break
                observation, done = self._run_step(observation)
                self.save_trajectory()
            exit_status = self.info.get("exit_status")
            if not isinstance(exit_status, str) or not exit_status:
                exit_status = "early_exit"
                reason = "Agent loop ended without an environment exit status"
            else:
                reason = str(self.info.get("terminal_reason") or f"Agent exited with {exit_status}")
            self.finalize_episode(exit_status, reason, self.info.get("error_type") or None)
            for hook in self.hooks:
                hook.on_run_done(trajectory=self.trajectory, info=self.info)
        except KeyboardInterrupt:
            if hasattr(self, "info") and hasattr(self, "trajectory"):
                self.finalize_episode("early_exit", "Episode interrupted by user", "KeyboardInterrupt")
            raise
        except BaseException as error:
            if hasattr(self, "info") and hasattr(self, "trajectory"):
                self.info["traceback"] = traceback.format_exc()
                if isinstance(error, TimeoutError):
                    self.finalize_episode("model_timeout", str(error), "model_timeout")
                else:
                    self.finalize_episode("runner_exception", str(error), type(error).__name__)
            raise
        self.logger.info("Trajectory saved to %s", self.traj_path)
        if return_type == "info":
            return self.info
        if return_type == "info_trajectory":
            return self.info, self.trajectory
        return self.trajectory[-1][return_type]
