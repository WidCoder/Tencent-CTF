# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


#
from __future__ import annotations

import logging
import time

from sweagent import CONFIG_DIR
from sweagent.utils.log import add_file_handler, get_logger

try:
    import rich
except ModuleNotFoundError as e:
    msg = (
        "You probably either forgot to install the dependencies "
        "or forgot to activate your conda or virtual environment."
    )
    raise RuntimeError(msg) from e
import json
import re
import subprocess
import traceback
from typing import Any

import rich.console
import rich.markdown
import rich.panel

try:
    from rich_argparse import RichHelpFormatter
except ImportError:
    msg = "Please install the rich_argparse package with `pip install rich_argparse`."
    raise ImportError(msg)
import datetime
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path

import yaml
from rich.markdown import Markdown
from simple_parsing import parse
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from unidiff import PatchSet

from sweagent.agent.agents import Agent, AgentArguments
from sweagent.agent.models import ModelArguments
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from sweagent.environment.utils import (
    InvalidGithubURL,
    extract_flag_format,
    get_associated_commit_urls,
    get_data_path_name,
    get_gh_issue_data,
    parse_gh_issue_url,
)

__doc__: str = """ Run inference. Usage examples:

```bash
# Run over a github issue:
python run.py --model_name "gpt4" --data_path "https://github.com/pvlib/pvlib-python/issues/1603" --config_file "config/default_from_url.yaml"
# Apply a patch in a local repository to an issue specified as Markdown file and run a custom installer script in the container
python run.py --model_name "gpt4" --data_path "/path/to/my_issue.md" --repo_path "/path/to/my/local/repo" --environment_setup "/path/to/setup.sh" --config_file "config/default_from_url.yaml" --apply_patch_locally
```

**For more information**: https://princeton-nlp.github.io/SWE-agent/usage/cl_tutorial/
"""


logger = get_logger("swe-agent-run")
logging.getLogger("simple_parsing").setLevel(logging.WARNING)


@dataclass(frozen=True)
class ActionsArguments(FlattenedAccess, FrozenSerializable):
    """Run real-life actions (opening PRs, etc.) if we can solve the issue."""

    # Open a PR with the patch if we can solve the issue
    open_pr: bool = False
    # When working with local repository: Apply patch
    apply_patch_locally: bool = False
    # Option to be used with open_pr: Skip action if there are already commits claiming
    # to fix the issue. Please only set this to False if you are sure the commits are
    # not fixes or if this is your own repository!
    skip_if_commits_reference_issue: bool = True
    # OBSOLETE. Do not use, will raise error. Please specify --repo_path instead.
    push_gh_repo_url: str = ""

    def __post_init__(self):
        if self.push_gh_repo_url:
            msg = "push_gh_repo_url is obsolete. Use repo_path instead"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the run.py script"""

    environment: EnvironmentArguments
    agent: AgentArguments
    actions: ActionsArguments
    # Only run instances that completely match this regex
    instance_filter: str = ".*"
    # Skip instances with existing trajectories
    skip_existing: bool = True
    # Suffix for the run name (used for example in trajectory directory naming)
    suffix: str = ""
    # Raise unhandled exceptions during the run (useful for debugging)
    raise_exceptions: bool = False
    # Dump the entire config to the log
    print_config: bool = True
    # Run the agent in CTF mode (SWE-agent: EnIGMA)
    ctf: bool = True
    # Custom trajectory output path (if not provided, uses default: trajectories/{username}/{run_name})
    trajectory_path: str = ""
    # Bypass step hit from previous history and remove trajectory to overwrite (unless flag was captured)
    bypass_step_limit_history: bool = False
    # Start container only without running agents (for debugging/manual interaction)
    container_only: bool = False
    # Writeup content to append to task description as a hint (avoid mentioning this explicitly during interaction)
    writeup: str = ""
    workers: int = 1
    trajectories_per_task: int = 1
    num_trajectories: int = 1

    @property
    def run_name(self) -> str:
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        data_stem = get_data_path_name(self.environment.data_path)
        assert self.agent.config_file is not None  # mypy
        config_stem = Path(self.agent.config_file).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p
        top_k = self.agent.model.top_k

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = self.environment.install_environment

        return (
            f"{model_name}__{data_stem}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}__k-{top_k}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )


class _ContinueLoop(Exception):
    """Used for internal control flow"""


class MainHook:
    """Hook structure for the web server or other addons to interface with"""

    @staticmethod
    def _is_promising_patch(info: dict[str, Any]) -> bool:
        """Do we actually believe that the patch will solve the issue?
        Or are we just submitting the last patch we generated before hitting an error?
        """
        # The exit status can also be `submitted (exit_cost)` etc.
        return info["exit_status"] == "submitted" and info.get("submission") is not None

    def on_init(self, *, args: ScriptArguments, agent: Agent, env: SWEEnv, traj_dir: Path):
        """Called when hook is initialized"""

    def on_start(self):
        """Called at the beginning of `Main.main`"""

    def on_end(self):
        """Called at the end of `Main.main`"""

    def on_instance_start(self, *, index: int, instance: dict[str, Any]):
        """Called at the beginning of each instance loop in `Main.run`"""

    def on_instance_skipped(
        self,
    ):
        """Called when an instance is skipped in `Main.run`"""

    def on_instance_completed(self, *, info, trajectory):
        """Called when an instance is completed in `Main.run`"""


class SaveApplyPatchHook(MainHook):
    """This hook saves patches to a separate directory and optionally applies them to a local repository."""

    def on_init(self, *, args: ScriptArguments, agent: Agent, env: SWEEnv, traj_dir: Path):
        self._traj_dir = traj_dir
        self._apply_patch_locally = args.actions.apply_patch_locally
        self._instance = None

    def on_instance_start(self, *, index: int, instance: dict[str, Any]):
        self._instance = instance

    def on_instance_completed(self, *, info, trajectory):
        assert self._instance is not None  # mypy
        instance_id = self._instance["instance_id"]
        patch_path = self._save_patch(instance_id, info)
        if patch_path:
            if not self._apply_patch_locally:
                return
            if not self._is_promising_patch(info):
                return
            assert self._instance  # mypy
            if self._instance["repo_type"] != "local":
                return
            local_dir = Path(self._instance["repo"])
            self._apply_patch(patch_path, local_dir)

    @staticmethod
    def _print_patch_message(patch_output_file: Path):
        console = rich.console.Console()
        msg = [
            "SWE-agent has produced a patch that it believes will solve the issue you submitted!",
            "Use the code snippet below to inspect or apply it!",
        ]
        panel = rich.panel.Panel.fit(
            "\n".join(msg),
            title="Submission successful",
        )
        console.print(panel)
        content = [
            "```bash",
            "# The patch has been saved to your local filesystem at:",
            f"PATCH_FILE_PATH='{patch_output_file.resolve()}'",
            "# Inspect it:",
            'cat "${PATCH_FILE_PATH}"',
            "# Apply it to a local repository:",
            "cd <your local repo root>",
            'git apply "${PATCH_FILE_PATH}"',
            "```",
        ]
        console.print(rich.markdown.Markdown("\n".join(content)))

    def _save_patch(self, instance_id: str, info) -> Path | None:
        """Create patch files that can be applied with `git am`.

        Returns:
            The path to the patch file, if it was saved. Otherwise, returns None.
        """
        patch_output_dir = self._traj_dir / "patches"
        patch_output_dir.mkdir(exist_ok=True, parents=True)
        patch_output_file = patch_output_dir / f"{instance_id}.patch"
        if info.get("submission") is None:
            logger.info("No patch to save.")
            return None
        model_patch = info["submission"]
        patch_output_file.write_text(model_patch)
        if self._is_promising_patch(info):
            # Only print big congratulations if we actually believe
            # the patch will solve the issue
            self._print_patch_message(patch_output_file)
        return patch_output_file

    def _apply_patch(self, patch_file: Path, local_dir: Path) -> None:
        """Apply a patch to a local directory."""

        assert local_dir.is_dir()
        assert patch_file.exists()
        # The resolve() is important, because we're gonna run the cmd
        # somewhere else
        cmd = ["git", "apply", str(patch_file.resolve())]
        try:
            subprocess.run(cmd, cwd=local_dir, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply patch {patch_file} to {local_dir}: {e}")
            return
        logger.info(f"Applied patch {patch_file} to {local_dir}")


class OpenPRHook(MainHook):
    """This hook opens a PR if the issue is solved and the user has enabled the option."""

    def on_init(self, *, args: ScriptArguments, agent: Agent, env: SWEEnv, traj_dir: Path):
        self._env = env
        self._token: str = env._github_token
        self._data_path = args.environment.data_path
        self._open_pr = args.actions.open_pr
        self._skip_if_commits_reference_issue = args.actions.skip_if_commits_reference_issue

    def on_instance_completed(self, *, info, trajectory):
        if self._open_pr and self.should_open_pr(info):
            self._env.open_pr(trajectory=trajectory)

    def should_open_pr(self, info: dict[str, Any]) -> bool:
        """Does opening a PR make sense?"""
        if not info.get("submission"):
            logger.info("Not opening PR because no submission was made.")
            return False
        if info["exit_status"] != "submitted":
            logger.info("Not opening PR because exit status was %s and not submitted.", info["exit_status"])
            return False
        try:
            issue = get_gh_issue_data(self._data_path, token=self._token)
        except InvalidGithubURL:
            logger.info("Currently only GitHub is supported to open PRs to. Skipping PR creation.")
            return False
        if issue.state != "open":
            logger.info(f"Issue is not open (state={issue.state}. Skipping PR creation.")
            return False
        if issue.assignee:
            logger.info("Issue is already assigned. Skipping PR creation. Be nice :)")
            return False
        if issue.locked:
            logger.info("Issue is locked. Skipping PR creation.")
            return False
        org, repo, issue_number = parse_gh_issue_url(self._data_path)
        associated_commits = get_associated_commit_urls(org, repo, issue_number, token=self._token)
        if associated_commits:
            commit_url_strs = ", ".join(associated_commits)
            if self._skip_if_commits_reference_issue:
                logger.info(f"Issue already has associated commits (see {commit_url_strs}). Skipping PR creation.")
                return False
            else:
                logger.warning(
                    "Proceeding with PR creation even though there are already commits "
                    f"({commit_url_strs}) associated with the issue. Please only do this for your own repositories "
                    "or after verifying that the existing commits do not fix the issue.",
                )
        return True


class Main:
    def __init__(self, args: ScriptArguments):
        if args.trajectory_path:
            self.traj_dir = Path(args.trajectory_path) / args.run_name
        else:
            self.traj_dir = Path("trajectories") / Path(getuser()) / args.run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        log_path = self.traj_dir / f"run-{timestamp}.log"
        logger.info("Logging to %s", log_path)
        add_file_handler(log_path)
        if args.print_config:
            logger.info(f"濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬶拷鎰垫線缁墎绱?Arguments: {args.dumps_yaml()}")
        self.args = args
        self.agent = Agent("primary", args.agent)
        self.env = SWEEnv(args.environment)
        self._save_arguments()
        default_hooks = [
            SaveApplyPatchHook(),
            OpenPRHook(),
        ]
        self.hooks: list[MainHook] = []
        for hook in default_hooks:
            self.add_hook(hook)

    def add_hook(self, hook: MainHook):
        hook.on_init(args=self.args, agent=self.agent, env=self.env, traj_dir=self.traj_dir)
        self.hooks.append(hook)

    def run(self, index: int) -> None:
        # Reset environment
        instance_id = self.env.data[index]["instance_id"]
        for hook in self.hooks:
            hook.on_instance_start(index=index, instance=self.env.data[index])
        assert isinstance(instance_id, str)  # mypy
        if self.should_skip(instance_id):
            for hook in self.hooks:
                hook.on_instance_skipped()
            raise _ContinueLoop
        logger.info("闂傚倸鍊搁崐鎼佸磹閹间礁纾圭€瑰嫭鍣磋ぐ鎺戠倞鐟滃寮稿澶嬬厸闁搞儮鏅涢埀锟介敐澶嬫櫜濠㈣泛锕﹂濠傗攽鎺抽崐鏇㈡晝閵夆晛鐤鹃柍鍝勬噺閳锋垹鎲搁悧鍫濈瑨濞存粈鍗抽弻娑㈡偐閾忥拷閻ㄦ垹绱? Beginning task " + str(index))

        observation, info = self.env.reset(index)
        if info is None:
            info = {"exit_status": "environment_reset_failed", "error_message": "environment reset returned no metadata"}
        if info.get("exit_status"):
            trajectory_path = self.traj_dir / f"{instance_id}.traj"
            info.update({"terminal_reason": info.get("error_message", "environment reset failed"), "error_type": "environment_error",
                         "flag_submitted": False, "flag_verified": "unknown", "episode_end_time": datetime.datetime.now(datetime.timezone.utc).isoformat(), "total_steps": 1})
            payload = {"environment": self.env.name, "trajectory": [{"step_id": 1, "thought": "", "action": "environment_reset",
                       "tool_name": "unknown", "tool_args": "", "observation": info["terminal_reason"], "return_code": None,
                       "execution_time": 0.0, "error": "environment_error", "terminal": True, "response": ""}], "history": [], "info": info}
            trajectory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.warning("Environment initialization failed for %s: %s", instance_id, info["exit_status"])
            self._save_predictions(instance_id, info, self.env.challenge)
            for hook in self.hooks:
                hook.on_instance_completed(info=info, trajectory=[])
            return False
        # Get info, patch information
        issue = getattr(self.env, "query", None)
        files = []
        assert self.env.record is not None  # mypy
        if "patch" in self.env.record:
            files = "\n".join([f"- {x.path}" for x in PatchSet(self.env.record["patch"]).modified_files])
        # Get test files, F2P tests information
        test_files = []
        if "test_patch" in self.env.record:
            test_patch_obj = PatchSet(self.env.record["test_patch"])
            test_files = "\n".join([f"- {x.path}" for x in test_patch_obj.modified_files + test_patch_obj.added_files])
        tests = ""
        if "FAIL_endTO_PASS" in self.env.record:
            tests = "\n".join([f"- {x}" for x in self.env.record["FAIL_TO_PASS"]])

        setup_args = {"issue": issue, "files": files, "test_files": test_files, "tests": tests}
        challenge = self.env.challenge
        if challenge is not None:
            flag_value = challenge.get("flag")

            if flag_value:
                setup_args["flag_format"] = extract_flag_format(flag_value)
            else:
                setup_args["flag_format"] = challenge.get("flag_format", ".*")

            setup_args["name"] = challenge["name"]

            # Handle description with optional writeup content
            description = challenge["description"]

            if self.args.writeup:
                description += (
                    "\n\n<<<INTERNAL_HINT_SPLITTER_START>>>\n"
                    f"{self.args.writeup}"
                    "\n<<<INTERNAL_HINT_SPLITTER_END>>>\n\n"
                )

            setup_args["description"] = description
            setup_args["category_friendly"] = challenge.get(
                "category_friendly",
                challenge.get("category", ""),
            )
            setup_args["points"] = challenge.get("points", 0)
            setup_args["files"] = (
                challenge.get("files")
                or "No files included in this challenge."
            )
            setup_args["box"] = challenge.get("server_name")
            setup_args["port"] = challenge.get("port")
            setup_args["server_description"] = challenge.get(
                "server_description"
            )

        # Get step limit from model arguments
        step_limit = self.args.agent.model.per_instance_step_limit
        # Create a custom version of the Agent.run method that enforces step limits
        if step_limit > 0:
            original_run = self.agent.run
            
            def limited_run(*args, **kwargs):
                # Create a wrapper around the agent._run_step method to count steps
                original_run_step = self.agent._run_step
                step_count = 0
                
                def step_limited_run_step(observation):
                    nonlocal step_count
                    observation, done = original_run_step(observation)
                    step_count += 1
                    
                    if not done and step_count >= step_limit:
                        logger.info(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾瑰瀣捣閻棗銆掑锝呬壕閻庤娲﹂崹璺虹暦缁嬭鏃€绻濋崘鈺傜彣濠碉紕鍋戦崐鏍箰妤ｅ啫纾婚柟鐗堟緲閸戠姷鈧箍鍎遍ˇ浼存偂韫囨稓鍙撻柛銉戝苯鍓伴梺璇叉禋閸犳氨妲?Step limit of {step_limit} reached after {step_count} steps")
                        # Set exit status in agent's info dict
                        self.agent.info["exit_status"] = f"step_{step_limit}_hit"
                        # Force early termination
                        return observation, True
                    
                    return observation, done
                
                # Replace the _run_step method with our counting version
                self.agent._run_step = step_limited_run_step
                
                try:
                    # Run the agent with our modified _run_step method
                    result = original_run(*args, **kwargs)
                    return result
                finally:
                    # Restore original method even if there was an exception
                    self.agent._run_step = original_run_step
            
            # Replace the run method with our limited version
            self.agent.run = limited_run
        
        try:
            # Run the agent
            info, trajectory = self.agent.run(
                setup_args=setup_args,
                env=self.env,
                observation=observation,
                traj_dir=self.traj_dir,
                return_type="info_trajectory",
            )
        finally:
            # Restore original run method if we modified it
            if step_limit > 0:
                self.agent.run = original_run
            
        self._save_predictions(instance_id, info, challenge)
        for hook in self.hooks:
            hook.on_instance_completed(info=info, trajectory=trajectory)

    def _save_runner_error_trajectory(self, index: int, error: BaseException) -> None:
        """Fallback persistence for errors raised outside Agent.run."""
        instance_id = self.env.data[index].get("instance_id", f"instance-{index}")
        path = self.traj_dir / f"{instance_id}.traj"
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        trajectory = payload.get("trajectory")
        if not isinstance(trajectory, list):
            trajectory = []
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        # Agent.run already persisted a terminal record; do not overwrite it or add a second terminal step.
        if info.get("exit_status") and any(isinstance(step, dict) and step.get("terminal") is True for step in trajectory):
            return
        traceback_text = traceback.format_exc()
        trajectory.append({"step_id": len(trajectory) + 1, "thought": "", "action": "runner_exception",
            "tool_name": "unknown", "tool_args": "", "observation": str(error), "return_code": None,
            "execution_time": 0.0, "error": type(error).__name__, "terminal": True, "response": ""})
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        info["artifact_type"] = "synthetic_error"
        info["trajectory_schema_version"] = 2
        info.update({"exit_status": "runner_exception", "terminal_reason": str(error), "error_type": type(error).__name__,
                     "traceback": traceback_text, "flag_submitted": bool(info.get("flag_submitted")),
                     "flag_verified": info.get("flag_verified", "unknown"),
                     "episode_end_time": datetime.datetime.now(datetime.timezone.utc).isoformat(), "total_steps": len(trajectory)})
        payload["trajectory_schema_version"] = 2
        payload.update({"environment": self.env.name, "trajectory": trajectory, "history": payload.get("history", []), "info": info})
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    def main(self) -> int:
        failures = 0
        for hook in self.hooks:
            hook.on_start()
        
        # Handle container-only mode
        if self.args.container_only:
            logger.info("濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬶拷鎰垫線濞岋拷闂傦拷?Starting container-only mode...")
            try:
                # Initialize the first instance to set up the container
                index = 0
                logger.info(f"濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬶拷鎰垫線缁墎绱?Loading instance {index} for container setup...")
                observation, info = self.env.reset(index)
                
                if info and "exit_status" in info:
                    logger.error(f"闂傦拷?Failed to initialize environment: {info}")
                    return
                
                logger.info("闂傦拷?Container started successfully!")
                logger.info(f"濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬶拷鎰垫線缁墎绱?Container name: {self.env.container_name}")
                logger.info(f"濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚伴獮鍥煛娴ｆ彃浜炬俊銈呭暟閻棗鈹戦悩鍙夊闁绘挸鍟伴幉绋匡拷濠傜墕缁€鍫ユ煟閺傛娈犳繛锟? Image: {self.env.image_name}")
                
                if self.env.record:
                    logger.info(f"濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬶拷鎰垫線缁墎绱?Repository: {self.env.record.get('repo', 'N/A')}")
                    logger.info(f"濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬶拷鎰垫線缁爼姊?Base commit: {self.env.record.get('base_commit', 'N/A')}")
                
                print("\n" + "=" * 70)
                print("濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚伴獮鍥煛娴ｆ彃浜炬俊銈勮兌閳伙拷?CONTAINER-ONLY MODE: Container ready for manual interaction!")
                print("=" * 70)
                print(f"Container name: {self.env.container_name}")
                print(f"Repository path: /{self.env._repo_name}")
                print("\nTo interact manually, run:")
                print(f"  docker exec -it {self.env.container_name} /bin/bash")
                print("\nTo run commands via the environment object:")
                print("  # In Python console:")
                print("  from sweagent.environment.swe_env import SWEEnv")
                print(f"  # Connect to existing container: {self.env.container_name}")
                print("\nPress Ctrl+C to stop and clean up the container.")
                print("=" * 70)
                
                # Keep the container alive
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("\n濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬪Ω瑜忛悡鈧俊锟?Received interrupt signal, cleaning up...")
                    
            except KeyboardInterrupt:
                logger.info("\n濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬪Ω瑜忛悡鈧俊锟?Interrupted by user")
            except Exception as e:
                logger.error(f"闂傦拷?Error in container-only mode: {e}")
                raise
            finally:
                logger.info("濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌?Cleaning up container...")
                self.env.close()
                logger.info("闂傦拷?Container cleaned up")
                for hook in self.hooks:
                    hook.on_end()
            return
        
        # Normal mode: run agents on instances
        for index in range(len(self.env.data)):
            try:
                if self.run(index) is False:
                    failures += 1
            except _ContinueLoop:
                continue
            except KeyboardInterrupt:
                logger.info("Exiting InterCode environment...")
                self.env.close()
                break
            except SystemExit as error:
                failures += 1
                self._save_runner_error_trajectory(index, error)
                logger.critical("SystemExit captured; saved error trajectory and continuing batch")
                self.env.reset_container()
                continue
            except Exception as e:
                failures += 1
                self._save_runner_error_trajectory(index, e)
                logger.warning(traceback.format_exc())
                if self.args.raise_exceptions:
                    self.env.close()
                    raise e
                if self.env.record:
                    logger.warning(f"闂傦拷?Failed on {self.env.record['instance_id']}: {e}")
                else:
                    logger.warning("闂傦拷?Failed on unknown instance")
                self.env.reset_container()
                continue
        self.env.close()
        for hook in self.hooks:
            hook.on_end()
        return 1 if failures else 0

    def _save_arguments(self) -> None:
        """Save the arguments to a yaml file to the run's trajectory directory."""
        log_path = self.traj_dir / "args.yaml"

        if log_path.exists():
            try:
                other_args = self.args.load_yaml(log_path)
                if self.args.dumps_yaml() != other_args.dumps_yaml():  # check yaml equality instead of object equality
                    logger.warning("**************************************************")
                    logger.warning("Found existing args.yaml with different arguments!")
                    logger.warning("**************************************************")
            except Exception:
                logger.warning(f"Failed to load existing args.yaml: {traceback.format_exc()}")

        with log_path.open("w") as f:
            self.args.dump_yaml(f)

    def should_skip(self, instance_id: str) -> bool:
        """Check if we should skip this instance based on the instance filter and skip_existing flag."""
        # Skip instances that don't match the instance filter
        if re.match(self.args.instance_filter, instance_id) is None:
            logger.info(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾圭€瑰嫭鍣磋ぐ鎺戠倞鐟滃寮稿澶嬬厸闁搞儯鍎遍悘鈺冪磼閻樿櫕鐨戦柍褜鍓欓崢婊堝磻閹剧粯鐓曢柡鍥ュ妼娴滅偤鏌ｉ幘鏉戠伌婵★拷閸涙惌鏁冮柕蹇娾偓鎰佹П婵犵數鍋涘鍓佸垝鎼达絽鍨?Instance filter not matched. Skipping instance {instance_id}")
            return True

        # If flag is set to False, don't skip
        if not self.args.skip_existing:
            return False

        # Check if there's an existing trajectory for this instance
        log_path = self.traj_dir / (instance_id + ".traj")
        if not log_path.exists():
            return False

        content = log_path.read_text()
        if not content.strip():
            logger.warning("Found empty trajectory: %s. Removing.", log_path)
            log_path.unlink()
            return False

        data = json.loads(content)
        # If the trajectory has no exit status, it's incomplete and we will redo it
        exit_status = data["info"].get("exit_status", None)
        n_calls = data["info"].get("summarizer", {}).get("n_calls", 0)
        
        # Handle bypass_step_limit_history flag
        if self.args.bypass_step_limit_history:
            # Check if previous run hit step limit
            step_limit_hit = exit_status and exit_status.startswith("step_") and exit_status.endswith("_hit")
            
            # If flag was captured (successful completion), still skip
            if exit_status == "submitted":
                logger.info(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾圭€瑰嫭鍣磋ぐ鎺戠倞鐟滃寮稿澶嬬厸闁搞儯鍎遍悘鈺冪磼閻樿櫕鐨戦柍褜鍓欓崢婊堝磻閹剧粯鐓曢柡鍥ュ妼娴滅偤鏌ｉ幘鏉戠伌婵★拷閸涙惌鏁冮柕蹇娾偓鎰佹П婵犵數鍋涘鍓佸垝鎼达絽鍨?Skipping existing trajectory - flag was captured: {log_path}")
                return True
            
            # If step limit was hit, remove trajectory to start fresh
            if step_limit_hit:
                logger.info(f"濠电姷鏁告慨鐑姐€傞挊澹╋綁宕ㄩ弶鎴狅紱闂佽宕樺▔娑氭閵堝悿褰掓偐瀹革拷閸庡繘鏌ｉ妶搴℃灍闁靛洤瀚板浠嬶拷鎰垫線缁爼姊?Bypassing step limit history, removing trajectory: {log_path}")
                log_path.unlink()
                return False
        
        if (exit_status == "early_exit" or exit_status is None) and n_calls < self.args.agent.model.per_instance_step_limit:
            logger.warning(f"Found existing trajectory with no exit status: {log_path}. Removing.")
            log_path.unlink()
            return False

        # Skip if the task was successfully completed (submitted) or if it's a valid exit status
        if exit_status == "submitted" or (exit_status is not None and exit_status != "early_exit"):
            logger.info(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾圭€瑰嫭鍣磋ぐ鎺戠倞鐟滃寮稿澶嬬厸闁搞儯鍎遍悘鈺冪磼閻樿櫕鐨戦柍褜鍓欓崢婊堝磻閹剧粯鐓曢柡鍥ュ妼娴滅偤鏌ｉ幘鏉戠伌婵★拷閸涙惌鏁冮柕蹇娾偓鎰佹П婵犵數鍋涘鍓佸垝鎼达絽鍨?Skipping existing trajectory with exit status '{exit_status}': {log_path}")
            return True

        logger.info(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾圭€瑰嫭鍣磋ぐ鎺戠倞鐟滃寮稿澶嬬厸闁搞儯鍎遍悘鈺冪磼閻樿櫕鐨戦柍褜鍓欓崢婊堝磻閹剧粯鐓曢柡鍥ュ妼娴滅偤鏌ｉ幘鏉戠伌婵★拷閸涙惌鏁冮柕蹇娾偓鎰佹П婵犵數鍋涘鍓佸垝鎼达絽鍨?Skipping existing trajectory: {log_path}")
        return True

    def _save_predictions(self, instance_id: str, info, challenge: dict[str, str] | None):
        output_file = self.traj_dir / "all_preds.jsonl"
        model_patch = info["submission"] if "submission" in info else None
        datum = {
            KEY_MODEL: Path(self.traj_dir).name,
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: model_patch,
        }
        if challenge is not None:
            challenge_datum = {
                "challenge_name": challenge["name"],
                "challenge_category": challenge["category"],
                "challenge_path": challenge["file_path"],
            }
            datum.update(challenge_datum)
        with open(output_file, "a+") as fp:
            print(json.dumps(datum), file=fp, flush=True)
        logger.info(f"Saved predictions to {output_file}")


def get_args(args=None) -> ScriptArguments:
    """Parse command line arguments and return a ScriptArguments object.

    Args:
        args: Optional list of arguments to parse. If not provided, uses sys.argv.
    """
    defaults = ScriptArguments(
        suffix="",
        environment=EnvironmentArguments(
            image_name="sweagent/enigma:latest",
            data_path="",  # No default data path for CTF, must be specified
            split="",
            verbose=True,
            install_environment=True,
            cache_task_images=False,
            enable_network_restrictions=False,
            enable_dynamic_ports=True,
        ),
        skip_existing=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name="gpt4",
                total_cost_limit=0.0,
                per_instance_cost_limit=0.0,  # EnIGMA+ uses turn limits, not cost limits
                temperature=0.0,
                top_p=0.95,
                top_k=20,
                per_instance_step_limit=40,  # EnIGMA+ uses turn limits instead of cost limits
            ),
            config_file=CONFIG_DIR / "default_ctf.yaml",
        ),
        actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
        ctf=True,  # Enable CTF mode by default
        trajectory_path="",
        bypass_step_limit_history=False,
        container_only=False,
        writeup="",
    )

    # Nicer yaml dumping of multiline strings
    def multiline_representer(dumper, data):
        """configures yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, multiline_representer)

    return parse(
        ScriptArguments,
        default=defaults,
        add_config_path_arg=False,
        args=args,
        formatter_class=RichHelpFormatter,
        description=Markdown(__doc__),
    )


def main(args: ScriptArguments) -> int:
    return Main(args).main()


if __name__ == "__main__":
    raise SystemExit(main(get_args()))

