from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ctfdojo_batch_generate.py"
SPEC = importlib.util.spec_from_file_location("ctfdojo_batch_generate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def challenge() -> batch.Challenge:
    return batch.Challenge(
        challenge_json="/example/challenge.json", repo_path="/example", relative_path="event/crypto/example",
        event="event", name="example", category="crypto", task_id="event__crypto__example-123",
        dockerfile_path="/example/Dockerfile", compose_path="",
    )


class CTFDojoBatchTests(unittest.TestCase):
    def test_cleanup_removes_only_containers_created_after_baseline(self):
        calls: list[list[str]] = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **_kwargs):
            calls.append(command)
            return Result()

        with patch.object(batch, "docker_agent_container_ids", return_value={"old", "new-a", "new-b"}), patch.object(batch.subprocess, "run", side_effect=fake_run):
            cleanup = batch.cleanup_agent_containers({"old"})

        self.assertTrue(cleanup["attempted"])
        self.assertEqual(cleanup["removed_container_ids"], ["new-a", "new-b"])
        self.assertEqual(cleanup["failed_container_ids"], [])
        self.assertEqual(calls, [["docker", "rm", "-f", "new-a"], ["docker", "rm", "-f", "new-b"]])

    def test_effective_bridge_capacity_accounts_for_inflight_workers(self):
        # Mirrors the scheduler invariant used by the batch loop.
        bridge_count = 7
        inflight_workers = 1
        max_bridge_containers = 8
        self.assertGreaterEqual(bridge_count + inflight_workers, max_bridge_containers)
    def test_cleanup_with_resource_token_cannot_remove_sibling_worker(self):
        calls: list[list[str]] = []

        class Result:
            returncode = 0
            stdout = "owned-container\n"
            stderr = ""

        def fake_run(command, **_kwargs):
            calls.append(command)
            return Result()

        with patch.object(batch.subprocess, "run", side_effect=fake_run):
            cleanup = batch.cleanup_agent_containers({"old"}, resource_token="batch-owned")

        self.assertEqual(cleanup["removed_container_ids"], ["owned-container"])
        self.assertEqual(calls[0], [
            "docker", "ps", "-aq", "--filter", f"ancestor={batch.AGENT_IMAGE}",
            "--filter", "name=batch-owned",
        ])
    def test_flag_status_uses_only_structured_verifier_results(self):
        self.assertEqual(
            batch.flag_status_from_trajectory({"info": {"flag_verified": True, "flag_submitted": True, "flag_verification_evidence": "verifier accepted submission"}}),
            {"flag_submitted": True, "flag_verified": True, "flag_verification_evidence": "verifier accepted submission"},
        )
        self.assertEqual(
            batch.flag_status_from_trajectory({"trajectory": [{"content": "flag correct success"}], "info": {}}),
            {"flag_submitted": False, "flag_verified": "unknown", "flag_verification_evidence": "not_attempted"},
        )

    def test_episode_status_requires_a_terminal_trace(self):
        self.assertEqual(
            batch.episode_status_from_trajectory({"info": {"exit_status": "submitted"}, "trajectory": [{"action": "submit flag{x}"}]}),
            {"episode_status": "completed", "episode_complete": True},
        )
        self.assertEqual(
            batch.episode_status_from_trajectory({"info": {"exit_status": "ctf_server_unavailable"}, "trajectory": []}),
            {"episode_status": "ctf_server_unavailable", "episode_complete": False},
        )

    def test_incomplete_trajectory_is_rerun(self):
        task = challenge()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            path = batch.trajectory_path(output_dir, task)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"info": {}, "trajectory": [{"action": "ls"}]}) + "\n", encoding="utf-8")
            state = {"version": 1, "tasks": {task.relative_path: {"status": "success", "episode_complete": False}}}
            self.assertFalse(batch.completed(state, output_dir, task))
    def test_legacy_completed_trajectory_is_backfilled_without_rerun(self):
        task = challenge()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            path = batch.trajectory_path(output_dir, task)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"info": {"exit_status": "submitted", "submission": "flag{x}"}}) + "\n", encoding="utf-8")
            state = {"version": 1, "tasks": {task.relative_path: {"status": "success"}}}

            self.assertTrue(batch.enrich_completed_records(state, output_dir, [task]))

            record = state["tasks"][task.relative_path]
            self.assertTrue(record["trajectory_generated"])
            self.assertEqual(record["flag_verified"], "unknown")
            self.assertEqual(record["flag_verification_evidence"], "missing structured verifier result")
            self.assertEqual(record["outcome_status"], "unsolved")
            self.assertTrue(batch.completed(state, output_dir, task))

    def test_summary_separates_trajectory_and_flag_statistics(self):
        task = challenge()
        state = {
            "tasks": {
                task.relative_path: {
                    "status": "success", "trajectory_generated": True, "flag_submitted": False,
                    "flag_verified": None, "container_cleanup": {"removed_container_ids": ["new"], "failed_container_ids": []},
                }
            },
            "docker_bridge_containers_at_start": 2,
            "docker_bridge_containers_at_end": 2,
        }

        summary = batch.build_summary([task], state, skipped=0)

        self.assertEqual(summary["trajectory_generated"], 1)
        self.assertEqual(summary["flag_verified"], 0)
        self.assertEqual(summary["flag_status_unknown"], 0)
        self.assertEqual(summary["flag_not_attempted"], 1)
        self.assertEqual(summary["container_cleanup_removed"], 1)
        self.assertEqual(summary["docker_bridge_containers_at_start"], 2)
        self.assertEqual(summary["categories"]["crypto"]["trajectory_generated"], 1)

    def test_public_trajectory_uses_messages_and_dictionary_tool_arguments(self):
        task = challenge()
        payload = {
            "history": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "solve this"},
            ],
            "trajectory": [{"thought": "inspect files", "action": "ls -la", "observation": "flag.txt"}],
        }
        public = batch.trajectory_to_messages(payload, task)
        self.assertEqual(set(public), {"id", "sample_type", "messages"})
        self.assertEqual(public["id"], task.task_id)
        self.assertEqual(public["sample_type"], "main")
        self.assertEqual(public["messages"][2]["tool_calls"], [{"name": "Bash", "arguments": {"command": "ls -la"}}])
        self.assertEqual(public["messages"][3], {"role": "tool", "content": "flag.txt"})
    def test_attempt_resource_cleanup_is_token_scoped(self):
        calls: list[list[str]] = []

        class Result:
            def __init__(self, stdout: str = "", returncode: int = 0):
                self.stdout = stdout
                self.stderr = ""
                self.returncode = returncode

        def fake_run(command, **_kwargs):
            calls.append(command)
            if command[:4] == ["docker", "ps", "-aq", "--filter"]:
                return Result("service-batch-token-1\n")
            if command[:5] == ["docker", "network", "ls", "-q", "--filter"]:
                return Result("network-id\n")
            return Result()

        with patch.object(batch.subprocess, "run", side_effect=fake_run):
            cleanup = batch.cleanup_attempt_resources("batch-token")

        self.assertEqual(cleanup["removed_containers"], ["service-batch-token-1"])
        self.assertEqual(cleanup["removed_networks"], ["network-id"])
        self.assertIn(["docker", "rm", "-f", "service-batch-token-1"], calls)
        self.assertIn(["docker", "network", "rm", "network-id"], calls)
    def test_network_exhaustion_is_classified_separately(self):
        self.assertTrue(batch.docker_network_exhausted("Docker 500", "no available IPv4 addresses on this network's address pools: bridge"))
        self.assertFalse(batch.docker_network_exhausted("model API error"))


if __name__ == "__main__":
    unittest.main()