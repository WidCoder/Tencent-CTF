from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "batch_generate_trajectories.py"
SPEC = importlib.util.spec_from_file_location("batch_generate_trajectories", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


class BatchTrajectoryTests(unittest.TestCase):
    def create_challenge(self, root: Path, relative_dir: str, *, name: str, category: str) -> Path:
        task_dir = root / relative_dir
        task_dir.mkdir(parents=True)
        challenge_path = task_dir / "challenge.json"
        challenge_path.write_text(json.dumps({"name": name, "category": category, "files": []}), encoding="utf-8")
        return challenge_path

    def test_discover_challenges_filters_by_category_and_disambiguates_ids(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.create_challenge(root, "HTB/crypto/First", name="Same Name", category="crypto")
            self.create_challenge(root, "Other/crypto/Second", name="Same Name", category="crypto")
            self.create_challenge(root, "HTB/rev/Third", name="Reverse", category="rev")

            crypto_tasks = batch.discover_challenges(root, "crypto")

            self.assertEqual(len(crypto_tasks), 2)
            self.assertEqual({task.category for task in crypto_tasks}, {"crypto"})
            self.assertEqual(len({task.task_id for task in crypto_tasks}), 2)

    def test_completed_task_requires_success_status_and_normalized_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            task = batch.Challenge(
                challenge_json="/example/challenge.json",
                source_dir="/example",
                relative_path="HTB/crypto/Example",
                platform="HTB",
                category="crypto",
                name="Example",
                task_id="HTB_crypto_Example",
            )
            status = {"version": 1, "tasks": {task.relative_path: {"status": "success"}}}
            self.assertFalse(batch.is_completed(status, output_dir, task))

            trajectory_path = batch.trajectory_path_for(output_dir, task.task_id)
            trajectory_path.parent.mkdir(parents=True)
            trajectory_path.write_text("{}\n", encoding="utf-8")
            self.assertTrue(batch.is_completed(status, output_dir, task))

    def test_summary_includes_step_token_and_category_statistics(self):
        task = batch.Challenge(
            challenge_json="/example/challenge.json",
            source_dir="/example",
            relative_path="HTB/crypto/Example",
            platform="HTB",
            category="crypto",
            name="Example",
            task_id="HTB_crypto_Example",
        )
        status = {
            "version": 1,
            "tasks": {
                task.relative_path: {
                    "status": "success",
                    "steps": 4,
                    "input_tokens": 100,
                    "output_tokens": 20,
                }
            },
        }

        summary = batch.build_summary([task], status)

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["average_steps"], 4.0)
        self.assertEqual(summary["average_token_consumption"], 120.0)
        self.assertEqual(summary["categories"]["crypto"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
