from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from sweagent.agent.context_compressor import ContextCompressionManager, count_tokens
from sweagent.agent.trajectory_recorder import TrajectoryRecorder
from sweagent.parallel_runner import generate_trajectories


def test_compression_trigger_and_event():
    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": "x" * 1000},
        {"role": "assistant", "content": "y" * 1000},
    ] * 8
    manager = ContextCompressionManager(
        enabled=True, max_context_tokens=count_tokens(messages) // 2, trigger_ratio=0.5,
        summary_model=lambda _: '{"task":"demo","next_steps":["continue"]}',
    )
    result = manager.maybe_compress(messages)
    assert result.compressed
    assert manager.events[0]["type"] == "compression"
    assert result.new_token_count < result.old_token_count


def test_parallel_multi_trajectory_isolation(tmp_path):
    seen = []

    def run_one(task, output, index):
        seen.append(output)
        return output

    results = generate_trajectories(
        [{"instance_id": "task001"}, {"instance_id": "task002"}],
        run_one,
        output_dir=tmp_path,
        workers=4,
        trajectories_per_task=3,
    )
    assert len(results) == 6
    assert len({str(path) for path in seen}) == 6


def test_thought_recording(tmp_path):
    recorder = TrajectoryRecorder()
    path = tmp_path / "trajectory.json"
    recorder.save(path, {"trajectory": [{"action": "ls", "observation": "ok", "thought": "inspect"}]})
    assert '"thought": "inspect"' in path.read_text()

