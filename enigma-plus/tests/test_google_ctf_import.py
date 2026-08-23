from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("google_ctf_to_ctfdojo", SCRIPT_DIR / "google_ctf_to_ctfdojo.py")
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class GoogleCtfImportTests(unittest.TestCase):
    def test_import_is_verifier_aware_and_does_not_leak_solution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "google"
            source = root / "2024" / "quals" / "crypto" / "hash-me"
            source.mkdir(parents=True)
            (source / "README.md").write_text("""# Hash challenge
# flag: CTF{secret}
Solve it.
""", encoding="utf-8")
            (source / "Dockerfile").write_text("""FROM alpine:3.20
""", encoding="utf-8")
            secret = "CTF{secret}"
            (source / "flag.sha256").write_text(hashlib.sha256(secret.encode()).hexdigest(), encoding="utf-8")
            (source / "flag").write_text(secret, encoding="utf-8")
            (source / "solution").mkdir()
            (source / "solution" / "writeup.md").write_text("secret solution", encoding="utf-8")

            output = Path(temporary) / "dataset"
            row = module.convert_one(source, root, output, "2024", "crypto", "quals", False)
            task = Path(row["path"])
            metadata = json.loads((task / "challenge.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["event"], "quals")
            self.assertTrue(metadata["task_id"])
            self.assertTrue(metadata["description_sanitized"])
            self.assertNotIn(secret, metadata["description"])
            self.assertFalse(metadata["has_plaintext_solution"])
            self.assertEqual(metadata["verification_method"], "sha256")
            self.assertEqual(metadata["verification"]["files"], ["flag.sha256"])
            self.assertFalse((task / "flag").exists())
            self.assertFalse((task / "solution").exists())

            discovered, warnings = module_batch_discover(output)
            self.assertFalse(warnings)
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].task_id, metadata["task_id"])


def module_batch_discover(dataset: Path):
    batch_path = SCRIPT_DIR / "ctfdojo_batch_generate.py"
    batch_spec = importlib.util.spec_from_file_location("ctfdojo_batch_generate_for_import_test", batch_path)
    assert batch_spec is not None and batch_spec.loader is not None
    batch = importlib.util.module_from_spec(batch_spec)
    sys.modules[batch_spec.name] = batch
    batch_spec.loader.exec_module(batch)
    return batch.discover_challenges(dataset, None, True, require_verification=True)


if __name__ == "__main__":
    unittest.main()