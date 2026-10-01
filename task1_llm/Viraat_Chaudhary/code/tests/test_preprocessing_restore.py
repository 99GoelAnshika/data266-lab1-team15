"""Cloned runs rebuild omitted arrays only when frozen evidence agrees."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import prepare, verify_processed


class Stories(list):
    _fingerprint = "fixture-source-v1"


class PreprocessingRestoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {
            "experiment": {"member_name": "Restoration fixture"},
            "data": {
                "dataset_id": "synthetic-fixture",
                "dataset_revision": "fixed-fixture",
                "source_split": "train",
                "text_field": "text",
                "train_sequences": 16,
                "validation_sequences": 4,
                "sequence_length": 8,
                "split_seed": 15,
                "separator": "\n\n",
            },
        }
        self.source = Stories(
            [{"text": "Once upon a time Lily found a little dog. " * 6} for _ in range(4)]
        )
        self.manifest = prepare(self.root, self.config, source=self.source)
        self.paths = [
            self.root / "data_processed/train.npy",
            self.root / "data_processed/validation.npy",
            self.root / "outputs/metrics/vocabulary.json",
            self.root / "outputs/metrics/split_manifest.json",
        ]
        self.original = [p.read_bytes() for p in self.paths]

    def remove_arrays(self):
        self.paths[0].unlink()
        self.paths[1].unlink()

    def assert_frozen_metadata(self):
        self.assertEqual(self.paths[2].read_bytes(), self.original[2])
        self.assertEqual(self.paths[3].read_bytes(), self.original[3])

    def test_restore_both_arrays_matches_frozen_bytes(self):
        self.remove_arrays()
        restored = prepare(self.root, self.config, source=self.source)
        self.assertEqual(restored, self.manifest)
        self.assertEqual([p.read_bytes() for p in self.paths], self.original)
        verify_processed(self.root, self.config)

    def test_restore_only_missing_array_preserves_existing_file(self):
        original_mtime = self.paths[0].stat().st_mtime_ns
        self.paths[1].unlink()
        prepare(self.root, self.config, source=self.source)
        self.assertEqual(self.paths[0].stat().st_mtime_ns, original_mtime)
        self.assertEqual(self.paths[1].read_bytes(), self.original[1])
        self.assert_frozen_metadata()

    def test_changed_source_publishes_no_arrays(self):
        self.remove_arrays()
        altered = copy.deepcopy(self.source)
        altered[0]["text"] = "Changed text must be rejected. " * 20
        with self.assertRaisesRegex(ValueError, "differs from the frozen manifest"):
            prepare(self.root, self.config, source=altered)
        self.assertFalse(self.paths[0].exists())
        self.assertFalse(self.paths[1].exists())
        self.assert_frozen_metadata()

    def test_changed_configuration_is_rejected(self):
        self.remove_arrays()
        altered = copy.deepcopy(self.config)
        altered["data"]["split_seed"] += 1
        with self.assertRaisesRegex(ValueError, "data configuration"):
            prepare(self.root, altered, source=self.source)
        self.assert_frozen_metadata()
        self.assertFalse(self.paths[0].exists())

    def test_corrupt_vocabulary_is_not_overwritten(self):
        self.remove_arrays()
        self.paths[2].write_bytes(b"corrupted vocabulary")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare(self.root, self.config, source=self.source)
        self.assertEqual(self.paths[2].read_bytes(), b"corrupted vocabulary")
        self.assertEqual(self.paths[3].read_bytes(), self.original[3])
        self.assertFalse(self.paths[0].exists())

    def test_wrong_expected_array_hash_publishes_no_arrays(self):
        self.remove_arrays()
        altered = copy.deepcopy(self.manifest)
        altered["files"]["data_processed/train.npy"] = "0" * 64
        self.paths[3].write_text(json.dumps(altered), encoding="utf-8")
        changed_bytes = self.paths[3].read_bytes()
        with self.assertRaisesRegex(ValueError, "differs from the frozen manifest"):
            prepare(self.root, self.config, source=self.source)
        self.assertEqual(self.paths[3].read_bytes(), changed_bytes)
        self.assertFalse(self.paths[0].exists())
        self.assertFalse(self.paths[1].exists())

    def test_corrupt_existing_array_is_not_replaced(self):
        self.paths[0].write_bytes(b"corrupted array")
        self.paths[1].unlink()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare(self.root, self.config, source=self.source)
        self.assertEqual(self.paths[0].read_bytes(), b"corrupted array")
        self.assertFalse(self.paths[1].exists())
        self.assert_frozen_metadata()

    def test_only_library_fingerprint_can_vary(self):
        self.remove_arrays()
        changed = copy.deepcopy(self.source)
        changed._fingerprint = "fixture-source-v2"
        restored = prepare(self.root, self.config, source=changed)
        self.assertEqual(restored["source_fingerprint"], "fixture-source-v1")
        self.assertEqual([p.read_bytes() for p in self.paths], self.original)

    def test_missing_frozen_metadata_is_not_guessed(self):
        self.paths[2].unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete preprocessing"):
            prepare(self.root, self.config, source=self.source)
        self.assertEqual(self.paths[0].read_bytes(), self.original[0])
        self.assertEqual(self.paths[1].read_bytes(), self.original[1])
        self.assertFalse(self.paths[2].exists())


if __name__ == "__main__":
    unittest.main()
