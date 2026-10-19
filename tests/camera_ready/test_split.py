"""The split manifest: sizes, disjointness, agreement with the stored test entries, tamper detection."""
import json
import tempfile
import unittest
from pathlib import Path

from _common import DATASET_DIR, SPLIT_MANIFEST, STORED_TEST
from piper_train.camera_ready.split import load_split_manifest


class SplitManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_split_manifest(SPLIT_MANIFEST, DATASET_DIR / "dataset.jsonl")
        cls.doc = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))

    def test_sizes_and_disjoint_ids(self):
        self.assertEqual({k: len(v) for k, v in self.m.ids.items()}, {"train": 12500, "validation": 100, "test": 500})
        all_ids = [i for v in self.m.ids.values() for i in v]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_raw_audio_hashes_disjoint(self):
        h = {k: {e["raw_wav_sha256"] for e in v} for k, v in self.doc["splits"].items()}
        self.assertFalse(h["train"] & h["validation"] or h["train"] & h["test"] or h["validation"] & h["test"])

    @unittest.skipUnless(STORED_TEST.is_file(), "stored test entries not available")
    def test_test_split_equals_stored_entries_in_order(self):
        stored = [json.loads(line)["audio_norm_path"] for line in STORED_TEST.read_text(encoding="utf-8").splitlines() if line]
        rows = [json.loads(line) for line in (DATASET_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual([rows[r]["audio_norm_path"] for r in self.m.rows["test"]], stored)

    def test_preview_ids_are_validation_not_test(self):
        self.assertTrue(set(self.m.preview_ids) <= set(self.m.ids["validation"]))
        self.assertFalse(set(self.m.preview_ids) & set(self.m.ids["test"]))

    def _tampered(self, mutate) -> Path:
        doc = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
        mutate(doc)
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(doc, f)
        f.close()
        return Path(f.name)

    def test_rejects_wrong_dataset_hash(self):
        p = self._tampered(lambda d: d.update(dataset_jsonl_sha256="0" * 64))
        with self.assertRaises(ValueError):
            load_split_manifest(p, DATASET_DIR / "dataset.jsonl")

    def test_rejects_row_id_mismatch(self):
        def swap(d):
            d["splits"]["train"][0]["row"], d["splits"]["train"][1]["row"] = d["splits"]["train"][1]["row"], d["splits"]["train"][0]["row"]
        with self.assertRaises(ValueError):
            load_split_manifest(self._tampered(swap), DATASET_DIR / "dataset.jsonl")

    def test_rejects_overlap(self):
        def overlap(d):
            d["splits"]["validation"].append(d["splits"]["test"][0])
        with self.assertRaises(ValueError):
            load_split_manifest(self._tampered(overlap), DATASET_DIR / "dataset.jsonl")


if __name__ == "__main__":
    unittest.main()
