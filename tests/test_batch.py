from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gorgias import batch


class BatchTests(unittest.TestCase):
    def test_outputs_and_audit_manifest_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "a.txt").write_text("Texte A", encoding="utf-8")
            (source / "b.txt").write_text("Texte B", encoding="utf-8")

            manifest = batch.process_directory(
                source, output, output_format="lpp",
                annotator=lambda text: f"source({text[-1].lower()}).",
            )

            self.assertEqual(manifest["summary"]["succeeded"], 2)
            self.assertEqual(manifest["summary"]["failed"], 0)
            self.assertEqual((output / "a.lpp").read_text(encoding="utf-8"),
                             "source(a).\n")
            on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["schema_version"], 1)
            self.assertEqual(len(on_disk["documents"][0]["source_sha256"]), 64)

    def test_one_failure_does_not_destroy_the_rest_of_the_batch(self):
        def annotate(text: str) -> str:
            if text == "cassé":
                raise RuntimeError("indisponible")
            return "ok"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            (source / "a.txt").write_text("cassé", encoding="utf-8")
            (source / "b.txt").write_text("valide", encoding="utf-8")
            manifest = batch.process_directory(source, output, annotator=annotate)
            self.assertEqual(manifest["summary"]["failed"], 1)
            self.assertEqual(manifest["summary"]["succeeded"], 1)
            self.assertTrue((output / "b.lpp").exists())

    def test_source_and_output_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distincts"):
                batch.process_directory(root, root, annotator=lambda _: "")


if __name__ == "__main__":
    unittest.main()
