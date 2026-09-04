from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gorgias import app


class QueueModel:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("réponse simulée manquante")
        return SimpleNamespace(content=json.dumps(self.payloads.pop(0)))


class ReasoningModel:
    """Petit oracle déterministe : teste l'orchestration, pas Qwen."""

    def invoke(self, messages):
        instructions = messages[0].content
        payload = messages[1].content

        if "ENUMERATED ALTERNATIVES" in instructions:
            document = {"enumerations": []}
        elif "TWO independent yes/no" in instructions:
            numeros = [int(n) for n in re.findall(r"(?m)^(\d+)\. ", payload)]
            labels = []
            for numero in numeros:
                labels.append(
                    {
                        "n": numero,
                        "p": "Y",
                        "m": "Y" if "prime sur" in self._segment(payload, numero) else "N",
                    }
                )
            document = {"labels": labels}
        elif "WHAT THE TEXT PUTS FORWARD" in instructions:
            examines = payload.rsplit("EXAMINE: ", 1)[1]
            numeros = [int(n) for n in examines.split(", ")]
            support = []
            for numero in numeros:
                grounds = {2: [1], 4: [3]}.get(numero, [])
                support.append({"c": numero, "g": grounds})
            document = {"support": support}
        elif "Determine the relation between two propositions" in instructions:
            numeros = [int(n) for n in re.findall(r"(?m)^(\d+)\. A:", payload)]
            document = {
                "relations": [{"n": numero, "r": "A_TO_B"} for numero in numeros]
            }
        elif "can the apparent connection" in instructions:
            numeros = [int(n) for n in re.findall(r"(?m)^(\d+)\. A:", payload)]
            document = {
                "narrative": [{"n": numero, "r": "NO"} for numero in numeros]
            }
        elif "PREFERENCES BETWEEN CONCLUSIONS" in instructions:
            if "prime sur" in payload:
                document = {
                    "preferences": [
                        {
                            "preferred": [2],
                            "contrasted": [4],
                            "delta": [1],
                            "reactivates": [],
                            "marker_n": 6,
                            "marker": "prime sur",
                        }
                    ]
                }
            else:
                document = {"preferences": []}
        else:
            raise AssertionError(f"étape simulée inconnue : {instructions[-80:]}")
        return SimpleNamespace(content=json.dumps(document))

    @staticmethod
    def _segment(payload: str, numero: int) -> str:
        match = re.search(rf"(?m)^{numero}\. (.*)$", payload)
        return match.group(1) if match else ""


class ComplementArticleModel:
    """Oracle du cas « préférés aux deux autres » de l'article."""

    def invoke(self, messages):
        instructions = messages[0].content
        payload = messages[1].content
        if "ENUMERATED ALTERNATIVES" in instructions:
            document = {
                "enumerations": [
                    {
                        "n": 4,
                        "items": ["choisir le poulet", "choisir le poisson"],
                    }
                ]
            }
        elif "TWO independent yes/no" in instructions:
            numeros = [int(n) for n in re.findall(r"(?m)^(\d+)\. ", payload)]
            document = {
                "labels": [
                    {
                        "n": numero,
                        "p": "Y",
                        "m": "Y" if numero == 8 else "N",
                    }
                    for numero in numeros
                ]
            }
        elif "WHAT THE TEXT PUTS FORWARD" in instructions:
            numeros = [int(n) for n in payload.rsplit("EXAMINE: ", 1)[1].split(", ")]
            document = {
                "support": [
                    {"c": numero, "g": [6] if numero in {2, 3, 4, 5} else []}
                    for numero in numeros
                ]
            }
        elif "Determine the relation between two propositions" in instructions:
            numeros = [int(n) for n in re.findall(r"(?m)^(\d+)\. A:", payload)]
            document = {
                "relations": [
                    {"n": numero, "r": "B_TO_A"} for numero in numeros
                ]
            }
        elif "can the apparent connection" in instructions:
            numeros = [int(n) for n in re.findall(r"(?m)^(\d+)\. A:", payload)]
            document = {
                "narrative": [{"n": numero, "r": "NO"} for numero in numeros]
            }
        elif "PREFERENCES BETWEEN CONCLUSIONS" in instructions:
            document = {
                "preferences": [
                    {
                        "preferred": [3, 4],
                        "contrasted": [],
                        "complement": True,
                        "delta": [6],
                        "reactivates": [],
                        "marker_n": 8,
                        "marker": "préférés aux deux autres",
                    }
                ]
            }
        else:
            raise AssertionError("étape simulée inconnue")
        return SimpleNamespace(content=json.dumps(document))


class PipelineTests(unittest.TestCase):
    def test_no_python_command_changes_the_qwen_model(self):
        root = Path(app.__file__).resolve().parent
        offenders = []
        for source in root.rglob("*.py"):
            if any(part in {".venv", "__pycache__", "tmp"} for part in source.parts):
                continue
            text = source.read_text(encoding="utf-8")
            for match in re.finditer(r"qwen[\w.-]*:[\w.-]+", text, re.IGNORECASE):
                if match.group().lower() != "qwen3:8b":
                    offenders.append(f"{source.relative_to(root)}: {match.group()}")
        self.assertEqual(offenders, [], "modèles Qwen non autorisés : " + ", ".join(offenders))

    def setUp(self):
        self.previous_votes = app.VOTES

    def tearDown(self):
        app.VOTES = self.previous_votes

    def test_default_model_is_unchanged(self):
        self.assertEqual(app.DEFAULT_MODEL, "qwen3:8b")

    def test_annotate_file_never_overwrites_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "document.txt"
            source.write_text("texte", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fichier source"):
                app.annotate_file(source, output_path=source)

    def test_no_safe_rule_is_a_successful_empty_result(self):
        model = QueueModel([
            {"enumerations": []},
            {"labels": [{"n": 1, "p": "Y", "m": "N"}]},
            {"support": [{"c": 1, "g": []}]},
        ])
        self.assertEqual(app.annotate("Une affirmation isolée.", model=model), "")

    def test_inline_condition_is_split_without_punctuation(self):
        text = "Un fonds n'est retenu que si son rendement est élevé."
        spans = app.segment_text(text)
        self.assertEqual(
            [text[start:end] for start, end in spans],
            ["Un fonds n'est retenu", "son rendement est élevé"],
        )

    def test_elided_inline_condition_is_split_without_splitting_verbs(self):
        text = "Le fonds n'est retenu que s'il présente un risque faible."
        spans = app.segment_text(text)
        self.assertEqual(
            [text[start:end] for start, end in spans],
            ["Le fonds n'est retenu", "présente un risque faible"],
        )
        ordinary = "La remise s'applique immédiatement."
        self.assertEqual(
            [ordinary[start:end] for start, end in app.segment_text(ordinary)],
            [ordinary[:-1]],
        )

    def test_causative_lexicon_is_not_fitted_to_the_benchmark(self):
        text = "Le transfert du dossier ouvre un délai de recours."
        self.assertEqual(
            [text[start:end] for start, end in app.segment_text(text)],
            [text[:-1]],
            "un verbe choisi dans le banc ne doit pas créer une règle de domaine",
        )

    def test_pronominal_declaration_is_not_split(self):
        text = "Le port du masque s'impose."
        self.assertEqual(
            [text[start:end] for start, end in app.segment_text(text)],
            [text[:-1]],
        )

    def test_opening_time_is_not_split(self):
        text = "Le guichet ouvre à dix heures."
        self.assertEqual(
            [text[start:end] for start, end in app.segment_text(text)],
            [text[:-1]],
        )

    def test_enumeration_stage_has_a_zero_cost_negative_gate(self):
        self.assertIsNone(
            app._ENUMERATION_SURFACE_CUE.search("Quand il pleut, nous restons.")
        )
        self.assertIsNotNone(
            app._ENUMERATION_SURFACE_CUE.search("choisir le train ou la voiture")
        )

    def test_long_document_label_batches_fit_the_output_budget(self):
        self.assertLess(app.LONG_DOCUMENT_LABEL_BATCH, app.LABEL_BATCH)
        self.assertGreater(app.LONG_DOCUMENT_THRESHOLD, 50)
        self.assertEqual(app.LONG_DOCUMENT_LABEL_BATCH, 8)

    def test_if_then_is_split_without_comma(self):
        text = "Si le disque est plein alors la sauvegarde échoue."
        spans = app.segment_text(text)
        self.assertEqual(
            [text[start:end] for start, end in spans],
            ["Si le disque est plein", "la sauvegarde échoue"],
        )

    def test_proposition_and_marker_are_independent(self):
        labels = app._read_labels(
            [{"n": 1, "p": "Y", "m": "Y"}, {"n": 2, "p": "N", "m": "N"}]
        )
        self.assertEqual(labels[1], frozenset({"P", "M"}))
        self.assertEqual(labels[2], frozenset({"N"}))

    def test_complement_anaphora_is_kept_for_deterministic_resolution(self):
        rankings = app._read_rankings(
            [
                {
                    "preferred": [2, 3],
                    "contrasted": ["les deux autres"],
                    "delta": [5],
                    "reactivates": [],
                    "marker_n": 6,
                    "marker": "préfère aux deux autres",
                }
            ]
        )
        self.assertTrue(rankings[0]["complement"])
        self.assertEqual(rankings[0]["contrasted"], [])

    def test_enumeration_keeps_its_active_option_scope(self):
        text = "acheter de l'agneau, du porc, du poulet ou du poisson"
        segments, scopes = app._split_enumerations(
            text,
            [(0, len(text))],
            {
                1: [
                    "acheter de l'agneau",
                    "du porc",
                    "du poulet",
                    "du poisson",
                ]
            },
            return_scopes=True,
        )
        self.assertEqual(
            [text[start:end] for start, end in segments],
            ["acheter de l'agneau", "du porc", "du poulet", "du poisson"],
        )
        self.assertEqual(scopes, [frozenset({1, 2, 3, 4})])

    def test_complement_uses_local_enumeration_not_all_document_options(self):
        text = "agneau porc poulet poisson préfère aux deux autres"
        segments = [(0, 6), (7, 11), (12, 18), (19, 26), (27, len(text))]
        contrasted, status = app._resolve_complement_scope(
            preferred=[2, 3],
            marker_number=5,
            option_scopes=[frozenset({1, 2, 3, 4})],
            previous_preferences=[],
            option_numbers=[1, 2, 3, 4, 99],
            texte=text,
            segments=segments,
        )
        self.assertEqual(status, "RESOLVED")
        self.assertEqual(contrasted, [1, 4])

    def test_complement_prefers_parent_scenario_set(self):
        text = "agneau porc poulet poisson groupe parent autre"
        segments = [(0, 6), (7, 11), (12, 18), (19, 26), (27, 33), (34, 40)]
        contrasted, status = app._resolve_complement_scope(
            preferred=[2],
            marker_number=6,
            option_scopes=[frozenset({1, 2, 3, 4})],
            previous_preferences=[{"preferred": [2, 3], "marker_n": 5}],
            option_numbers=[1, 2, 3, 4],
            texte=text,
            segments=segments,
        )
        self.assertEqual(status, "RESOLVED")
        self.assertEqual(contrasted, [3])

    def test_complement_without_local_universe_is_ambiguous(self):
        text = "porc préféré aux autres"
        segments = [(0, 4), (5, len(text))]
        contrasted, status = app._resolve_complement_scope(
            preferred=[1],
            marker_number=2,
            option_scopes=[],
            previous_preferences=[],
            option_numbers=[1, 9],
            texte=text,
            segments=segments,
        )
        self.assertIsNone(contrasted)
        self.assertEqual(status, "AMBIGUOUS")

        assembler = app.Assembler(text)
        assembler.diagnostic(
            "AMBIGUOUS complement: univers parent local introuvable"
        )
        self.assertIn("% AMBIGUOUS complement:", assembler.render_lpp())

    def test_incomplete_ballot_is_retried_then_votes_atomically(self):
        model = QueueModel(
            [
                {"labels": [{"n": 1, "p": "Y", "m": "N"}]},
                {
                    "labels": [
                        {"n": 1, "p": "Y", "m": "N"},
                        {"n": 2, "p": "Y", "m": "N"},
                    ]
                },
                {
                    "labels": [
                        {"n": 1, "p": "Y", "m": "N"},
                        {"n": 2, "p": "N", "m": "N"},
                    ]
                },
                {
                    "labels": [
                        {"n": 1, "p": "Y", "m": "N"},
                        {"n": 2, "p": "Y", "m": "N"},
                    ]
                },
            ]
        )
        app.VOTES = 3
        result = app._ask_stage(
            model,
            "test",
            "1. A\n2. B",
            "labels",
            app._read_labels,
            expected=[1, 2],
        )
        self.assertEqual(result[1], frozenset({"P"}))
        self.assertEqual(result[2], frozenset({"P"}))
        self.assertEqual(model.calls, 4)

    def test_hybrid_pipeline_builds_rules_and_anchored_priority(self):
        text = (
            "Quand il pleut, nous prenons le train. "
            "Quand la route est sèche, nous prenons la voiture. "
            "En hiver, le train prime sur la voiture."
        )
        annotations = app.annotate(
            text,
            etage="hybride",
            votes=1,
            model=ReasoningModel(),
        )
        app.validate_annotation_graph(annotations)
        self.assertEqual(annotations.count("\trule:"), 2)
        self.assertEqual(annotations.count("\tprefer:"), 1)
        self.assertIn("\tMarker ", annotations)
        marker_line = next(
            line for line in annotations.splitlines() if "\tMarker " in line
        )
        self.assertTrue(marker_line.endswith("\tprime sur"))

        lpp = app.annotate(
            text,
            etage="hybride",
            votes=1,
            model=ReasoningModel(),
            output_format="lpp",
        )
        self.assertIn("rule(r1,", lpp)
        self.assertIn("prefer(p1,", lpp)
        self.assertIn("complement(o1, o2).", lpp)

    def test_output_path_matches_requested_format(self):
        from pathlib import Path

        self.assertEqual(app.output_path_for(Path("case.txt")), Path("case.ann"))
        self.assertEqual(
            app.output_path_for(Path("case.txt"), "lpp"), Path("case.lpp")
        )

    def test_article_complement_is_compiled_from_parent_option_set(self):
        text = (
            "Dans ce menu, choisir l'agneau, choisir le porc, choisir le poulet "
            "ou choisir le poisson. Les quatre choix sont disponibles. "
            "Le porc et le poulet sont moins chers, ils sont préférés aux deux autres."
        )
        lpp = app.annotate(
            text,
            etage="hybride",
            votes=1,
            model=ComplementArticleModel(),
            output_format="lpp",
        )
        complements = [
            line for line in lpp.splitlines() if line.startswith("complement(")
        ]
        preferences = [
            line for line in lpp.splitlines() if line.startswith("prefer(")
        ]
        self.assertEqual(
            set(complements),
            {
                "complement(o1, o2).",
                "complement(o1, o3).",
                "complement(o2, o4).",
                "complement(o3, o4).",
            },
        )
        self.assertEqual(len(preferences), 4)


if __name__ == "__main__":
    unittest.main()
