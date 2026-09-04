"""« Comme A et que B, alors C » énonce DEUX conditions d'une seule règle.

`segment_text` sépare déjà A et B — c'est la découpe des coordonnées — mais
jetait aussitôt leur fraternité. Deux dégâts mesurés, tous deux corrigés ici :

  - la seconde condition n'était rattachée à rien et la règle sortait amputée
    (`25-vol-retarde`, `26-chaine-froid`) ;
  - la syntaxe prouvait parfois « A => B » entre les deux sœurs, fabriquant
    une règle là où il n'y a qu'une conjonction de conditions.

La propagation ne part que d'une condition PROUVÉE. Mesuré sur le banc :
5 cas où la référence réclame la sœur, ZÉRO où elle la refuse.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from gorgias import app


class DetectionDesFreres(unittest.TestCase):
    def test_deux_subordonnees_coordonnees_sont_soeurs(self):
        texte = ("Comme le retard dépasse trois heures et que la cause relève "
                 "du transporteur, une indemnité forfaitaire est due.")
        segments = app.segment_text(texte)
        freres = app._freres_coordonnes(texte)
        self.assertEqual(freres, {(1, 2)})
        self.assertIn("et que", texte[segments[1][0]:segments[1][1]])

    def test_deux_phrases_voisines_ne_sont_pas_soeurs(self):
        """La fraternité vient d'une coupure de coordination, pas du voisinage."""
        texte = ("Quand le sol est gelé, le sel est appliqué. "
                 "Quand la pluie tombe, le sable est répandu.")
        self.assertEqual(app._freres_coordonnes(texte), set())


class UsageDansLaVerification(unittest.TestCase):
    def _grounds(self, paires, prouvees, freres):
        def juge(model, instructions, payload, key, interpret, *a, **k):
            attendus = k.get("expected") or ()
            return {n: ("N" if key == "narrative" else "A_TO_B") for n in attendus}

        segments = [(i * 10, i * 10 + 8) for i in range(8)]
        with patch.object(app.syntaxe, "relations_prouvees", return_value=prouvees), \
             patch.object(app.singularite, "evenement_singulier", return_value=False), \
             patch.object(app, "_freres_coordonnes", return_value=freres), \
             patch.object(app, "_ask_stage", side_effect=juge):
            return app._verifier_paires(
                None, "x" * 80, segments, sorted(paires), lambda *a: None
            )

    def test_une_paire_de_soeurs_n_est_jamais_une_relation(self):
        grounds = self._grounds({(1, 2)}, {}, {(1, 2)})
        self.assertEqual(grounds, {}, "les sœurs ne s'infèrent pas l'une l'autre")

    def test_la_condition_prouvee_se_propage_a_sa_soeur(self):
        grounds = self._grounds({(1, 3)}, {(1, 3): "A_TO_B"}, {(1, 2)})
        self.assertEqual(sorted(grounds.get(3, [])), [1, 2])

    def test_aucune_propagation_sans_preuve(self):
        """Une suggestion du modèle ne suffit pas à entraîner la sœur."""
        grounds = self._grounds({(1, 3)}, {}, {(1, 2)})
        self.assertEqual(grounds.get(3), [1])


if __name__ == "__main__":
    unittest.main()
