"""Une conclusion prouvée grammaticalement n'accepte pas de condition de plus.

Le défaut visé n'est pas un manque de rappel mais une POLLUTION. Sur
`37-arbitrage-sportif`, la condition juste — prouvée par la grammaire — était
bien produite ; le modèle en ajoutait une seconde, non prouvée, et la règle
entière devenait fausse, emportant la priorité qui s'y appuyait. Une prémisse
parasite détruit deux éléments de référence.

Mesuré hors ligne : règles justes 73 -> 75, priorités 15 -> 17, règles
produites inchangées.

L'élagage doit rester CIBLÉ : il ne s'applique qu'aux conclusions ayant au
moins une prémisse prouvée, et ne doit jamais vider une conclusion ni toucher
aux règles dont toutes les conditions sont prouvées — celles que produit la
découpe des subordonnées coordonnées.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from gorgias import app


def _grounds(orientees, prouvees):
    """Rejoue `_verifier_paires` sans modèle : le juge répond A_TO_B, le filtre
    narratif ne rejette rien, et seule la table des relations prouvées varie."""

    def juge(model, instructions, payload, key, interpret, *a, **k):
        attendus = k.get("expected") or ()
        if key == "narrative":
            return {n: "N" for n in attendus}
        return {n: "A_TO_B" for n in attendus}

    segments = [(i * 10, i * 10 + 8) for i in range(8)]
    texte = "x" * 80
    with patch.object(app.syntaxe, "relations_prouvees", return_value=prouvees), \
         patch.object(app.singularite, "evenement_singulier", return_value=False), \
         patch.object(app, "_ask_stage", side_effect=juge):
        return app._verifier_paires(
            None, texte, segments, sorted(orientees), lambda *a: None
        )


class PureteDesConclusionsProuvees(unittest.TestCase):
    def test_une_premisse_non_prouvee_est_ecartee(self):
        """(1,3) est prouvée ; (2,3) ne l'est pas et vise la même conclusion."""
        grounds = _grounds(
            {(1, 3): "A_TO_B", (2, 3): "A_TO_B"}, {(1, 3): "A_TO_B"}
        )
        self.assertEqual(grounds.get(3), [1])

    def test_toutes_les_premisses_prouvees_sont_conservees(self):
        """Le cas des subordonnées coordonnées : chacune porte son `mark`."""
        grounds = _grounds(
            {(1, 3): "A_TO_B", (2, 3): "A_TO_B"},
            {(1, 3): "A_TO_B", (2, 3): "A_TO_B"},
        )
        self.assertEqual(sorted(grounds.get(3, [])), [1, 2])

    def test_une_conclusion_sans_preuve_est_intacte(self):
        """Sans ancrage grammatical, le modèle garde la main."""
        grounds = _grounds({(1, 3): "A_TO_B", (2, 3): "A_TO_B"}, {})
        self.assertEqual(sorted(grounds.get(3, [])), [1, 2])

    def test_l_elagage_ne_vide_jamais_une_conclusion(self):
        grounds = _grounds({(1, 3): "A_TO_B"}, {(1, 3): "A_TO_B"})
        self.assertEqual(grounds.get(3), [1])

    def test_les_autres_conclusions_ne_sont_pas_touchees(self):
        grounds = _grounds(
            {(1, 3): "A_TO_B", (2, 3): "A_TO_B", (4, 5): "A_TO_B"},
            {(1, 3): "A_TO_B"},
        )
        self.assertEqual(grounds.get(3), [1])
        self.assertEqual(grounds.get(5), [4])


if __name__ == "__main__":
    unittest.main()
