"""Le contexte d'une priorité est le segment qui la précède, verbe ou non.

« En zone de pente, la replantation prime sur le reboisement. » — « En zone de
pente » porte tout l'ancrage de la priorité, mais c'est un circonstant SANS
VERBE : l'étape d'étiquetage l'écarte, parce qu'il n'énonce rien de vrai ou
faux par lui-même.

La compilation se rabattait alors sur le contexte d'une règle voisine. La
priorité était construite, ancrée au mauvais endroit, et comptait donc DEUX
fois : une manquée et une fausse. C'est le seul poste du projet où une
correction gagne simultanément en rappel et en précision.

Mesuré : le segment précédant un classement isolé est un Context de référence
27 fois sur 32.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

from gorgias import app  # noqa: E402
from gorgias import priorites  # noqa: E402


class SegmentPrecedantUnClassement(unittest.TestCase):
    def test_le_circonstant_sans_verbe_est_bien_en_amont(self):
        """Le cas réel : `43-forets`, où l'ancrage était perdu."""
        texte = ("Quand une parcelle brûle sur plus de dix hectares, le "
                 "reboisement est programmé. Quand le sol reste instable, la "
                 "replantation est ajournée. En zone de pente, la replantation "
                 "prime sur le reboisement.")
        segments = app.segment_text(texte)
        classements, _ = priorites.isoler(texte, segments)
        self.assertTrue(classements, "le classement doit être repéré")
        adjacent = classements[0] - 1
        debut, fin = segments[adjacent - 1]
        self.assertEqual(texte[debut:fin], "En zone de pente")

    def test_ce_segment_n_a_pas_de_verbe(self):
        """C'est bien pour cela que l'étiquetage le laissait tomber."""
        from gorgias import syntaxe
        nlp = syntaxe._charger(syntaxe.MODELE_DEFAUT)
        if nlp is None:
            self.skipTest("spaCy indisponible")
        self.assertFalse(
            any(t.pos_ in ("VERB", "AUX") for t in nlp("En zone de pente")),
            "sans verbe : l'étape 1 ne peut pas le retenir comme proposition",
        )


class ConventionDeReference(unittest.TestCase):
    """Un contexte répété ne donne qu'UNE entité, et la priorité vise la
    première occurrence — sans quoi le second exemplaire compte comme faux
    positif. Vérifié sur `10-scenarios-freres`, où « En hiver » est énoncé
    deux fois et n'est annoté qu'une."""

    def test_un_contexte_repete_n_a_qu_une_entite_dans_la_reference(self):
        chemin = RACINE / "data" / "cas" / "10-scenarios-freres"
        texte = chemin.with_suffix(".txt").read_text(encoding="utf-8")
        annotations = chemin.with_suffix(".ann").read_text(encoding="utf-8")
        self.assertEqual(texte.count("En hiver"), 2, "le texte le répète")
        entites = [ligne for ligne in annotations.splitlines()
                   if ligne.startswith("T") and ligne.endswith("En hiver")]
        self.assertEqual(len(entites), 1, "la référence n'en annote qu'une")

class ReprisesEtExclusions(unittest.TestCase):
    """Deux cas où le segment adjacent NE doit PAS donner une entité neuve."""

    def _adjacent(self, nom):
        chemin = RACINE / "data" / "cas" / f"{nom}.txt"
        texte = chemin.read_text(encoding="utf-8")
        segments = app.segment_text(texte)
        classements, _ = priorites.isoler(texte, segments)
        return texte, segments, classements

    def test_un_frere_coordonne_n_ancre_pas_la_priorite(self):
        """« et que la pluie tombe » appartient à la conjonction de conditions
        d'une règle, pas au scénario de la préférence. Le prendre pour contexte
        produisait une entité fausse ET une priorité fausse."""
        texte, segments, classements = self._adjacent("11-combinaison")
        freres = app._freres_coordonnes(texte)
        vise = [c - 1 for c in classements
                if c - 1 >= 1 and (c - 2, c - 1) in freres]
        self.assertTrue(vise, "le cas doit être présent dans le banc")
        for adjacent in vise:
            debut, fin = segments[adjacent - 1]
            self.assertTrue(texte[debut:fin].startswith("et que"))

    def test_un_contexte_repete_reprend_sa_premiere_enonciation(self):
        """« En hiver » ouvre `09` SANS VERBE : l'étiquetage ne le retient pas,
        donc la reprise ne peut pas s'appuyer sur une entité déjà créée. Elle
        doit chercher parmi les SEGMENTS antérieurs, sans quoi un doublon
        apparaît à la seconde occurrence là où la référence annote la première.
        """
        texte, segments, classements = self._adjacent("09-raffinement-explicite")
        adjacent = classements[0] - 1
        debut, fin = segments[adjacent - 1]
        fragment = texte[debut:fin]
        self.assertEqual(fragment, "En hiver")
        premier = next(
            (i for i in range(1, adjacent)
             if texte[segments[i - 1][0]:segments[i - 1][1]] == fragment),
            None,
        )
        self.assertEqual(premier, 1, "la première énonciation est le segment 1")


if __name__ == "__main__":
    unittest.main()
