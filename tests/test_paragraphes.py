"""La frontière de paragraphe borne la portée des paires.

La fenêtre de portée (`PAIR_WINDOW = 6`) a été calibrée sur des documents d'un
seul paragraphe et de six à sept segments : tout y est à portée de tout. Dès
que le texte s'allonge, elle relie des paragraphes entiers. Mesuré sur un texte
de 47 segments et 11 paragraphes : 170 des 261 paires à portée (65 %) sautaient
une frontière, et les fausses règles observées en venaient toutes.

Ces tests fixent la propriété sous ses deux faces : la contrainte doit rester
INERTE sur un document d'un seul paragraphe — ce qui garantit qu'elle ne peut
rien dégrader des 79 documents historiques — et elle doit AGIR dès qu'un
document en compte plusieurs.

Jusqu'au 10 août la seconde face n'était pas vérifiable : aucun document du
banc n'avait plus d'un paragraphe. L'ajout de `48-drones-autonomes` (47
segments, 11 paragraphes) l'a rendue mesurable.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from gorgias import app

RACINE = Path(__file__).resolve().parent.parent


class FrontieresDeParagraphe(unittest.TestCase):
    def test_chaque_segment_recoit_son_paragraphe(self):
        texte = "Alpha bravo charlie delta.\n\nEcho foxtrot golf hotel."
        segments = app.segment_text(texte)
        index = app._index_paragraphes(texte, segments)
        self.assertEqual(len(index), len(segments) + 1)
        premiers = {index[n] for n, (d, _) in enumerate(segments, 1)
                    if d < texte.index("Echo")}
        seconds = {index[n] for n, (d, _) in enumerate(segments, 1)
                   if d >= texte.index("Echo")}
        self.assertEqual(premiers, {0})
        self.assertEqual(seconds, {1})

    def test_une_paire_traversant_un_blanc_est_ecartee(self):
        paragraphe = [0, 0, 0, 1, 1]        # segments 1-2 puis 3-4
        paires = [(1, 2), (2, 3), (3, 4), (1, 4)]
        retenues = app._meme_paragraphe(paires, paragraphe)
        self.assertEqual(retenues, [(1, 2), (3, 4)])

    def test_un_texte_d_un_seul_paragraphe_est_inchange(self):
        """La garantie qui rend le garde-fou sans risque sur le banc."""
        paragraphe = [0] * 8
        paires = [(1, 2), (2, 5), (3, 7)]
        self.assertEqual(app._meme_paragraphe(paires, paragraphe), paires)


def _paires_et_contrainte(chemin):
    texte = chemin.read_text(encoding="utf-8")
    segments = app.segment_text(texte)
    paragraphe = app._index_paragraphes(texte, segments)
    paires = app._paires_candidates(
        list(range(1, len(segments) + 1)), app.PAIR_WINDOW
    )
    return paragraphe, paires, app._meme_paragraphe(paires, paragraphe)


class EffetSurLeBanc(unittest.TestCase):
    def test_inerte_sur_les_documents_d_un_seul_paragraphe(self):
        """La garantie de non-régression sur les 79 documents historiques."""
        vus = 0
        for dossier in ("cas", "narratif"):
            for chemin in sorted((RACINE / "data" / dossier).glob("*.txt")):
                paragraphe, paires, retenues = _paires_et_contrainte(chemin)
                if max(paragraphe) > 0:
                    continue                      # traité par le test suivant
                vus += 1
                with self.subTest(document=chemin.stem):
                    self.assertEqual(
                        retenues, paires,
                        "la contrainte ne doit rien retirer d'un document "
                        "d'un seul paragraphe",
                    )
        self.assertEqual(vus, 79, "les 79 documents historiques sont mono-bloc")

    def test_agit_sur_un_document_multi_paragraphes(self):
        """La face que le banc ne pouvait pas vérifier avant le 10 août."""
        chemin = RACINE / "data" / "cas" / "48-drones-autonomes.txt"
        paragraphe, paires, retenues = _paires_et_contrainte(chemin)
        self.assertGreater(max(paragraphe), 0, "ce document est multi-paragraphes")
        self.assertLess(
            len(retenues), len(paires),
            "la contrainte doit retirer les paires qui sautent une frontière",
        )
        self.assertTrue(
            all(paragraphe[a] == paragraphe[b] for a, b in retenues),
            "aucune paire retenue ne doit traverser un paragraphe",
        )


if __name__ == "__main__":
    unittest.main()
