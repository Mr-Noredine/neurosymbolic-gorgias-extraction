"""Une prémisse et sa conclusion appartiennent au même locuteur.

Dans un dialogue argumentatif, la conclusion d'un interlocuteur n'est pas la
prémisse d'un autre : c'est une position CONCURRENTE, et c'est la priorité qui
en rend compte, pas une règle. Lier les deux fabrique une inférence que
personne n'a faite.

Mesuré sur les 5 dialogues du banc : 11 relations de référence, dont ZÉRO
franchit un changement de locuteur ; 8 relations produites, dont 4 le
franchissent et les QUATRE sont fausses.

La frontière est la même que celle du paragraphe, et se compose avec elle : un
auteur qui va à la ligne clôt son raisonnement, un locuteur qui cède la parole
aussi.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from gorgias import app

RACINE = Path(__file__).resolve().parent.parent
DIALOGUE = (
    "ANNA. Le prototype tient la charge, nous pouvons livrer vendredi.\n"
    "BRUNO. Deux tests d'intégration restent rouges, il faut repousser.\n"
    "ANNA. Ces tests portent sur un module que le client n'utilise pas.\n"
)


class FrontiereDeLocuteur(unittest.TestCase):
    def test_les_tours_d_un_meme_locuteur_partagent_leur_numero(self):
        """ANNA parle deux fois : ses deux tours doivent se rejoindre.

        Une règle de référence de `05-dialogue` réunit deux prémisses énoncées
        dans deux tours distincts d'ANNA — un garde-fou par TOUR la casserait.
        """
        segments = app.segment_text(DIALOGUE)
        index = app._index_locuteurs(DIALOGUE, segments)
        premier = index[1]
        dernier = index[len(segments)]
        self.assertEqual(premier, dernier, "les deux tours d'ANNA diffèrent")
        milieu = {index[n] for n, (d, _) in enumerate(segments, 1)
                  if DIALOGUE.index("BRUNO") <= d < DIALOGUE.index("Ces tests")}
        self.assertNotIn(premier, milieu, "BRUNO doit être distinct d'ANNA")

    def test_un_texte_sans_dialogue_est_inerte(self):
        texte = "Quand le sol est gelé, le sel est appliqué. La route est sûre."
        segments = app.segment_text(texte)
        index = app._index_locuteurs(texte, segments)
        self.assertEqual(set(index), {0}, "aucun locuteur ne doit être inventé")

    def test_le_filtre_ecarte_une_paire_entre_deux_locuteurs(self):
        index = [0, 0, 0, 1, 1]
        paires = [(1, 2), (2, 3), (3, 4), (1, 4)]
        self.assertEqual(app._meme_bloc(paires, index), [(1, 2), (3, 4)])


if __name__ == "__main__":
    unittest.main()
