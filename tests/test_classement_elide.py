"""Le comparatif à premier terme élidé est un classement, pas une proposition.

« il les préfère aux deux autres » reprend le terme gagnant par un pronom. Ni
l'amorce contiguë ni le comparatif discontinu ne le reconnaissent : tous deux
exigent un terme explicite entre le verbe et sa préposition.

La conséquence n'est pas une priorité manquée mais une POLLUTION : le segment
reste dans le flux, se fait étiqueter comme proposition, et sert de prémisse
aux options énumérées. Mesuré sur `01-courses-3-niveaux` — trois règles de
référence en deviennent fausses.
"""
from __future__ import annotations

import unittest

from gorgias import priorites


class ComparatifElide(unittest.TestCase):
    def test_le_classement_elide_est_reconnu(self):
        for fragment in ("il les préfère aux deux autres",
                         "elle les privilégie aux autres options"):
            with self.subTest(fragment=fragment):
                self.assertIsNotNone(priorites.marqueur_dans(fragment))

    def test_aucune_priorite_n_est_fabriquee_sans_gagnante(self):
        """Le gagnant est anaphorique : on retire la pollution, on n'invente pas.

        `cotes_du_classement` doit rendre None plutôt qu'un couple bancal —
        une préférence dont un côté est vide ne désigne rien.
        """
        self.assertIsNone(
            priorites.cotes_du_classement("il les préfère aux deux autres")
        )

    def test_les_classements_explicites_restent_reconnus(self):
        for fragment in ("la remise l'emporte sur le paiement d'avance",
                         "il préfère le porc au poulet",
                         "le sable est préféré au sel"):
            with self.subTest(fragment=fragment):
                self.assertIsNotNone(priorites.marqueur_dans(fragment))

    def test_une_conclusion_de_regle_n_est_pas_un_classement(self):
        """Garde-fou contre l'élargissement au verbe SEUL, mesuré et refusé.

        « Les communications directes entre drones sont privilégiées » est la
        conclusion d'une règle de `48-drones-autonomes`, pas un classement. Un
        motif qui reconnaîtrait le verbe sans sa préposition la soustrairait du
        flux et détruirait l'option, sa règle et la priorité qui s'y appuie.
        """
        for fragment in (
            "Les communications directes entre drones sont privilégiées",
            "les drones évitent généralement d’utiliser la station",
        ):
            with self.subTest(fragment=fragment):
                self.assertIsNone(priorites.marqueur_dans(fragment))


if __name__ == "__main__":
    unittest.main()
