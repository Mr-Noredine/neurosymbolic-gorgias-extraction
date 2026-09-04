"""Deux formes ne peuvent pas être la CONCLUSION d'une règle.

  - une subordonnée marquée : « Puisque l'exposition est critique » énonce une
    prémisse. C'est le miroir du court-circuit — là où lui dit « la subordonnée
    EST la condition », celui-ci dit « la subordonnée n'est PAS la conclusion » ;
  - un adjoint concessif : « même en hiver », « même à budget contraint » —
    « même » suivi d'une préposition introduit une circonstance concédée, sans
    verbe ni prédication.

Le test ne porte QUE sur la conclusion : les deux formes sont au contraire des
conditions parfaitement légitimes, et les confondre détruirait la moitié du
banc.

Mesuré : 7 règles fausses retirées, ZÉRO règle de référence détruite, précision
des règles 80 % -> 87 %.
"""
from __future__ import annotations

import unittest

from gorgias import syntaxe


def conclut(fragment: str) -> bool:
    return syntaxe.peut_conclure(fragment, 0, len(fragment))


class ConclusionsRefusees(unittest.TestCase):
    def test_une_subordonnee_marquee_ne_conclut_pas(self):
        for fragment in ("Puisque l'exposition est critique",
                         "Comme le constat signale une fragilité",
                         "Quand le sol est gelé"):
            with self.subTest(fragment=fragment):
                self.assertFalse(conclut(fragment))

    def test_un_adjoint_concessif_ne_conclut_pas(self):
        for fragment in ("même en hiver", "même à budget contraint",
                         "même en sol argileux"):
            with self.subTest(fragment=fragment):
                self.assertFalse(conclut(fragment))


class ConclusionsLegitimes(unittest.TestCase):
    def test_une_proposition_ordinaire_conclut(self):
        for fragment in ("le sel est appliqué",
                         "il le garde",
                         "une indemnité forfaitaire est due",
                         "l'avertissement est donné"):
            with self.subTest(fragment=fragment):
                self.assertTrue(conclut(fragment))

    def test_un_groupe_nominal_conclut(self):
        """Le banc en compte : « du porc », « du poulet » sont des Options."""
        for fragment in ("du porc", "du poulet ou du poisson"):
            with self.subTest(fragment=fragment):
                self.assertTrue(conclut(fragment))

    def test_meme_sans_preposition_reste_admis(self):
        """« Même les experts se trompent » prédique : on ne le refuse pas.

        La restriction à « même + préposition » est délibérée ; « même » seul
        est un adverbe de portée qui n'empêche rien.
        """
        self.assertTrue(conclut("Même les experts se trompent"))

    def test_une_conjonction_de_coordination_ne_conclut_pas(self):
        """« car X » donne la RAISON de ce qui précède : c'est une prémisse."""
        for fragment in ("car elles permettent une réaction plus rapide",
                         "Mais si le poulet est produit localement"):
            with self.subTest(fragment=fragment):
                self.assertFalse(conclut(fragment))

    def test_un_pronom_relatif_ne_conclut_pas(self):
        """« qui peut ensuite… » modifie, il n'affirme rien de lui-même."""
        self.assertFalse(conclut("qui peut ensuite les retransmettre"))

    def test_un_pronom_personnel_conclut(self):
        """Le trait PronType=Rel distingue « qui » de « il ».

        Confondre les deux détruirait des conclusions de référence comme
        « il le garde ».
        """
        self.assertTrue(conclut("il le garde"))

    def test_sans_spacy_le_repli_textuel_reste_actif(self):
        """Les connecteurs fermés ne dépendent pas du modèle spaCy."""
        import unittest.mock as mock
        with mock.patch.object(syntaxe, "_charger", return_value=None):
            self.assertFalse(conclut("Puisque l'exposition est critique"))
            self.assertTrue(conclut("La mise en production est bloquée"))



class ConditionsRefusees(unittest.TestCase):
    """Une subordonnée d'exception ou de concession DÉFAIT la règle.

    « sauf lorsqu'il constitue le seul moyen » ne conditionne pas, elle excepte ;
    « même lorsque la consommation est importante » dit que la règle tient
    MALGRÉ cela. Le formalisme l'exprime par une priorité — en faire une
    prémisse inverse le sens de la règle.
    """

    def conditionne(self, fragment: str) -> bool:
        return syntaxe.peut_conditionner(fragment, 0, len(fragment))

    def test_une_exception_ne_conditionne_pas(self):
        for fragment in ("sauf lorsqu’il constitue le seul moyen disponible",
                         "même lorsque leur transmission entraîne un surcoût"):
            with self.subTest(fragment=fragment):
                self.assertFalse(self.conditionne(fragment))

    def test_une_temporelle_ordinaire_conditionne(self):
        """Le motif purement structurel (ADP + SCONJ) a été REFUSÉ pour cela.

        « dès que la communication le permet » est une condition de référence :
        c'est la valeur de la particule qui compte, pas la structure.
        """
        for fragment in ("dès que la communication le permet",
                         "Quand le sol est gelé",
                         "Si aucun relais n’est disponible"):
            with self.subTest(fragment=fragment):
                self.assertTrue(self.conditionne(fragment))


class CritereFonctionnel(unittest.TestCase):
    """La dépendance `mark` plutôt que la seule catégorie du mot.

    spaCy étiquette « Comme » tantôt SCONJ, tantôt ADP selon le contexte.
    L'incohérence a mordu ce projet plusieurs fois — elle avait déjà coûté une
    relation au filtre de singularité. `dep_ == "mark"` dit la FONCTION, qui ne
    varie pas.
    """

    def test_comme_etiquete_adp_est_quand_meme_un_subordonnant(self):
        self.assertFalse(conclut("Comme l’étude conclut à des travaux lourds"))

    def test_une_conclusion_ordinaire_n_est_pas_touchee(self):
        for fragment in ("du porc", "il le garde", "le sel est appliqué"):
            with self.subTest(fragment=fragment):
                self.assertTrue(conclut(fragment))


class ConnecteurSeul(unittest.TestCase):
    def test_un_connecteur_isole_ne_conditionne_rien(self):
        """« Toutefois » articule le discours, il n'affirme aucun fait."""
        for fragment in ("Toutefois", "Néanmoins", "Mais"):
            with self.subTest(fragment=fragment):
                self.assertFalse(
                    syntaxe.peut_conditionner(fragment, 0, len(fragment)))

    def test_un_circonstant_bref_conditionne(self):
        """« En hiver » est une condition de référence : deux mots, pas un."""
        for fragment in ("En hiver", "En phase finale"):
            with self.subTest(fragment=fragment):
                self.assertTrue(
                    syntaxe.peut_conditionner(fragment, 0, len(fragment)))

if __name__ == "__main__":
    unittest.main()
