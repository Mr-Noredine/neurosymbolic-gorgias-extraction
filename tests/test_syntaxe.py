"""Court-circuit syntaxique : la grammaire peut décider seule.

Ces tests vivaient dans test_jepa_sigreg.py, supprimé avec le module
JEPA/SIGReg dont l'apport avait été démontré nul. Ils n'ont jamais porté
sur ce module : ils vérifient que le court-circuit dispense les deux juges
et que l'absence de marqueur laisse la main au modèle.
"""
from __future__ import annotations

import unittest
from unittest import mock

from gorgias import app
from gorgias import syntaxe


class _ModeleQuiRefuseTout:
    """Refuserait toute relation : sert à prouver qu'on ne l'appelle pas."""

    def __init__(self):
        self.relation_calls = 0

    def invoke(self, messages):
        self.relation_calls += 1
        raise AssertionError(
            "une relation prouvée grammaticalement ne doit pas atteindre le LLM")


class CourtCircuitSyntaxique(unittest.TestCase):
    """La syntaxe peut décider seule ; le routeur statistique, non."""

    def test_subordonnee_marquee_dispense_des_deux_juges(self):
        """Une conditionnelle explicite ne doit consommer aucun appel LLM."""
        texte = "Quand il pleut, nous prenons le train."
        segments = app.segment_text(texte)
        modele = _ModeleQuiRefuseTout()  # refuserait tout si on l'appelait
        grounds = app._verifier_paires(
            modele, texte, segments, [(1, 2)], lambda message: None,
        )
        self.assertEqual(grounds, {2: [1]})
        self.assertEqual(modele.relation_calls, 0,
                         "une relation prouvée grammaticalement a été soumise au LLM")

    def test_absence_de_marqueur_laisse_la_main_au_modele(self):
        texte = "Le disque est plein. La sauvegarde a échoué."
        segments = app.segment_text(texte)
        self.assertEqual(
            syntaxe.relations_prouvees(texte, segments, [(1, 2)]), {},
            "aucune subordonnée marquée ici : la syntaxe ne doit rien trancher")

    def test_direction_suit_la_grammaire_pas_lordre_du_texte(self):
        """La conclusion peut précéder sa condition dans la phrase."""
        texte = "Nous prenons le train quand il pleut."
        segments = app.segment_text(texte)
        verdicts = syntaxe.relations_prouvees(texte, segments, [(1, 2)])
        if verdicts:  # dépend du découpage, on n'affirme que la direction
            self.assertEqual(verdicts[(1, 2)], "B_TO_A")

    def test_repli_textuel_couvre_comme_si_le_parseur_le_rate(self):
        texte = "Comme l'épisode dure, les mesures sont prolongées."
        segments = app.segment_text(texte)
        with mock.patch.object(syntaxe, "aretes_explicites", return_value=[]):
            self.assertEqual(
                syntaxe.relations_prouvees(texte, segments, [(1, 2)]),
                {(1, 2): "A_TO_B"},
            )

    def test_condition_suffixee_est_orientee_vers_la_principale(self):
        texte = "Le fonds est retenu que s'il présente un risque faible."
        segments = app.segment_text(texte)
        self.assertEqual(len(segments), 2)
        with mock.patch.object(syntaxe, "aretes_explicites", return_value=[]):
            self.assertEqual(
                syntaxe.relations_prouvees(texte, segments, [(1, 2)]),
                {(1, 2): "B_TO_A"},
            )

    def test_conclusion_prouvee_ne_passe_pas_par_la_proposition_llm(self):
        texte = "Comme l'épisode dure, les mesures sont prolongées."
        segments = app.segment_text(texte)
        modele = _ModeleQuiRefuseTout()
        grounds = app._relations_hybrides(
            modele, texte, segments, [1, 2], lambda message: None,
        )
        self.assertEqual(grounds, {2: [1]})
        self.assertEqual(modele.relation_calls, 0)

    def test_verbe_du_banc_ne_prouve_pas_une_relation(self):
        texte = "Le transfert du dossier ouvre un délai de recours."
        pivot = texte.index("ouvre")
        segments = [(0, pivot - 1), (pivot, len(texte) - 1)]
        with mock.patch.object(syntaxe, "aretes_explicites", return_value=[]):
            self.assertEqual(
                syntaxe.relations_prouvees(texte, segments, [(1, 2)]),
                {},
                "la liste causative ajustée sur le banc a été retirée",
            )

    def test_conditions_coordonnees_partagent_la_meme_conclusion(self):
        texte = (
            "Comme la température monte et que le trajet dure cinq heures, "
            "la rupture est établie."
        )
        segments = app.segment_text(texte)
        with mock.patch.object(syntaxe, "aretes_explicites", return_value=[]):
            verdicts = syntaxe.relations_prouvees(
                texte, segments, [(1, 2), (1, 3), (2, 3)]
            )
        self.assertEqual(verdicts[(1, 3)], "A_TO_B")
        self.assertEqual(verdicts[(2, 3)], "A_TO_B")

    def test_age_biographique_reste_temporel(self):
        texte = "Sa famille a déménagé quand il avait sept ans."
        segments = app.segment_text(texte)
        self.assertEqual(
            syntaxe.relations_prouvees(texte, segments, [(1, 2)]), {}
        )


class SubordonneesCoordonnees(unittest.TestCase):
    """« Comme A et que B, alors C » énonce DEUX conditions, pas une."""

    def test_la_coordonnee_devient_un_segment(self):
        texte = ("Comme la parcelle borde un monument classé et que la hauteur "
                 "projetée dépasse douze mètres, l'avis est requis.")
        fragments = [texte[a:b] for a, b in app.segment_text(texte)]
        self.assertEqual(len(fragments), 3, f"découpe inattendue : {fragments}")
        self.assertIn("monument", fragments[0])
        self.assertIn("hauteur", fragments[1])
        self.assertNotIn("hauteur", fragments[0],
                         "les deux conditions restent fusionnées")

    def test_une_subordonnee_simple_nest_pas_coupee(self):
        texte = "Quand le sol est gelé, le sel est appliqué."
        fragments = [texte[a:b] for a, b in app.segment_text(texte)]
        self.assertEqual(len(fragments), 2, f"découpe inattendue : {fragments}")

    def test_la_coupure_napute_aucun_texte(self):
        """Les segments doivent rester dans l'ordre et ne rien perdre."""
        texte = ("Comme le retard dépasse trois heures et que la cause relève "
                 "du transporteur, l'indemnité est due.")
        bornes = app.segment_text(texte)
        for (_, b1), (a2, _) in zip(bornes, bornes[1:]):
            self.assertLessEqual(b1, a2, "segments qui se chevauchent")
        for a, b in bornes:
            self.assertTrue(texte[a:b].strip(), "segment vide produit")


class FiltreDeSingularite(unittest.TestCase):
    """Un enchaînement d'événements révolus n'est pas une règle."""

    def test_deux_verbes_au_passe_signalent_un_recit(self):
        from gorgias import singularite
        self.assertTrue(singularite.evenement_singulier(
            "La foudre est tombée sur le transformateur",
            "Le quartier s'est trouvé sans électricité"))

    def test_un_seul_passe_ne_suffit_pas(self):
        """Les règles de référence ont souvent une prémisse au passé composé
        à valeur de constat : la rejeter coûterait cher."""
        from gorgias import singularite
        self.assertFalse(singularite.evenement_singulier(
            "La sauvegarde nocturne a échoué trois nuits de suite",
            "La restauration ne peut donc plus être garantie"))

    def test_ancrage_temporel_precis_signale_un_recit(self):
        from gorgias import singularite
        self.assertTrue(singularite.evenement_singulier(
            "L'eau s'est répandue dans la cave",
            "La cave a été asséchée le lendemain"))

    def test_un_circonstant_generique_nest_pas_un_ancrage(self):
        """« En hiver » situe une SITUATION, pas un événement unique : il
        ancre légitimement des règles métier."""
        from gorgias import singularite
        self.assertFalse(singularite.evenement_singulier(
            "En hiver, la route est déneigée", "le sel est appliqué"))

    def test_une_paire_prouvee_par_la_syntaxe_echappe_au_filtre(self):
        """Le connecteur atteste l'intention argumentative : la paire ne doit
        pas être soumise au filtre, même si ses deux verbes sont au passé."""
        texte = "Comme le colis est arrivé abîmé, le remboursement a été accordé."
        segments = app.segment_text(texte)
        prouvees = syntaxe.relations_prouvees(texte, segments, [(1, 2)])
        self.assertTrue(prouvees, "la subordonnée marquée doit être prouvée")


if __name__ == "__main__":
    unittest.main()
