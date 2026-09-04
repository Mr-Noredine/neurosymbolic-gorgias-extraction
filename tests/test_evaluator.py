from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))



class EvaluationSemantique(unittest.TestCase):
    """Mission 4 : comparer les conséquences, pas la forme."""

    REFERENCE = (
        "T1\tContext 0 10\tsol gele\n"
        "T2\tOption 12 22\tsel appli\n"
        "T3\tContext 24 34\tpluie tom\n"
        "T4\tOption 36 46\tsable app\n"
        "T5\tContext 48 58\tles deux\n"
        "T6\tMarker 60 70\tprime sur\n"
        "E1\trule:T2 Condition:T1 Effect:T2\n"
        "E2\trule:T4 Condition:T3 Effect:T4\n"
        "E3\tprefer:T6 Winner:E2 Loser:E1 When:T5\n"
    )

    def test_priorite_active_evince_la_regle_perdante(self):
        from gorgias import lpp_asp
        programme = lpp_asp.vers_asp(self.REFERENCE, ["T1", "T3", "T5"])
        modeles = lpp_asp.modeles_stables(programme)
        self.assertEqual(len(modeles), 1)
        self.assertEqual(modeles[0], frozenset({"conclut(t4)"}),
                         "la règle préférée doit seule conclure")

    def test_sans_le_contexte_les_deux_regles_concluent(self):
        from gorgias import lpp_asp
        modeles = lpp_asp.modeles_stables(
            lpp_asp.vers_asp(self.REFERENCE, ["T1", "T3"]))
        self.assertEqual(modeles[0], frozenset({"conclut(t2)", "conclut(t4)"}),
                         "hors de son contexte, la priorité ne s'applique pas")

    def test_annotation_identique_score_parfait(self):
        from gorgias import lpp_asp
        mesure = lpp_asp.comparer(self.REFERENCE, self.REFERENCE)
        self.assertEqual(mesure["f1"], 1.0)
        self.assertEqual(mesure["accords_conclusions"], mesure["essais"])

    def test_tous_les_sous_ensembles_de_contextes_sont_essayes(self):
        """Régime exhaustif : la comparaison est démontrée, pas sondée."""
        from gorgias import lpp_asp
        essais, regime = lpp_asp.echantillon_contextes(self.REFERENCE)
        self.assertEqual(regime, "exhaustif")
        self.assertEqual(len(essais), 2 ** 3,
                         "trois contextes -> huit sous-ensembles")

    def test_marqueur_discontinu_est_reconnu(self):
        """« préfère X au Y » place les termes entre le verbe et sa préposition."""
        from gorgias import priorites
        self.assertIsNotNone(
            priorites.marqueur_dans("En hiver, il préfère le porc au poulet"))
        self.assertIsNone(
            priorites.marqueur_dans("Il a préféré rentrer chez lui"),
            "un usage non comparatif ne doit pas déclencher")

    def test_priorite_inversee_est_detectee(self):
        """Le scoreur structurel voit deux règles justes ; la sémantique voit
        que les conclusions diffèrent."""
        from gorgias import lpp_asp
        inverse = self.REFERENCE.replace("Winner:E2 Loser:E1", "Winner:E1 Loser:E2")
        mesure = lpp_asp.comparer(self.REFERENCE, inverse)
        self.assertLess(mesure["f1"], 1.0)
        self.assertLess(mesure["accords_conclusions"], mesure["essais"])

    def test_regle_bien_formee_qui_ne_conclut_rien(self):
        """Biais inverse : une règle dont la condition n'est jamais activable
        compte juste pour le scoreur structurel, mais ne produit aucune
        conséquence."""
        from gorgias import lpp_asp
        modeles = lpp_asp.modeles_stables(
            lpp_asp.vers_asp(self.REFERENCE, []))
        self.assertEqual(modeles[0], frozenset(),
                         "sans contexte actif, aucune conclusion")


class FormeLogique(unittest.TestCase):
    """Mission 3 : arité adaptée à la langue réelle, pas au moule SVO."""

    def test_passif_donne_un_predicat_unaire(self):
        from gorgias import predicats
        self.assertEqual(predicats.forme_logique("le sel est appliqué"),
                         "appliquer(sel)")

    def test_groupe_nominal_donne_un_atome_constant(self):
        from gorgias import predicats
        forme = predicats.forme_logique("les courses de la semaine")
        self.assertIsNotNone(forme)
        self.assertNotIn("(", forme, "un groupe nominal n'a pas de foncteur")

    def test_negation_prefixe_le_litteral(self):
        from gorgias import predicats
        forme = predicats.forme_logique(
            "Le client stratégique n'obtient pas de remise")
        self.assertTrue(forme.startswith("-"),
                        f"négation non détectée dans {forme!r}")

    def test_meme_lemme_pour_deux_formulations(self):
        """C'est l'objectif : rendre l'unification possible."""
        from gorgias import predicats
        a = predicats.forme_logique("le sel est appliqué")
        b = predicats.forme_logique("on applique le sel")
        self.assertTrue(a.startswith("appliquer") and b.startswith("appliquer"),
                        f"{a!r} et {b!r} devraient partager leur foncteur")


class AncragePriorites(unittest.TestCase):
    """Problème 4 : une priorité exige un classement repéré symboliquement."""

    def test_document_sans_classement_ne_produit_aucune_priorite(self):
        """Mesuré : là où aucun classement n'est détecté, le modèle produit
        8 priorités et aucune n'est juste."""
        from gorgias import app, priorites
        texte = ("Comme la parcelle borde un monument classé, l'avis est requis. "
                 "Le permis est accordé.")
        segments = app.segment_text(texte)
        retires, _ = priorites.isoler(texte, segments)
        self.assertEqual(retires, [],
                         "aucun classement ne doit être repéré dans ce texte")

    def test_un_classement_explicite_est_bien_repere(self):
        from gorgias import app, priorites
        texte = ("En cas d'arrêt de ligne, le remplacement l'emporte sur "
                 "la réparation.")
        segments = app.segment_text(texte)
        retires, amorces = priorites.isoler(texte, segments)
        self.assertEqual(len(retires), 1)
        self.assertIn("emporte sur", amorces[retires[0]])


class DecoupageDesClassements(unittest.TestCase):
    """Le découpage gagnante/perdante ne doit jamais trancher un mot."""

    def test_comparatif_discontinu_lit_les_groupes(self):
        from gorgias import priorites
        self.assertEqual(
            priorites.cotes_du_classement("il préfère le porc au poulet"),
            ("le porc", "poulet"))

    def test_amorce_contigue_exige_une_frontiere_de_mot(self):
        """« prefere a » ne doit pas correspondre au préfixe de « préfère aux »."""
        from gorgias import priorites
        cotes = priorites.cotes_du_classement("il les préfère aux deux autres")
        if cotes is not None:
            for cote in cotes:
                self.assertNotIn("ux deux", cote,
                                 f"découpage au milieu d'un mot : {cotes}")

    def test_les_amorces_contigues_restent_intactes(self):
        from gorgias import priorites
        self.assertEqual(
            priorites.cotes_du_classement("le camion prime sur la péniche"),
            ("le camion", "la péniche"))
        self.assertEqual(
            priorites.cotes_du_classement(
                "le remplacement l'emporte sur la réparation"),
            ("le remplacement", "la réparation"))

    def test_comparatif_elliptique_sabstient(self):
        """Sans second terme exprimé, mieux vaut ne rien produire."""
        from gorgias import priorites
        self.assertIsNone(
            priorites.cotes_du_classement("il préfère le poulet même en hiver"))


if __name__ == "__main__":
    unittest.main()
