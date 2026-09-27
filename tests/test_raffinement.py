"""Les méta-préférences, et les mécanismes déterministes qui les rendent
possibles.

Le défaut d'origine n'était pas un défaut de rappel : le chemin déterministe
ne posait JAMAIS de ``reactivates``, si bien qu'aucune méta-préférence n'était
compilable quel que soit le texte. `_compile_preferences` déduit la
méta-préférence de l'inclusion stricte des scénarios cumulés, et cette
inclusion n'était jamais établie. Mesuré : 0 sur les 6 de `data/cas`, 0 sur
les 20 de `data/synthetique`.

Ce que fixent ces tests, et qui commande tout le reste : **l'inversion de deux
priorités ne suffit pas à conclure au raffinement**. La famille « frères » du
corpus synthétique range la même paire de règles en sens inverse dans deux
situations SŒURS, et n'attend aucune méta-préférence. Il faut en plus que la
phrase du classement annonce le raffinement — adversative en ouverture, ou
reprise concessive en clôture. Discriminant mesuré sur les 188 documents
annotés du dépôt : 25 documents attendent une méta-préférence et 25
déclenchent l'annonce, 202 n'en attendent aucune et AUCUN ne la déclenche.

Tous les tests de ce fichier tournent avec un modèle MUET : ce qu'ils
vérifient est produit sans aucun appel à un modèle de langue, donc de façon
reproductible.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from gorgias import app, priorites, syntaxe

RACINE = Path(__file__).resolve().parent.parent


def _annoter_sans_modele(texte: str) -> str:
    """Exécute le pipeline avec l'étape de modèle rendue indisponible.

    Aucun appel réseau : `_ask_stage` lève, donc chaque étage générative
    échoue proprement et seul le chemin symbolique subsiste. C'est le
    « plancher symbolique » du système.
    """
    vrai = app._ask_stage

    def muet(*args, **kwargs):
        raise app.StageError("modèle indisponible (test)")

    app._ask_stage = muet
    try:
        return app.annotate(texte, model=object(), output_format="brat")
    except ValueError:
        return ""
    finally:
        app._ask_stage = vrai


def _cas(nom: str) -> tuple[str, str]:
    texte = (RACINE / "data" / "cas" / f"{nom}.txt").read_text(encoding="utf-8")
    reference = (RACINE / "data" / "cas" / f"{nom}.ann").read_text(encoding="utf-8")
    return texte, reference


class AnnonceDuRaffinement(unittest.TestCase):
    """Le discriminant entre raffinement et scénarios frères."""

    def test_adversative_en_ouverture_annonce_un_raffinement(self):
        self.assertTrue(priorites.annonce_un_raffinement(
            "Mais si la route est déneigée, la voiture l'emporte sur le train."))

    def test_reprise_concessive_en_cloture_annonce_un_raffinement(self):
        self.assertTrue(priorites.annonce_un_raffinement(
            "Si la pente est faible, le saleur prime sur la lame, "
            "même en montagne."))

    def test_scenarios_freres_n_annoncent_aucun_raffinement(self):
        """Le contre-exemple qui donne sa forme au test.

        « En hiver, le train passe avant la voiture. Aux heures de pointe, la
        voiture prime sur le train. » : deux situations sœurs, deux priorités
        inverses, AUCUNE méta-préférence attendue.
        """
        for phrase in ("Aux heures de pointe, la voiture prime sur le train.",
                       "En été, la voiture l'emporte sur le train.",
                       "En périphérie, l'éclairage tamisé prime sur le renforcé."):
            with self.subTest(phrase=phrase):
                self.assertFalse(priorites.annonce_un_raffinement(phrase))

    def test_l_annonce_se_lit_dans_la_phrase_du_classement_seulement(self):
        """Une adversative ailleurs dans le document ne compte pas.

        Les documents « frères » du corpus synthétique placent «  En revanche,
        la ligne compte dix-sept stations » dans une phrase SANS classement :
        une recherche à l'échelle du document la prendrait pour une annonce.
        """
        texte = ("En hiver, le train passe avant la voiture. "
                 "Aux heures de pointe, la voiture prime sur le train. "
                 "En revanche, la ligne compte dix-sept stations.")
        debut = texte.index("la voiture prime")
        phrase = priorites.phrase_autour(texte, debut, debut + 10)
        self.assertNotIn("En revanche", phrase)
        self.assertFalse(priorites.annonce_un_raffinement(phrase))


class RepriseDeContexte(unittest.TestCase):
    """« même en hiver » reprend un contexte ; « même si » défait une règle."""

    def test_adjoint_de_reprise_est_reconnu(self):
        for fragment in ("même en hiver", "même à budget contraint",
                         "même en sol argileux", "même pour un contrat récent"):
            with self.subTest(fragment=fragment):
                self.assertTrue(priorites.est_reprise_de_contexte(fragment))

    def test_meme_si_n_est_pas_une_reprise(self):
        """« même si » défait la règle au lieu de rappeler une situation, et
        porte un marqueur de référence sur `48-drones-autonomes`."""
        for fragment in ("même si le drone est en vol",
                         "même lorsque la transmission est entravée"):
            with self.subTest(fragment=fragment):
                self.assertFalse(priorites.est_reprise_de_contexte(fragment))

    def test_la_reprise_est_soustraite_du_flux(self):
        """Les références ne donnent aucune entité à ces adjoints."""
        texte, _ = _cas("09-raffinement-explicite")
        produit = _annoter_sans_modele(texte)
        self.assertNotIn("même en hiver", produit)


class ComparatifAPerdanteElidee(unittest.TestCase):
    """« il préfère le poulet même en hiver » nomme la gagnante seule."""

    FRAGMENT = "il préfère le poulet même en hiver"

    def test_la_gagnante_est_lue_la_perdante_reste_implicite(self):
        self.assertEqual(
            priorites.cote_gagnante_raffinee(self.FRAGMENT), "le poulet")
        self.assertIsNone(
            priorites.cotes_du_classement(self.FRAGMENT),
            "aucun couple ne peut être rendu : la perdante est anaphorique")

    def test_le_marqueur_s_arrete_a_l_adjoint_concessif(self):
        """Les références bornent le marqueur préposition de reprise exclue."""
        empan = priorites.empan_du_marqueur(self.FRAGMENT)
        self.assertIsNotNone(empan)
        self.assertEqual(self.FRAGMENT[empan[0]:empan[1]],
                         "préfère le poulet même")

    def test_le_segment_devient_un_classement_donc_sort_du_flux(self):
        texte, _ = _cas("01-courses-3-niveaux")
        segments = app.segment_text(app.neutraliser_enumerations(texte))
        classements, _ = priorites.isoler(
            app.neutraliser_enumerations(texte), segments)
        fragments = [texte[segments[n - 1][0]:segments[n - 1][1]]
                     for n in classements]
        self.assertIn("il préfère le poulet même en hiver", fragments)

    def test_un_usage_non_comparatif_ne_declenche_pas(self):
        self.assertIsNone(
            priorites.cote_gagnante_raffinee("il a préféré rentrer même tard"),
            "sans préposition de reprise, ce n'est pas un classement")


class LexiqueDesAmorces(unittest.TestCase):
    def test_les_amorces_sont_essayees_de_la_plus_longue_a_la_plus_courte(self):
        longueurs = [len(a) for a in priorites._AMORCES_PAR_LONGUEUR]
        self.assertEqual(longueurs, sorted(longueurs, reverse=True))

    def test_prioritaire_par_rapport_est_reconnu_et_decoupe(self):
        """Forme longue portant un marqueur de référence sur 48-drones.

        La préposition finale reste hors de l'amorce : « par rapport à » et
        « par rapport au » sont la même tournure.
        """
        for clause, attendu in (
            ("le drone devient prioritaire par rapport au véhicule",
             ("le drone devient", "au véhicule")),
            ("la piste est prioritaire par rapport à la route",
             ("la piste est", "à la route")),
        ):
            with self.subTest(clause=clause):
                self.assertEqual(priorites.cotes_du_classement(clause), attendu)

    def test_les_amorces_historiques_restent_intactes(self):
        self.assertEqual(
            priorites.cotes_du_classement("le camion prime sur la péniche"),
            ("le camion", "la péniche"))
        self.assertEqual(
            priorites.cotes_du_classement(
                "le remplacement l'emporte sur la réparation"),
            ("le remplacement", "la réparation"))


class ConnecteurDeConsequence(unittest.TestCase):
    """« Par conséquent » isolé : la conclusion suit, la prémisse précède."""

    def test_la_forme_isolee_est_reconnue(self):
        for fragment in ("Par conséquent", "Dès lors", "Donc",
                         "C'est pourquoi", "De ce fait"):
            with self.subTest(fragment=fragment):
                self.assertTrue(syntaxe.est_connecteur_de_consequence(fragment))

    def test_des_lors_QUE_est_une_condition_pas_une_consequence(self):
        """L'ancrage sur la fin de segment suffit à les séparer."""
        self.assertFalse(syntaxe.est_connecteur_de_consequence(
            "dès lors que le sinistre relève d'un événement climatique"))

    def test_le_connecteur_enchasse_est_ecarte(self):
        """« La restauration ne peut DONC plus être garantie » : les
        références y demandent DEUX prémisses, que la surface ne permet pas de
        compter. Produire la seule prémisse adjacente fabriquerait une règle
        fausse."""
        self.assertFalse(syntaxe.est_connecteur_de_consequence(
            "La restauration ne peut donc plus être garantie"))

    def test_la_premisse_est_la_principale_pas_la_subordonnee(self):
        """« A, car B. Par conséquent, C. » : C suit de A, jamais de B."""
        texte = ("une relecture externe est commandée, car la traduction "
                 "n'a pas été relue. Par conséquent, le tirage doit être "
                 "réduit.")
        segments = app.segment_text(texte)
        fragments = [texte[d:f] for d, f in segments]
        aretes = syntaxe.aretes_textuelles(texte, segments)
        principale = segments[fragments.index(
            "une relecture externe est commandée")]
        conclusion = segments[fragments.index("le tirage doit être réduit")]
        cause = segments[fragments.index(
            "car la traduction n'a pas été relue")]
        self.assertIn((principale, conclusion), aretes)
        self.assertNotIn((cause, conclusion), aretes)

    def test_le_connecteur_isole_ne_recoit_aucune_entite(self):
        texte = ("Quand le service dépasse sa capacité, les admissions sont "
                 "reportées. Par conséquent, le plan blanc est déclenché.")
        produit = _annoter_sans_modele(texte)
        self.assertNotIn("Par conséquent", produit)
        self.assertIn("le plan blanc est déclenché", produit)


class CausePostposee(unittest.TestCase):
    """« car » est une conjonction de COORDINATION, invisible au parseur."""

    def test_car_fonde_la_principale_qui_le_precede(self):
        texte = ("une relecture externe est commandée, car la traduction "
                 "n'a pas été relue.")
        segments = app.segment_text(texte)
        aretes = syntaxe.aretes_textuelles(texte, segments)
        self.assertIn((segments[1], segments[0]), aretes)

    def test_la_cause_postposee_produit_bien_la_regle(self):
        texte = ("nous hébergeons sur serveur dédié, car le certificat "
                 "expire dans quarante-huit heures.")
        produit = _annoter_sans_modele(texte)
        self.assertIn("\trule:", produit)
        self.assertIn("Option", produit)


class FrontiereDePhraseGrammaticale(unittest.TestCase):
    """Un point suivi d'une minuscule : la grammaire tranche, pas la casse."""

    def test_une_clause_autonome_ouvre_une_phrase(self):
        self.assertTrue(syntaxe.ouvre_une_clause(
            "les couloirs ont été rénovés l'an dernier"))

    def test_un_groupe_sans_verbe_fini_n_en_ouvre_pas(self):
        self.assertFalse(syntaxe.ouvre_une_clause("le reste du document"))

    def test_deux_phrases_a_initiale_minuscule_sont_separees(self):
        texte = ("le plan blanc est déclenché. les couloirs ont été rénovés "
                 "l'an dernier.")
        fragments = [texte[d:f] for d, f in app.segment_text(texte)]
        self.assertIn("le plan blanc est déclenché", fragments)

    def test_une_abreviation_ne_coupe_toujours_pas(self):
        texte = "Voir art. 5 du règlement pour le détail."
        self.assertEqual(len(app.segment_text(texte)), 1)


class ConditionDetacheeEnTeteDePhrase(unittest.TestCase):
    """Une subordonnée POSTPOSÉE ne conditionne pas la phrase suivante."""

    def test_la_condition_en_tete_conditionne_bien_sa_principale(self):
        texte = "Quand la commande dépasse mille euros, une remise s'applique."
        segments = app.segment_text(texte)
        self.assertIn((segments[0], segments[1]),
                      syntaxe.aretes_textuelles(texte, segments))

    def test_la_subordonnee_postposee_n_enjambe_pas_le_point(self):
        """Mesuré sur `g0131-preference-informatique` : l'arête produisait une
        règle fausse, une Option là où la référence veut un Context, et un
        ancrage de priorité faux."""
        texte = ("nous hébergeons dans le cloud, puisque la sauvegarde "
                 "nocturne a échoué. Pour un budget contraint, le serveur "
                 "dédié l'emporte sur le cloud.")
        segments = app.segment_text(texte)
        fragments = [texte[d:f] for d, f in segments]
        cause = segments[fragments.index("puisque la sauvegarde nocturne a échoué")]
        suivante = segments[fragments.index("Pour un budget contraint")]
        self.assertNotIn((cause, suivante),
                         syntaxe.aretes_textuelles(texte, segments))


class AncrageDuScenario(unittest.TestCase):
    """Le scénario d'une priorité se lit dans la phrase du classement."""

    def test_le_when_ne_traverse_pas_la_frontiere_de_phrase(self):
        """Mesuré sur les 140 priorités de référence de `data/cas` et
        `data/synthetique` : 135 ancrages sont dans la phrase du marqueur, et
        les 5 autres sont des reprises ramenées à leur première occurrence."""
        texte, reference = _cas("32-tri-urgences")
        produit = _annoter_sans_modele(texte)
        self.assertIn("En cas de suspicion de fracture", produit)
        # La condition de la règle gagnante ne doit PAS contenir le contexte
        # de la priorité : « en cas de » ouvre ici sa propre phrase.
        fracture = next(ligne for ligne in produit.splitlines()
                        if "En cas de suspicion de fracture" in ligne)
        identifiant = fracture.split("\t")[0]
        for ligne in produit.splitlines():
            if "\trule:" in ligne:
                self.assertNotIn(f"Condition:{identifiant} ", ligne + " ")

    def test_le_terme_gauche_sert_de_scenario_a_defaut_de_circonstant(self):
        """« Une demande venant d'un supérieur hiérarchique l'emporte sur son
        propre besoin » ne pose aucune circonstance à part : la circonstance
        EST le terme comparé, et la référence lui donne son entité."""
        texte, _ = _cas("03-pret-objet")
        produit = _annoter_sans_modele(texte)
        self.assertIn("Une demande venant d'un supérieur hiérarchique", produit)


class MetaPreferencesDeBoutEnBout(unittest.TestCase):
    """Sans aucun appel au modèle, la structure complète doit sortir."""

    ATTENDUS = ("09-raffinement-explicite", "33-deneigement",
                "34-hebergement", "35-semis")
    FRERES = ("10-scenarios-freres", "36-eclairage", "37-arbitrage-sportif")

    def test_les_documents_de_raffinement_produisent_leur_meta_preference(self):
        for nom in self.ATTENDUS:
            with self.subTest(document=nom):
                texte, reference = _cas(nom)
                produit = _annoter_sans_modele(texte)
                self.assertEqual(
                    produit.count("\tmeta_prefer:"), 1,
                    f"{nom} : une méta-préférence attendue")
                self.assertEqual(produit.count("\tprefer:"), 2)
                self.assertEqual(reference.count("\tmeta_prefer:"), 1,
                                 "la référence elle-même en attend une")

    def test_les_scenarios_freres_n_en_produisent_aucune(self):
        for nom in self.FRERES:
            with self.subTest(document=nom):
                texte, reference = _cas(nom)
                produit = _annoter_sans_modele(texte)
                self.assertEqual(produit.count("\tmeta_prefer:"), 0,
                                 f"{nom} : deux priorités sœurs, pas un "
                                 "raffinement")
                self.assertEqual(reference.count("\tmeta_prefer:"), 0)
                self.assertEqual(produit.count("\tprefer:"), 2,
                                 "les deux priorités doivent bien sortir")

    def test_la_meta_preference_est_ancree_sur_le_delta_du_raffinement(self):
        texte, reference = _cas("09-raffinement-explicite")
        produit = _annoter_sans_modele(texte)
        meta = next(ligne for ligne in produit.splitlines()
                    if "\tmeta_prefer:" in ligne)
        quand = meta.split("When:")[1].split()[0]
        empan = next(ligne for ligne in produit.splitlines()
                     if ligne.startswith(f"{quand}\t"))
        self.assertIn("la route est déneigée", empan)

    def test_les_recits_negatifs_restent_muets(self):
        """Aucun des mécanismes ajoutés ne doit pouvoir polluer un récit."""
        dossier = RACINE / "data" / "narratif"
        pollues = [source.name for source in sorted(dossier.glob("*.txt"))
                   if _annoter_sans_modele(
                       source.read_text(encoding="utf-8")).strip()]
        self.assertEqual(pollues, [])


if __name__ == "__main__":
    unittest.main()
