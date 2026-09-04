"""Le jeu HORS ÉCHANTILLON, et ce qui le rend valide.

Les 48 documents de `data/cas` ont TOUS servi à régler le système : chaque
mécanisme y a été mesuré avant d'être écrit. Aucun chiffre du projet ne dit
donc ce que vaut le système sur du texte qu'il n'a jamais vu — c'est le
plafond scientifique que ce jeu lève.

Ce qui le rend valide, et qu'il faut protéger :

  - les textes sont RÉELS et EXTERNES : dix paragraphes du RGPD (règlement
    UE 2016/679), pris verbatim sur EUR-Lex ;
  - la sélection est MÉCANIQUE, donc reproductible et non choisie à la main :
    paragraphe numéroté d'article, 160 à 420 caractères, contenant un marqueur
    conditionnel, phrase autonome — les dix premiers dans l'ordre du document ;
  - ils ont été GELÉS avant d'être annotés, et n'ont jamais servi à dériver
    un mécanisme.

Ce qu'il ne corrige PAS : l'annotateur reste unique. Ce jeu mesure la
généralisation, pas la fiabilité de l'annotation.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
JEU = RACINE / "data" / "horschantillon"
JEU2 = RACINE / "data" / "horschantillon2"
JEU3 = RACINE / "data" / "horschantillon3"
JEU4 = RACINE / "data" / "horschantillon4"

from gorgias import app  # noqa: E402


class JeuHorsEchantillon(unittest.TestCase):
    def test_le_jeu_est_present_et_complet(self):
        textes = sorted(JEU.glob("*.txt"))
        self.assertEqual(len(textes), 10)
        for texte in textes:
            with self.subTest(document=texte.stem):
                self.assertTrue(texte.with_suffix(".ann").exists())

    def test_la_provenance_est_enregistree(self):
        """Sans la règle de sélection, le jeu n'est pas reproductible."""
        source = json.loads((JEU / "SOURCE.json").read_text(encoding="utf-8"))
        for cle in ("source", "regle", "sha256_source", "documents"):
            with self.subTest(cle=cle):
                self.assertIn(cle, source)
        self.assertIn("2016/679", source["source"])
        self.assertEqual(len(source["documents"]), 10)

    def test_les_offsets_sont_exacts(self):
        """Un empan décalé rend la mesure fausse sans rien signaler."""
        for annotation in sorted(JEU.glob("*.ann")):
            texte = annotation.with_suffix(".txt").read_text(encoding="utf-8")
            for ligne in annotation.read_text(encoding="utf-8").splitlines():
                if not ligne.startswith("T"):
                    continue
                identifiant, champs, fragment = ligne.split("\t")
                _, debut, fin = champs.split()
                with self.subTest(document=annotation.stem, entite=identifiant):
                    # EUR-Lex emploie des espaces insécables : la comparaison
                    # les normalise, la substitution conservant la longueur.
                    self.assertEqual(
                        texte[int(debut):int(fin)].replace(" ", " "),
                        fragment.replace(" ", " "),
                    )

    def test_le_graphe_de_chaque_annotation_est_valide(self):
        for annotation in sorted(JEU.glob("*.ann")):
            with self.subTest(document=annotation.stem):
                app.validate_annotation_graph(
                    annotation.read_text(encoding="utf-8").strip())

    def test_le_jeu_est_disjoint_du_banc(self):
        """S'il recoupait `data/cas`, il ne mesurerait plus rien."""
        banc = {p.read_text(encoding="utf-8").strip()
                for p in (RACINE / "data" / "cas").glob("*.txt")}
        for texte in JEU.glob("*.txt"):
            with self.subTest(document=texte.stem):
                self.assertNotIn(texte.read_text(encoding="utf-8").strip(), banc)


class LotsSuivants(unittest.TestCase):
    """Les lots 11-20 et 21-30, et pourquoi il a fallu les constituer.

    Un jeu sur lequel on a CORRIGÉ quelque chose ne mesure plus la
    généralisation, il mesure un réglage. Le lot 1 a donné la numérotation et
    les incises ; le lot 2 a donné le subordonnant rendu à sa proposition.
    Chacun a donc été remplacé par le suivant, tiré de la même règle mécanique
    au rang suivant. Le dernier lot est le seul chiffre honnête à un instant
    donné — et il se périme dès qu'on en tire un mécanisme.
    """

    LOTS = ((JEU2, 11), (JEU3, 21), (JEU4, 31))

    def test_les_lots_sont_presents_et_complets(self):
        for jeu, _ in self.LOTS:
            textes = sorted(jeu.glob("*.txt"))
            with self.subTest(lot=jeu.name):
                self.assertEqual(len(textes), 10)
                for texte in textes:
                    self.assertTrue(texte.with_suffix(".ann").exists())

    def test_la_regle_de_selection_est_executable(self):
        """Une règle en prose ne se rejoue pas ; celle-ci est un script."""
        self.assertTrue((JEU / "selectionner.py").exists())
        for jeu, rang in self.LOTS:
            source = json.loads(
                (jeu / "SELECTION.json").read_text(encoding="utf-8"))
            with self.subTest(lot=jeu.name):
                self.assertEqual(source["rang_debut"], rang)
                self.assertIn("selectionner.py", source["reproduction"])

    def test_les_offsets_sont_exacts(self):
        for jeu, _ in self.LOTS:
            for annotation in sorted(jeu.glob("*.ann")):
                texte = annotation.with_suffix(".txt").read_text(encoding="utf-8")
                for ligne in annotation.read_text(encoding="utf-8").splitlines():
                    if not ligne.startswith("T"):
                        continue
                    identifiant, champs, fragment = ligne.split("\t")
                    _, debut, fin = champs.split()
                    with self.subTest(document=annotation.stem,
                                      entite=identifiant):
                        self.assertEqual(texte[int(debut):int(fin)], fragment)

    def test_les_graphes_sont_valides(self):
        for jeu, _ in self.LOTS:
            for annotation in sorted(jeu.glob("*.ann")):
                contenu = annotation.read_text(encoding="utf-8").strip()
                with self.subTest(document=annotation.stem):
                    if contenu:
                        app.validate_annotation_graph(contenu)

    def test_les_sondes_de_precision_sont_vides(self):
        """`hs11` et `hs28` n'ont pour seule subordonnée qu'une exception ou une
        extension de portée : ils ne portent AUCUNE règle, et tout ce que le
        système y produira sera un faux positif. Décidé avant la mesure."""
        self.assertEqual((JEU2 / "hs11.ann").read_text(encoding="utf-8"), "")
        self.assertEqual((JEU3 / "hs28.ann").read_text(encoding="utf-8"), "")

    def test_tous_les_lots_sont_deux_a_deux_disjoints(self):
        """Un même texte dans deux jeux fausserait la mesure sans rien dire.

        UN doublon existe et il est ANTÉRIEUR à ce travail :
        `cas/07-transitions-trompeuses` est mot pour mot `narratif/n01-gare`.
        Les deux références sont VIDES — c'est un négatif rangé parmi les
        positifs — donc il n'entre dans aucun dénominateur et ne déplace aucun
        ratio ; il gonfle seulement le décompte annoncé de « 48 documents
        argumentatifs ». Il est constaté ici, pas supprimé : retirer une
        donnée du banc n'est pas à moi de le décider.
        """
        connu = frozenset({"07-transitions-trompeuses", "n01-gare"})
        vus: dict[str, str] = {}
        for dossier in (RACINE / "data" / "cas", RACINE / "data" / "narratif",
                        JEU, JEU2, JEU3, JEU4):
            for texte in sorted(dossier.glob("*.txt")):
                contenu = texte.read_text(encoding="utf-8").strip()
                jumeau = vus.get(contenu)
                with self.subTest(document=f"{dossier.name}/{texte.stem}"):
                    if jumeau is not None:
                        self.assertEqual({jumeau, texte.stem}, set(connu))
                vus[contenu] = texte.stem


class MarqueurDEnumeration(unittest.TestCase):
    """« 4.   Au moment de déterminer… » — le numéro n'est pas la proposition.

    Collé au premier segment, il en devenait le premier jeton : `dep_ ==
    "mark"`, `peut_conclure` et le court-circuit étaient tous aveuglés, et les
    empans décalés par rapport à la référence. Le banc ne pouvait pas le
    montrer — sa prose n'est pas numérotée — et c'est le jeu hors échantillon
    qui l'a révélé au premier essai, sur du texte réglementaire réel.
    """

    def test_le_numero_est_retire_du_segment(self):
        texte = "4.   Lorsque le seuil est franchi, une alerte est émise."
        premier = app.segment_text(texte)[0]
        self.assertTrue(texte[premier[0]:premier[1]].startswith("Lorsque"))

    def test_les_formes_usuelles_sont_couvertes(self):
        for tete in ("4.", "12)", "a)", "iv."):
            with self.subTest(tete=tete):
                texte = f"{tete}   Quand le sol gèle, le sel est répandu."
                premier = app.segment_text(texte)[0]
                self.assertTrue(texte[premier[0]:premier[1]].startswith("Quand"))

    def test_une_annee_ou_une_civilite_ne_sont_pas_entamees(self):
        """Le motif est borné à trois chiffres et aux minuscules."""
        for texte in ("2026. L'année suivante le seuil change.",
                      "M. Dupont signe le contrat."):
            with self.subTest(texte=texte):
                premier = app.segment_text(texte)[0]
                self.assertEqual(premier[0], 0, "rien ne doit être rogné")

    def test_la_neutralisation_conserve_les_offsets(self):
        """Substituer des espaces, à longueur égale, ne bouge aucun empan."""
        for chemin in sorted(JEU.glob("*.txt")):
            texte = chemin.read_text(encoding="utf-8")
            with self.subTest(document=chemin.stem):
                self.assertEqual(
                    len(app.neutraliser_enumerations(texte)), len(texte))

    def test_aucun_segment_ne_recouvre_un_caractere_efface(self):
        """Sinon le `.ann` porterait des espaces là où le fichier a « 4. »."""
        for chemin in sorted(JEU.glob("*.txt")):
            texte = chemin.read_text(encoding="utf-8")
            propre = app.neutraliser_enumerations(texte)
            for debut, fin in app.segment_text(texte):
                with self.subTest(document=chemin.stem, empan=(debut, fin)):
                    self.assertEqual(propre[debut:fin], texte[debut:fin])

    def test_le_correctif_est_inerte_sur_le_banc(self):
        """Aucun segment des 80 documents historiques ne porte le motif."""
        for dossier in ("cas", "narratif"):
            for chemin in sorted((RACINE / "data" / dossier).glob("*.txt")):
                texte = chemin.read_text(encoding="utf-8")
                for debut, fin in app.segment_text(texte):
                    with self.subTest(document=chemin.stem):
                        self.assertIsNone(
                            app._ENUMERATION_EN_TETE.match(texte, debut, fin))

class IncisesRecollees(unittest.TestCase):
    """« …, y compris X, » n'est pas une proposition : c'est une apposition.

    Découpée, elle était étiquetée puis câblée comme condition — la source
    dominante de faux positifs hors échantillon une fois la numérotation
    traitée.
    """

    def test_une_apposition_rejoint_la_proposition_precedente(self):
        texte = ("Lorsque les données sont traitées à des fins de prospection, "
                 "la personne concernée a le droit de s'opposer, y compris au "
                 "profilage dans la mesure où il est lié à cette prospection.")
        segments = [texte[a:b] for a, b in app.segment_text(texte)]
        self.assertEqual(len(segments), 2)
        self.assertIn("y compris au profilage", segments[1])

    def test_une_exception_reste_une_proposition(self):
        """« sauf … » porte une condition négative : la fusionner l'effacerait."""
        texte = ("Le drone ne peut être envoyé vers une nouvelle zone, "
                 "sauf lorsqu'il constitue le seul moyen de maintenir la liaison.")
        self.assertEqual(len(app.segment_text(texte)), 2)

    def test_un_point_interdit_la_fusion(self):
        """Par-dessus une frontière de phrase, « notamment » n'est plus une
        incise : la fusion doit s'arrêter à la ponctuation forte."""
        texte = ("La personne concernée a le droit de s'opposer. "
                 "Notamment, le profilage est visé par cette opposition.")
        premier = app.segment_text(texte)[0]
        self.assertEqual(texte[premier[0]:premier[1]],
                         "La personne concernée a le droit de s'opposer")

    def test_la_fusion_est_inerte_sur_le_banc(self):
        """509 segments avant, 509 après : le banc ne peut pas avoir bougé."""
        total = 0
        for dossier in ("cas", "narratif"):
            for chemin in sorted((RACINE / "data" / dossier).glob("*.txt")):
                total += len(app.segment_text(
                    chemin.read_text(encoding="utf-8")))
        self.assertEqual(total, 509)


class ConnecteurRendu(unittest.TestCase):
    """« Si, et dans la mesure où, il n'est pas possible… »

    La virgule INTERNE au connecteur déclenchait la découpe : le subordonnant
    devenait un segment à part, étiqueté puis câblé, et la proposition
    suivante perdait le mot qui portait sa conditionnalité.
    """

    def test_le_subordonnant_rejoint_sa_proposition(self):
        texte = ("Si, et dans la mesure où, il n'est pas possible de tout "
                 "fournir, les informations sont communiquées par étapes.")
        segments = [texte[a:b] for a, b in app.segment_text(texte)]
        self.assertEqual(len(segments), 2)
        self.assertTrue(segments[0].startswith("Si, et dans la mesure où"))
        self.assertIn("il n'est pas possible", segments[0])

    def test_une_proposition_entiere_n_est_pas_absorbee(self):
        """Le critère exige l'ABSENCE de prédication, pas la présence d'un
        connecteur : une subordonnée complète reste un segment."""
        texte = ("Quand le fournisseur change de recette, "
                 "la fiche technique est mise à jour.")
        self.assertEqual(len(app.segment_text(texte)), 2)

    def test_le_mecanisme_est_inerte_sur_le_banc_et_sur_le_lot_1(self):
        total = 0
        for dossier in ("cas", "narratif"):
            for chemin in sorted((RACINE / "data" / dossier).glob("*.txt")):
                total += len(app.segment_text(
                    chemin.read_text(encoding="utf-8")))
        self.assertEqual(total, 509)
        premier = sum(len(app.segment_text(p.read_text(encoding="utf-8")))
                      for p in sorted(JEU.glob("*.txt")))
        # Le classificateur grammatical de virgules, ajouté plus tard et
        # indépendant de ce mécanisme, réunit dans hs08 les deux compléments
        # qui appartiennent au même empan de référence : 31 -> 30.
        self.assertEqual(premier, 30)


class StructuresDuLotDeDeveloppement(unittest.TestCase):
    """Régressions grammaticales révélées par le lot 21-30.

    Ce lot n'est plus une mesure hors échantillon dès lors que ces tests en
    dérivent. Le lot 31-40 reste scellé jusqu'à la fin de ces corrections.
    """

    def _fragments(self, texte):
        return [texte[a:b] for a, b in app.segment_text(texte)]

    def test_une_liste_interne_ne_devient_pas_une_chaine_de_regles(self):
        texte = ("Lorsque l'avis confirme que le projet de code, la modification "
                 "ou la prorogation respecte le règlement ou, dans la situation "
                 "prévue, offre des garanties, le comité rend son avis.")
        fragments = self._fragments(texte)
        self.assertEqual(len(fragments), 2, fragments)
        self.assertTrue(fragments[0].startswith("Lorsque"))
        self.assertTrue(fragments[1].startswith("le comité"))

    def test_les_complements_entre_virgules_restent_dans_la_conclusion(self):
        texte = ("Lorsqu'un État institue plusieurs autorités, il définit le "
                 "mécanisme de contrôle, par les autres autorités, des règles.")
        fragments = self._fragments(texte)
        self.assertEqual(len(fragments), 2, fragments)
        self.assertIn("par les autres autorités", fragments[1])

    def test_y_compris_lorsque_reste_une_extension_de_portee(self):
        texte = ("L'autorité dispose des ressources nécessaires, y compris "
                 "lorsqu'elle agit avec ses homologues.")
        fragments = self._fragments(texte)
        self.assertEqual(len(fragments), 1, fragments)

    def test_les_conditions_nominales_suffixees_sont_separees(self):
        texte = ("Les fonctions prennent fin à l'échéance du mandat, en cas de "
                 "démission.")
        fragments = self._fragments(texte)
        self.assertEqual(fragments, [
            "Les fonctions prennent fin",
            "à l'échéance du mandat",
            "en cas de démission",
        ])

    def test_la_condition_a_la_condition_que_est_separee(self):
        texte = ("Le transfert est permis que s'il existe des garanties "
                 "appropriées et à la condition que les recours soient effectifs.")
        fragments = self._fragments(texte)
        self.assertEqual(len(fragments), 3, fragments)
        self.assertIn("garanties appropriées", fragments[1])
        self.assertIn("recours soient effectifs", fragments[2])

    def test_ou_si_est_une_alternative_et_non_une_conjonction(self):
        texte = "La licence est retirée si A est faux ou si B est faux."
        segments = app.segment_text(texte)
        alternatives = app._alternatives_coordonnees(texte, segments)
        self.assertEqual(alternatives, {(2, 3)})
        self.assertEqual(
            app._variantes_conditions([1, 2, 3], alternatives),
            [[1, 2], [1, 3]],
        )

    def test_et_que_reste_une_conjonction(self):
        texte = "La licence est retirée si A est faux et que B est faux."
        segments = app.segment_text(texte)
        self.assertEqual(app._alternatives_coordonnees(texte, segments), set())
        self.assertEqual(app._variantes_conditions([2, 3], set()), [[2, 3]])


if __name__ == "__main__":
    unittest.main()
