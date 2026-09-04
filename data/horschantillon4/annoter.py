"""Référence gelée du lot RGPD 31–40, écrite avant toute mesure système.

Conventions identiques aux lots précédents : alternatives conditionnelles en
règles séparées, reprises démonstratives rattachées au contexte explicite,
incises et relatives descriptives non transformées en règles.
"""
from __future__ import annotations

import pathlib


ANNOTATIONS = {
    "hs31": ([
        ("T1", "Context", "le traitement est effectué par des autorités publiques ou des organismes privés agissant sur la base de l'article 6, paragraphe 1, point c) ou e)"),
        ("T2", "Option", "l'autorité de contrôle de l'État membre concerné est compétente"),
        ("T3", "Option", "l'article 56 n'est pas applicable"),
    ], [
        ("E1", "rule:T2 Condition:T1 Effect:T2"),
        ("E2", "rule:T3 Condition:T1 Effect:T3"),
    ]),
    "hs32": ([
        ("T1", "Option", "chaque autorité de contrôle est compétente pour traiter une réclamation introduite auprès d'elle ou une éventuelle violation du présent règlement"),
        ("T2", "Context", "son objet concerne uniquement un établissement dans l'État membre dont elle relève"),
        ("T3", "Context", "affecte sensiblement des personnes concernées dans cet État membre uniquement"),
    ], [
        ("E1", "rule:T1 Condition:T2 Effect:T1"),
        ("E2", "rule:T1 Condition:T3 Effect:T1"),
    ]),
    "hs33": ([
        ("T1", "Context", "l'autorité de contrôle chef de file décide de traiter le cas"),
        ("T2", "Option", "la procédure prévue à l'article 60 s'applique"),
        ("T3", "Option", "L'autorité de contrôle qui a informé l'autorité de contrôle chef de file peut lui soumettre un projet de décision"),
        ("T4", "Context", "elle élabore le projet de décision visé à l'article 60, paragraphe 3"),
        ("T5", "Option", "L'autorité de contrôle chef de file tient le plus grand compte de ce projet"),
    ], [
        ("E1", "rule:T2 Condition:T1 Effect:T2"),
        ("E2", "rule:T3 Condition:T1 Effect:T3"),
        ("E3", "rule:T5 Condition:T1 Condition:T4 Effect:T5"),
    ]),
    "hs34": ([
        ("T1", "Context", "l'autorité de contrôle chef de file décide de ne pas traiter le cas"),
        ("T2", "Option", "l'autorité de contrôle qui l'a informée le traite conformément aux articles 61 et 62"),
    ], [("E1", "rule:T2 Condition:T1 Effect:T2")]),
    "hs35": ([
        ("T1", "Context", "les demandes sont manifestement infondées ou excessives, en raison, notamment, de leur caractère répétitif"),
        ("T2", "Option", "l'autorité de contrôle peut exiger le paiement de frais raisonnables basés sur les coûts administratifs"),
        ("T3", "Option", "refuser de donner suite à la demande"),
        ("T4", "Option", "Il incombe à l'autorité de contrôle de démontrer le caractère manifestement infondé ou excessif de la demande"),
    ], [
        ("E1", "rule:T2 Condition:T1 Effect:T2"),
        ("E2", "rule:T3 Condition:T1 Effect:T3"),
        ("E3", "rule:T4 Condition:T1 Effect:T4"),
    ]),
    "hs36": ([
        ("T1", "Context", "l'autorité de contrôle chef de file entend suivre l'objection pertinente et motivée formulée"),
        ("T2", "Option", "elle soumet aux autres autorités de contrôle concernées un projet de décision révisé en vue d'obtenir leur avis"),
        ("T3", "Option", "Ce projet de décision révisé est soumis à la procédure visée au paragraphe 4 dans un délai de deux semaines"),
    ], [
        ("E1", "rule:T2 Condition:T1 Effect:T2"),
        ("E2", "rule:T3 Condition:T1 Effect:T3"),
    ]),
    "hs37": ([
        ("T1", "Context", "aucune des autres autorités de contrôle concernées n'a formulé d'objection à l'égard du projet de décision soumis par l'autorité de contrôle chef de file dans le délai visé aux paragraphes 4 et 5"),
        ("T2", "Option", "l'autorité de contrôle chef de file et les autorités de contrôle concernées sont réputées approuver ce projet de décision et sont liées par lui"),
    ], [("E1", "rule:T2 Condition:T1 Effect:T2")]),
    "hs38": ([
        ("T1", "Context", "une réclamation est refusée ou rejetée"),
        ("T2", "Option", "l'autorité de contrôle auprès de laquelle la réclamation a été introduite adopte la décision, la notifie à l'auteur de la réclamation et en informe le responsable du traitement"),
    ], [("E1", "rule:T2 Condition:T1 Effect:T2")]),
    "hs39": ([
        ("T1", "Context", "dans des circonstances exceptionnelles, une autorité de contrôle concernée a des raisons de considérer qu'il est urgent d'intervenir pour protéger les intérêts des personnes concernées"),
        ("T2", "Option", "la procédure d'urgence visée à l'article 66 s'applique"),
    ], [("E1", "rule:T2 Condition:T1 Effect:T2")]),
    "hs40": ([
        ("T1", "Context", "conformément au paragraphe 1, les agents de l'autorité de contrôle d'origine opèrent dans un autre État membre"),
        ("T2", "Option", "l'État membre dont relève l'autorité de contrôle d'accueil assume la responsabilité de leurs actions, y compris la responsabilité des dommages qu'ils causent au cours des opérations dont ils sont chargés, conformément au droit de l'État membre sur le territoire duquel ils opèrent"),
    ], [("E1", "rule:T2 Condition:T1 Effect:T2")]),
}


racine = pathlib.Path(__file__).parent
erreurs = 0
for nom, (entites, evenements) in ANNOTATIONS.items():
    texte = (racine / f"{nom}.txt").read_text(encoding="utf-8")
    cherchable = texte.replace("\u00a0", " ")
    lignes = []
    for identifiant, genre, fragment in entites:
        position = cherchable.find(fragment.replace("\u00a0", " "))
        if position < 0:
            print(f"{nom}: INTROUVABLE {fragment!r}")
            erreurs += 1
            continue
        fin = position + len(fragment)
        lignes.append(f"{identifiant}\t{genre} {position} {fin}\t{texte[position:fin]}")
    lignes.extend(f"{identifiant}\t{corps}" for identifiant, corps in evenements)
    (racine / f"{nom}.ann").write_text("\n".join(lignes) + "\n", encoding="utf-8")

print(f"{len(ANNOTATIONS)} documents, "
      f"{sum(len(e) for e, _ in ANNOTATIONS.values())} entités, "
      f"{sum(len(r) for _, r in ANNOTATIONS.values())} règles")
print(f"erreurs : {erreurs}")
