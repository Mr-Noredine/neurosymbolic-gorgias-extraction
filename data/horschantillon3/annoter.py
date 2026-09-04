"""Annotation de référence du TROISIÈME lot hors échantillon (RGPD, 21-30).

Même raison que pour le second lot : le lot 11-20 a servi à dériver un
mécanisme — le subordonnant rendu à sa proposition — et ne mesure donc plus la
généralisation. Ce lot-ci reprend la même règle mécanique au rang suivant,
`selectionner.py --debut 21`, et a été annoté AVANT toute exécution du
système sur ces textes.

Conventions, identiques aux deux premiers lots :
  - le `Context` exclut le subordonnant, l'`Option` porte la conclusion ;
  - les empans sont localisés par recherche de chaîne EXACTE ;
  - le fragment écrit dans le `.ann` est celui du FICHIER, espaces insécables
    comprises ;
  - une clause d'EXCEPTION n'est pas annotée (convention relevée sur
    `48-drones-autonomes`) ;
  - des conditions ALTERNATIVES (« si A ou si B ») donnent DEUX règles de même
    conclusion, la disjonction n'ayant pas de forme propre dans le schéma.

`hs28` reste SANS annotation : sa seule subordonnée est une extension de
portée (« y compris lorsque celle-ci doit agir… »), pas une condition
déclenchante — la principale, elle, est inconditionnelle. C'est donc, comme
`hs11` du lot précédent, une sonde de précision. Décidé avant la mesure.

Annotateur unique, comme le reste du banc.
"""
from __future__ import annotations

import pathlib

ANNOTATIONS = {
"hs21": ([
    ("T1", "Context", "le projet de code, la modification ou la prorogation est approuvé conformément au paragraphe 5"),
    ("T2", "Context", "le code de conduite concerné ne porte pas sur des activités de traitement menées dans plusieurs États membres"),
    ("T3", "Option",  "l'autorité de contrôle enregistre et publie le code de conduite"),
], [("E1", "rule:T3 Condition:T1 Condition:T2 Effect:T3")]),

"hs22": ([
    ("T1", "Context", "l'avis visé au paragraphe 7 confirme que le projet de code, la modification ou la prorogation respecte le présent règlement ou, dans la situation visée au paragraphe 3, offre des garanties appropriées"),
    ("T2", "Option",  "le comité soumet son avis à la Commission"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),

"hs23": ([
    ("T1", "Option",  "L'autorité de contrôle compétente révoque l'agrément d'un organisme visé au paragraphe 1"),
    ("T2", "Context", "les conditions d'agrément ne sont pas ou ne sont plus réunies"),
    ("T3", "Context", "les mesures prises par l'organisme constituent une violation du présent règlement"),
], [("E1", "rule:T1 Condition:T2 Effect:T1"),
    ("E2", "rule:T1 Condition:T3 Effect:T1")]),

"hs24": ([
    ("T1", "Option",  "l'autorité de contrôle compétente ou l'organisme national d'accréditation révoque l'agrément d'un organisme de certification en application du paragraphe 1 du présent article"),
    ("T2", "Context", "les conditions d'agrément ne sont pas ou ne sont plus réunies"),
    ("T3", "Context", "les mesures prises par l'organisme de certification constituent une violation du présent règlement"),
], [("E1", "rule:T1 Condition:T2 Effect:T1"),
    ("E2", "rule:T1 Condition:T3 Effect:T1")]),

"hs25": ([
    ("T1", "Option",  "Un transfert de données à caractère personnel vers un pays tiers ou à une organisation internationale peut avoir lieu"),
    ("T2", "Context", "la Commission a constaté par voie de décision que le pays tiers, un territoire ou un ou plusieurs secteurs déterminés dans ce pays tiers, ou l'organisation internationale en question assure un niveau de protection adéquat"),
    ("T3", "Option",  "Un tel transfert ne nécessite pas d'autorisation spécifique"),
], [("E1", "rule:T1 Condition:T2 Effect:T1"),
    ("E2", "rule:T3 Condition:T2 Effect:T3")]),

"hs26": ([
    ("T1", "Context", "En l'absence de décision en vertu de l'article 45, paragraphe 3"),
    ("T2", "Option",  "le responsable du traitement ou le sous-traitant ne peut transférer des données à caractère personnel vers un pays tiers ou à une organisation internationale"),
    ("T3", "Context", "il a prévu des garanties appropriées"),
    ("T4", "Context", "les personnes concernées disposent de droits opposables et de voies de droit effectives"),
], [("E1", "rule:T2 Condition:T1 Condition:T3 Condition:T4 Effect:T2")]),

"hs27": ([
    ("T1", "Context", "un État membre institue plusieurs autorités de contrôle"),
    ("T2", "Option",  "il désigne celle qui représente ces autorités au comité et définit le mécanisme permettant de s'assurer du respect, par les autres autorités, des règles relatives au mécanisme de contrôle de la cohérence visé à l'article 63"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),

"hs28": ([], []),

"hs29": ([
    ("T1", "Option",  "Les fonctions d'un membre prennent fin"),
    ("T2", "Context", "à l'échéance de son mandat"),
    ("T3", "Context", "en cas de démission ou de mise à la retraite d'office"),
], [("E1", "rule:T1 Condition:T2 Effect:T1"),
    ("E2", "rule:T1 Condition:T3 Effect:T1")]),

"hs30": ([
    ("T1", "Option",  "Un membre ne peut être démis de ses fonctions"),
    ("T2", "Context", "il a commis une faute grave"),
    ("T3", "Context", "il ne remplit plus les conditions nécessaires à l'exercice de ses fonctions"),
], [("E1", "rule:T1 Condition:T2 Effect:T1"),
    ("E2", "rule:T1 Condition:T3 Effect:T1")]),
}

racine = pathlib.Path(__file__).parent
erreurs = 0
for nom, (entites, evenements) in ANNOTATIONS.items():
    texte = (racine / f"{nom}.txt").read_text(encoding="utf-8")
    cherchable = texte.replace("\u00a0", " ")
    lignes, bornes = [], {}
    for identifiant, genre, fragment in entites:
        position = cherchable.find(fragment.replace("\u00a0", " "))
        if position < 0:
            print(f"  {nom} : INTROUVABLE {fragment[:56]!r}")
            erreurs += 1
            continue
        debut, fin = position, position + len(fragment)
        bornes[identifiant] = (debut, fin)
        lignes.append(f"{identifiant}\t{genre} {debut} {fin}\t{texte[debut:fin]}")
    for identifiant, corps in evenements:
        lignes.append(f"{identifiant}\t{corps}")
    (racine / f"{nom}.ann").write_text(
        ("\n".join(lignes) + "\n") if lignes else "", encoding="utf-8")
    for identifiant, (debut, fin) in bornes.items():
        attendu = next(f for i, _, f in entites if i == identifiant)
        if texte[debut:fin].replace("\u00a0", " ") != attendu.replace("\u00a0", " "):
            print(f"  {nom} : OFFSET FAUX {identifiant}")
            erreurs += 1
    ordonnes = sorted(bornes.items(), key=lambda kv: kv[1])
    for (a, sa), (b, sb) in zip(ordonnes, ordonnes[1:]):
        if sa[1] > sb[0]:
            print(f"  {nom} : CHEVAUCHEMENT {a}/{b} — {sa} et {sb}")
            erreurs += 1

regles = sum(len(e) for _, e in ANNOTATIONS.values())
entites_tot = sum(len(t) for t, _ in ANNOTATIONS.values())
print(f"{len(ANNOTATIONS)} documents, {entites_tot} entités, {regles} règles")
print(f"erreurs : {erreurs}")
