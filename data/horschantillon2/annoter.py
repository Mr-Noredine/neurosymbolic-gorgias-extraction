"""Annotation de référence du SECOND lot hors échantillon (RGPD, rangs 11-20).

Pourquoi un second lot. Les dix premiers documents ont servi à diagnostiquer
puis à corriger deux défauts réels (numérotation, incises) : ils ne sont donc
PLUS hors échantillon, et les chiffres qu'on y mesure après correctif ne
mesurent plus la généralisation. Ce lot-ci reprend la MÊME règle mécanique au
rang suivant — `selectionner.py --debut 11` — et n'a servi à rien d'autre qu'à
être mesuré.

Conventions, identiques au premier lot :
  - le `Context` exclut le subordonnant (« le sol est gelé », pas « Quand le
    sol est gelé ») ;
  - l'`Option` porte la conclusion, incises comprises, comme le fait le texte ;
  - les empans sont localisés par recherche de chaîne EXACTE — aucun offset
    écrit à la main ;
  - une CLAUSE D'EXCEPTION n'est pas annotée. C'est la convention du banc,
    relevée sur `48-drones-autonomes` : « …ne doit pas être envoyé vers une
    nouvelle zone, SAUF LORSQU'IL constitue le seul moyen disponible… » n'a
    qu'une règle de référence, sans l'exception.

    `hs11` est tout entier de cette forme — une interdiction dont la seule
    clause subordonnée est « à moins que … ne s'applique ». Il reste donc SANS
    aucune annotation, et devient de fait une sonde de précision : tout ce que
    le système y produira sera un faux positif. Cette décision est prise AVANT
    la première mesure et ne sera pas révisée en fonction du score.

Annotateur unique, comme le reste du banc.
"""
from __future__ import annotations

import pathlib

# (document, [(id, type, texte exact)], [(id, corps d'événement)])
ANNOTATIONS = {
"hs11": ([], []),

"hs12": ([
    ("T1", "Context", "cela est proportionné au regard des activités de traitement"),
    ("T2", "Option",  "les mesures visées au paragraphe 1 comprennent la mise en œuvre de politiques appropriées en matière de protection des données par le responsable du traitement"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),

"hs13": ([
    ("T1", "Context", "un traitement doit être effectué pour le compte d'un responsable du traitement"),
    ("T2", "Option",  "celui-ci fait uniquement appel à des sous-traitants qui présentent des garanties suffisantes quant à la mise en œuvre de mesures techniques et organisationnelles appropriées de manière à ce que le traitement réponde aux exigences du présent règlement et garantisse la protection des droits de la personne concernée"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),

"hs14": ([
    ("T1", "Context", "un sous-traitant détermine les finalités et les moyens du traitement"),
    ("T2", "Option",  "il est considéré comme un responsable du traitement pour ce qui concerne ce traitement"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),

"hs15": ([
    ("T1", "Context", "il n'est pas possible de fournir toutes les informations en même temps"),
    ("T2", "Option",  "les informations peuvent être communiquées de manière échelonnée sans autre retard indu"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),

"hs16": ([
    ("T1", "Context", "une violation de données à caractère personnel est susceptible d'engendrer un risque élevé pour les droits et libertés d'une personne physique"),
    ("T2", "Option",  "le responsable du traitement communique la violation de données à caractère personnel à la personne concernée dans les meilleurs délais"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),

"hs17": ([
    ("T1", "Context", "il effectue une analyse d'impact relative à la protection des données"),
    ("T2", "Option",  "le responsable du traitement demande conseil au délégué à la protection des données"),
    ("T3", "Context", "un tel délégué a été désigné"),
], [("E1", "rule:T2 Condition:T1 Condition:T3 Effect:T2")]),

"hs18": ([
    ("T1", "Option",  "le responsable du traitement procède à un examen afin d'évaluer si le traitement est effectué conformément à l'analyse d'impact relative à la protection des données"),
    ("T2", "Context", "il se produit une modification du risque présenté par les opérations de traitement"),
], [("E1", "rule:T1 Condition:T2 Effect:T1")]),

"hs19": ([
    ("T1", "Option",  "Le responsable du traitement consulte l'autorité de contrôle préalablement au traitement"),
    ("T2", "Context", "une analyse d'impact relative à la protection des données effectuée au titre de l'article 35 indique que le traitement présenterait un risque élevé si le responsable du traitement ne prenait pas de mesures pour atténuer le risque"),
], [("E1", "rule:T1 Condition:T2 Effect:T1")]),

"hs20": ([
    ("T1", "Context", "le responsable du traitement ou le sous-traitant est une autorité publique ou un organisme public"),
    ("T2", "Option",  "un seul délégué à la protection des données peut être désigné pour plusieurs autorités ou organismes de ce type, compte tenu de leur structure organisationnelle et de leur taille"),
], [("E1", "rule:T2 Condition:T1 Effect:T2")]),
}

racine = pathlib.Path(__file__).parent
erreurs = 0
for nom, (entites, evenements) in ANNOTATIONS.items():
    texte = (racine / f"{nom}.txt").read_text(encoding="utf-8")
    # EUR-Lex truffe son texte d'espaces INSÉCABLES (U+00A0). La substitution
    # est de longueur égale : on cherche dans la version normalisée et les
    # offsets restent valides sur le texte source, conservé verbatim.
    cherchable = texte.replace(" ", " ")
    lignes, bornes = [], {}
    for identifiant, genre, fragment in entites:
        position = cherchable.find(fragment.replace(" ", " "))
        if position < 0:
            print(f"  {nom} : INTROUVABLE {fragment[:56]!r}")
            erreurs += 1
            continue
        debut, fin = position, position + len(fragment)
        bornes[identifiant] = (debut, fin)
        # Le fragment écrit est celui du FICHIER, espaces insécables comprises,
        # et non ma version normalisée : un `.ann` dont le texte ne correspond
        # pas à ses propres offsets est malformé au sens brat. Le scoreur ne
        # lit que le type et les offsets — la fidélité ne déplace aucun score.
        lignes.append(f"{identifiant}\t{genre} {debut} {fin}\t{texte[debut:fin]}")
    for identifiant, corps in evenements:
        lignes.append(f"{identifiant}\t{corps}")
    (racine / f"{nom}.ann").write_text(
        ("\n".join(lignes) + "\n") if lignes else "", encoding="utf-8")
    # contrôles : offsets exacts, aucun chevauchement
    for identifiant, (debut, fin) in bornes.items():
        attendu = next(f for i, _, f in entites if i == identifiant)
        if texte[debut:fin].replace(" ", " ") != attendu.replace(" ", " "):
            print(f"  {nom} : OFFSET FAUX {identifiant}")
            erreurs += 1
    ordonnes = sorted(bornes.items(), key=lambda kv: kv[1])
    for (a, sa), (b, sb) in zip(ordonnes, ordonnes[1:]):
        if sa[1] > sb[0]:
            print(f"  {nom} : CHEVAUCHEMENT {a}/{b}")
            erreurs += 1

regles = sum(len(e) for _, e in ANNOTATIONS.values())
entites_tot = sum(len(t) for t, _ in ANNOTATIONS.values())
print(f"{len(ANNOTATIONS)} documents, {entites_tot} entités, {regles} règles")
print(f"erreurs : {erreurs}")
