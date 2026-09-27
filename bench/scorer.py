"""Scoreur du banc : entités, règles, préférences, méta-préférences.

Appariement en deux temps, comme le veut la structure brat : les entités
d'abord (par recouvrement de leurs empans), les événements ensuite (par
identité de leurs rôles, une fois ceux-ci traduits par l'appariement des
entités). Un événement ne peut donc être juste que si TOUTES les entités
qu'il désigne le sont — ce qui est le comportement voulu : une règle ancrée
sur le mauvais contexte est fausse, pas à moitié juste.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LIGNE_T = re.compile(r"^(T\d+)\t(Context|Option|Marker) (\d+) (\d+)\t(.*)$")
_LIGNE_E = re.compile(r"^(E\d+)\t(rule|prefer|meta_prefer):(\S+)(.*)$")
_ROLE = re.compile(r"([A-Za-z]+):([TE]\d+)")

# Un empan produit ne coïncide pas au caractère près avec l'empan de
# référence : la segmentation coupe sur la ponctuation, l'annotateur humain
# sur le groupe syntaxique. On apparie donc par recouvrement (Jaccard sur les
# intervalles de caractères), avec un seuil qui interdit les appariements de
# complaisance.
SEUIL_RECOUVREMENT = 0.5


@dataclass
class Analyse:
    entites: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    evenements: dict[str, tuple[str, dict[str, list[str]]]] = field(default_factory=dict)


def analyser(annotations: str) -> Analyse:
    analyse = Analyse()
    for ligne in (annotations or "").splitlines():
        trouve = _LIGNE_T.match(ligne)
        if trouve:
            analyse.entites[trouve.group(1)] = (
                trouve.group(2), int(trouve.group(3)), int(trouve.group(4))
            )
            continue
        trouve = _LIGNE_E.match(ligne)
        if trouve:
            roles: dict[str, list[str]] = {}
            for role, cible in _ROLE.findall(trouve.group(4)):
                roles.setdefault(role, []).append(cible)
            analyse.evenements[trouve.group(1)] = (trouve.group(2), roles)
    return analyse


def _recouvrement(a: tuple[int, int], b: tuple[int, int]) -> float:
    commun = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if not commun:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return commun / union if union else 0.0


def apparier_entites(reference: Analyse, produit: Analyse,
                     strict: bool = False) -> dict[str, str]:
    """{identifiant produit -> identifiant de référence}, appariement glouton.

    Glouton par recouvrement décroissant : c'est stable, et l'optimum de
    couplage ne changerait rien à ces échelles (les empans d'un même type ne
    se chevauchent pratiquement jamais entre eux).
    """
    couples = []
    for ip, (tp, dp, fp) in produit.entites.items():
        for ir, (tr, dr, fr) in reference.entites.items():
            if tp != tr:
                continue
            if strict:
                score = 1.0 if (dp, fp) == (dr, fr) else 0.0
            else:
                score = _recouvrement((dp, fp), (dr, fr))
            if score >= (1.0 if strict else SEUIL_RECOUVREMENT):
                couples.append((score, ip, ir))
    couples.sort(key=lambda c: (-c[0], c[1], c[2]))
    pris_p: set[str] = set()
    pris_r: set[str] = set()
    liaison: dict[str, str] = {}
    for _, ip, ir in couples:
        if ip in pris_p or ir in pris_r:
            continue
        liaison[ip] = ir
        pris_p.add(ip)
        pris_r.add(ir)
    return liaison


def _signature_regle(identifiant, analyse: Analyse, traduire) -> tuple | None:
    genre, roles = analyse.evenements[identifiant]
    if genre != "rule":
        return None
    effets = [traduire(t) for t in roles.get("Effect", [])]
    conditions = [traduire(t) for t in roles.get("Condition", [])]
    if not effets or effets[0] is None or any(c is None for c in conditions):
        return None
    return ("rule", effets[0], frozenset(conditions))


def _signature_priorite(identifiant, analyse: Analyse, traduire,
                        signature_evenement, avec_quand=True) -> tuple | None:
    genre, roles = analyse.evenements[identifiant]
    if genre not in ("prefer", "meta_prefer"):
        return None
    gagnantes = [signature_evenement(e) for e in roles.get("Winner", [])]
    perdantes = [signature_evenement(e) for e in roles.get("Loser", [])]
    if not gagnantes or not perdantes or gagnantes[0] is None or perdantes[0] is None:
        return None
    quand = frozenset(
        t for t in (traduire(c) for c in roles.get("When", [])) if t is not None
    ) if avec_quand else frozenset()
    if avec_quand and len(quand) != len(roles.get("When", [])):
        return None
    return (genre, gagnantes[0], perdantes[0], quand)


def _signatures(analyse: Analyse, traduire, avec_quand=True) -> dict[str, tuple]:
    """Signature canonique de chaque événement, résolue récursivement.

    Une méta-préférence désigne des préférences, qui désignent des règles :
    la signature se calcule donc de bas en haut, avec mémoïsation.
    """
    cache: dict[str, tuple | None] = {}

    def signature(identifiant: str) -> tuple | None:
        if identifiant in cache:
            return cache[identifiant]
        cache[identifiant] = None          # coupe les cycles éventuels
        if identifiant not in analyse.evenements:
            return None
        genre = analyse.evenements[identifiant][0]
        valeur = (
            _signature_regle(identifiant, analyse, traduire)
            if genre == "rule"
            else _signature_priorite(identifiant, analyse, traduire,
                                     signature, avec_quand)
        )
        cache[identifiant] = valeur
        return valeur

    return {i: s for i in analyse.evenements if (s := signature(i)) is not None}


def _compter(references: list[tuple], produits: list[tuple]) -> tuple[int, int, int]:
    """(justes, produits, attendus) avec multiplicité — un sac, pas un ensemble."""
    restants = list(references)
    justes = 0
    for signature in produits:
        if signature in restants:
            restants.remove(signature)
            justes += 1
    return justes, len(produits), len(references)


def scorer_document(reference: str, produit: str, strict: bool = False) -> dict:
    ref, pro = analyser(reference), analyser(produit)
    liaison = apparier_entites(ref, pro, strict=strict)

    entites: dict[str, dict] = {}
    for genre in ("Context", "Option", "Marker"):
        attendus = [i for i, (t, *_) in ref.entites.items() if t == genre]
        produits = [i for i, (t, *_) in pro.entites.items() if t == genre]
        justes = sum(1 for i in produits if i in liaison
                     and ref.entites[liaison[i]][0] == genre)
        entites[genre] = {"justes": justes, "produits": len(produits),
                          "attendus": len(attendus)}

    signatures_ref = _signatures(ref, lambda t: t)
    signatures_pro = _signatures(pro, lambda t: liaison.get(t))
    # Sans le contexte d'ancrage : isole ce que coûte le seul ancrage.
    laxes_ref = _signatures(ref, lambda t: t, avec_quand=False)
    laxes_pro = _signatures(pro, lambda t: liaison.get(t), avec_quand=False)

    def sac(signatures, genre):
        return [s for s in signatures.values() if s[0] == genre]

    resultat = {"entites": entites}
    for genre in ("rule", "prefer", "meta_prefer"):
        justes, produits, attendus = _compter(
            sac(signatures_ref, genre), sac(signatures_pro, genre))
        resultat[genre] = {"justes": justes, "produits": produits,
                           "attendus": attendus}
    justes, produits, attendus = _compter(
        sac(laxes_ref, "prefer") + sac(laxes_ref, "meta_prefer"),
        sac(laxes_pro, "prefer") + sac(laxes_pro, "meta_prefer"))
    resultat["priorite_sans_ancrage"] = {
        "justes": justes, "produits": produits, "attendus": attendus}
    justes, produits, attendus = _compter(
        sac(signatures_ref, "prefer") + sac(signatures_ref, "meta_prefer"),
        sac(signatures_pro, "prefer") + sac(signatures_pro, "meta_prefer"))
    resultat["priorite"] = {"justes": justes, "produits": produits,
                            "attendus": attendus}
    return resultat


def _pr(justes: int, produits: int, attendus: int) -> dict:
    precision = justes / produits if produits else (1.0 if not attendus else 0.0)
    rappel = justes / attendus if attendus else 1.0
    f1 = (2 * precision * rappel / (precision + rappel)
          if precision + rappel else 0.0)
    return {"precision": precision, "rappel": rappel, "f1": f1,
            "justes": justes, "produits": produits, "attendus": attendus}


CLES = ("rule", "prefer", "meta_prefer", "priorite", "priorite_sans_ancrage")


def agreger(scores: list[dict]) -> dict:
    total: dict[str, dict] = {}
    for cle in CLES:
        total[cle] = _pr(*(sum(s[cle][champ] for s in scores)
                           for champ in ("justes", "produits", "attendus")))
    for genre in ("Context", "Option", "Marker"):
        total[genre] = _pr(*(sum(s["entites"][genre][champ] for s in scores)
                             for champ in ("justes", "produits", "attendus")))
    total["entites"] = _pr(*(
        sum(s["entites"][g][champ] for s in scores
            for g in ("Context", "Option", "Marker"))
        for champ in ("justes", "produits", "attendus")))
    return total
