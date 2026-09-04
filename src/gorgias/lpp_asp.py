"""Traduction LPP/GORGIAS vers ASP, et comparaison par modèles stables.

Pourquoi ce module
------------------
Le scoreur historique apparie des STRUCTURES : une règle produite compte juste
si son effet et l'ensemble de ses conditions correspondent au gold. Deux biais
opposés en découlent, tous deux non quantifiés jusqu'ici :

  - deux programmes logiquement équivalents mais écrits autrement sont comptés
    comme un échec ;
  - une règle bien formée qui ne conclut rien est comptée juste.

Avec 43 règles justes sur 101, on ignore si le F1 sous-estime ou surestime la
qualité réelle. Comparer les CONSÉQUENCES plutôt que la forme lève ce doute.

Encodage retenu
---------------
Chaque entité brat devient un atome, chaque événement une règle ASP ::

    actif(t1).                       % contexte donné en fait initial
    appl(e1) :- actif(t1), actif(t3).% la règle s'applique si TOUTES ses
                                     % conditions sont actives
    tete(e1, t2).                    % sa conclusion
    pref(e1, e2, t5).                % e1 bat e2 quand t5 est actif
    vaincu(L) :- pref(W, L, C), actif(C), appl(W), appl(L), not neutralisee(...).
    conclut(O) :- appl(R), tete(R, O), not vaincu(R).

La défaite est exprimée par négation par échec. GORGIAS repose sur LPwNF, mais
l'objet ici n'est pas de reproduire son moteur : c'est de comparer deux
extractions sous une sémantique unique et explicite. Les deux côtés — référence
et production — subissent exactement le même encodage, donc tout biais de
l'encodage s'annule dans la comparaison.

Portée de ce que le score affirme
---------------------------------
Injecter des faits et comparer les modèles teste l'équivalence UNIFORME sur les
contextes essayés, et non l'équivalence FORTE au sens de Lifschitz, Pearce et
Valverde — laquelle exige l'équivalence sous TOUT programme ajouté et se teste
en logique Ici-et-Là. Quand le document compte au plus douze contextes — la quasi-totalité du banc —
TOUS les sous-ensembles sont essayés : l'équivalence uniforme sur les faits de
contexte est alors DÉMONTRÉE et non sondée, et le régime est rapporté dans le
JSON. Cela reste plus faible que l'équivalence forte, qui quantifie sur tout
programme ajoutable et non sur les seuls faits ; mais c'est la notion utile
ici, puisque la seule chose qui varie d'un déploiement à l'autre est
l'ensemble des contextes qui se réalisent.
"""
from __future__ import annotations

import itertools
import re

_LIGNE_T = re.compile(r"^(T\d+)\t(\w+) (\d+) (\d+)\t(.*)$")
_LIGNE_E = re.compile(r"^(E\d+)\t(\w+):(\w+)(.*)$")
_ROLE = re.compile(r"(\w+):(\w+)")


def _atome(identifiant: str) -> str:
    """T12 -> t12, E3 -> e3 : les constantes ASP commencent en minuscule."""
    return identifiant.lower()


def analyser(annotations: str) -> dict:
    """Décompose une annotation brat en entités et événements."""
    entites: dict[str, str] = {}
    evenements: dict[str, dict] = {}
    for ligne in annotations.splitlines():
        correspondance = _LIGNE_T.match(ligne)
        if correspondance:
            entites[correspondance.group(1)] = correspondance.group(2)
            continue
        correspondance = _LIGNE_E.match(ligne)
        if correspondance:
            identifiant, genre = correspondance.group(1), correspondance.group(2)
            roles: dict[str, list[str]] = {}
            for role, cible in _ROLE.findall(correspondance.group(4)):
                roles.setdefault(role, []).append(cible)
            evenements[identifiant] = {"genre": genre, "roles": roles}
    return {"entites": entites, "evenements": evenements}


def contextes(annotations: str) -> list[str]:
    """Entités de type Context : ce sont les faits que l'on peut activer."""
    analyse = analyser(annotations)
    return sorted(t for t, genre in analyse["entites"].items()
                  if genre == "Context")


def vers_asp(annotations: str, actifs: list[str]) -> str:
    """Programme ASP correspondant à l'annotation, sous les contextes `actifs`."""
    analyse = analyser(annotations)
    evenements = analyse["evenements"]
    lignes = ["% --- contextes activés ---"]
    lignes += [f"actif({_atome(t)})." for t in sorted(actifs)]

    lignes.append("% --- règles d'objet ---")
    for identifiant, evenement in sorted(evenements.items()):
        if evenement["genre"] != "rule":
            continue
        conditions = evenement["roles"].get("Condition", [])
        effets = evenement["roles"].get("Effect", [])
        if not effets:
            continue
        corps = ", ".join(f"actif({_atome(c)})" for c in conditions)
        tete = _atome(identifiant)
        # Une règle sans condition s'applique inconditionnellement.
        lignes.append(f"appl({tete}) :- {corps}." if corps else f"appl({tete}).")
        lignes.append(f"tete({tete}, {_atome(effets[0])}).")

    lignes.append("% --- priorités ---")
    for _, evenement in sorted(evenements.items()):
        if evenement["genre"] not in ("prefer", "meta_prefer"):
            continue
        gagnantes = evenement["roles"].get("Winner", [])
        perdantes = evenement["roles"].get("Loser", [])
        quand = evenement["roles"].get("When", [])
        if not gagnantes or not perdantes:
            continue
        contexte = _atome(quand[0]) if quand else "toujours"
        predicat = "pref" if evenement["genre"] == "prefer" else "metapref"
        lignes.append(
            f"{predicat}({_atome(gagnantes[0])}, {_atome(perdantes[0])}, {contexte})."
        )

    lignes += [
        "% --- sémantique ---",
        "actif(toujours).",
        # Déclare les prédicats éventuellement absents : sans cela clingo
        # signale « atom does not occur in any rule head » sur les
        # documents sans méta-priorité, ce qui est un cas normal.
        "#defined metapref/3.",
        "#defined pref/3.",
        "#defined appl/1.",
        "#defined tete/2.",
        # Une méta-priorité neutralise la priorité qu'elle bat, quand son
        # contexte est actif : c'est le niveau 2 du formalisme.
        "neutralisee(P) :- metapref(_, P, C), actif(C).",
        # Une règle est vaincue si une règle concurrente applicable lui est
        # préférée dans un contexte actif, et que cette préférence n'a pas été
        # elle-même neutralisée.
        "vaincu(L) :- pref(W, L, C), actif(C), appl(W), appl(L), "
        "not neutralisee(W).",
        "conclut(O) :- appl(R), tete(R, O), not vaincu(R).",
        "#show conclut/1.",
    ]
    return "\n".join(lignes)


def modeles_stables(programme: str) -> list[frozenset[str]]:
    """Answer sets du programme, sous forme d'ensembles d'atomes affichés."""
    import clingo

    modeles: list[frozenset[str]] = []
    controle = clingo.Control(["0"])  # 0 = énumérer TOUS les modèles
    controle.add("base", [], programme)
    controle.ground([("base", [])])
    controle.solve(on_model=lambda m: modeles.append(
        frozenset(str(atome) for atome in m.symbols(shown=True))))
    return modeles


def _prudentes(modeles) -> frozenset[str]:
    """Conséquences prudentes : intersection de tous les answer sets."""
    if not modeles:
        return frozenset()
    return frozenset.intersection(*modeles)


BUDGET_EXHAUSTIF = 4096   # 2**12 sous-ensembles : quelques secondes de clingo


def echantillon_contextes(annotations: str, maximum: int = 8,
                          graine: int = 20260804) -> tuple[list[list[str]], str]:
    """Sous-ensembles de contextes à tester, et le régime employé.

    Quand le document compte peu de contextes, TOUS les sous-ensembles sont
    essayés : la comparaison devient alors exhaustive sur les faits de
    contexte, et non plus un sondage. C'est le régime « exhaustif ». Au-delà du
    budget, on retombe sur un tirage à graine fixe, régime « échantillonné ».

    La consigne demandait des faits initiaux ALÉATOIRES. Tirés dans l'absolu,
    ils n'activeraient presque jamais de règle : les deux programmes rendraient
    des modèles vides et l'accord serait trivialement parfait — on mesurerait
    l'inactivité, pas l'équivalence. Le tirage se fait donc parmi les CONTEXTES
    DU DOCUMENT, et l'ensemble complet est toujours inclus pour garantir qu'au
    moins un essai active ce qui peut l'être. La graine est fixe : deux
    exécutions comparent les mêmes contextes.
    """
    import random

    disponibles = contextes(annotations)
    if not disponibles:
        return [[]], "exhaustif"
    if 2 ** len(disponibles) <= BUDGET_EXHAUSTIF:
        # Tous les sous-ensembles : l'équivalence uniforme sur les faits de
        # contexte est alors DÉMONTRÉE, pas sondée.
        essais = [list(c) for taille in range(len(disponibles) + 1)
                  for c in itertools.combinations(disponibles, taille)]
        regime = "exhaustif"
    else:
        essais = [list(disponibles)]
        alea = random.Random(graine)
        for _ in range(maximum - 1):
            taille = alea.randint(1, len(disponibles))
            essais.append(sorted(alea.sample(disponibles, taille)))
        regime = "echantillonne"
    # Dédoublonnage en conservant l'ordre.
    vus, uniques = set(), []
    for essai in essais:
        cle = tuple(essai)
        if cle not in vus:
            vus.add(cle)
            uniques.append(essai)
    return (uniques if regime == "exhaustif" else uniques[:maximum]), regime


def comparer(reference: str, produit: str, maximum: int = 8) -> dict:
    """Compare deux annotations par leurs conséquences, contexte par contexte.

    Les contextes d'essai sont dérivés de la RÉFÉRENCE : c'est elle qui définit
    les situations pertinentes. Les identifiants d'entités étant des positions
    (T1, T2…), les deux annotations partagent le même vocabulaire dès lors que
    leurs empans coïncident.
    """
    essais, regime = echantillon_contextes(reference, maximum)
    accords_prudents = accords_exacts = 0
    attendus_total = produits_total = communs_total = 0
    details = []
    for actifs in essais:
        modeles_ref = modeles_stables(vers_asp(reference, actifs))
        modeles_pro = modeles_stables(vers_asp(produit, actifs))
        prudentes_ref, prudentes_pro = _prudentes(modeles_ref), _prudentes(modeles_pro)
        communs = prudentes_ref & prudentes_pro
        accords_prudents += int(prudentes_ref == prudentes_pro)
        accords_exacts += int(set(modeles_ref) == set(modeles_pro))
        attendus_total += len(prudentes_ref)
        produits_total += len(prudentes_pro)
        communs_total += len(communs)
        details.append({
            "contextes": actifs,
            "conclusions_reference": sorted(prudentes_ref),
            "conclusions_produites": sorted(prudentes_pro),
            "identiques": prudentes_ref == prudentes_pro,
        })
    precision = communs_total / produits_total if produits_total else 0.0
    rappel = communs_total / attendus_total if attendus_total else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    return {
        "regime": regime,
        "essais": len(essais),
        "accords_conclusions": accords_prudents,
        "accords_answer_sets": accords_exacts,
        "precision": round(precision, 4),
        "rappel": round(rappel, 4),
        "f1": round(f1, 4),
        "conclusions_attendues": attendus_total,
        "conclusions_produites": produits_total,
        "conclusions_communes": communs_total,
        "details": details,
    }
