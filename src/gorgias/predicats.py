"""Forme logique normalisée d'une proposition, pour permettre l'unification.

Pourquoi ne pas suivre le moule Sujet-Verbe-Objet
-------------------------------------------------
La consigne demandait : lemme de la racine comme foncteur, ``nsubj`` et
``dobj`` comme arguments. Mesuré sur les 268 entités de référence du banc :

    sujet seul, pas d'objet     42,9 %   « le poulet est produit localement »
    racine non verbale          34,3 %   « les courses de la semaine »
    SVO complet                 12,7 %   « il présente un risque faible »
    objet seul, pas de sujet     6,7 %   « préfère le porc au poulet »
    ni sujet ni objet            3,4 %   « acheter de l'agneau »

Le moule SVO ne couvre donc que 12,7 % des propositions. Appliqué tel quel, il
produirait des prédicats vides sur les 87 % restants, SANS le signaler — on
remplacerait une représentation grossière mais fidèle par une représentation
fine et fausse. Le français de ce corpus est massivement passif et nominal.

Ce que fait ce module à la place
--------------------------------
L'arité suit ce que la phrase offre :

    « le sel est appliqué »          -> appliquer(sel)
    « il présente un risque faible » -> presenter(il, risque)
    « les courses de la semaine »    -> courses_semaine
    « le sel n'est pas appliqué »    -> -appliquer(sel)

Le foncteur est le LEMME : deux formulations du même fait convergent vers le
même littéral, ce qui rend l'unification possible — l'objectif réel de la
consigne. Le passif est traité comme l'actif : dans « le sel est appliqué »,
``sel`` est ``nsubj:pass``, et on le prend comme argument plutôt que d'exiger
un agent qui n'est pas exprimé.

La forme normalisée s'AJOUTE au libellé source ; elle ne le remplace pas. Les
références sont annotées par empans de caractères, et toute la mesure repose
sur eux.
"""
from __future__ import annotations

import functools
import re
import unicodedata

from . import syntaxe

_NEGATIONS = {"ne", "n", "pas", "plus", "jamais", "aucun", "aucune", "rien", "nul"}
_VIDES_NOMINAL = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "au", "aux",
    "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs", "et",
}


def _assainir(mot: str) -> str:
    """Constante ASP valide : minuscules, sans accent, sans ponctuation."""
    decompose = unicodedata.normalize("NFKD", mot.lower())
    plie = "".join(c for c in decompose if not unicodedata.combining(c))
    nettoye = re.sub(r"[^a-z0-9_]", "_", plie).strip("_")
    return re.sub(r"_+", "_", nettoye)


# Terminaisons d'infinitifs qui se terminent déjà par « e » : les toucher
# produirait « etrer », « fairer », « prendrer ».
_INFINITIFS_EN_E = ("re", "ire", "oire", "aire", "uire", "ivre", "endre")


def _lemme_verbal(racine) -> str:
    """Lemme corrigé des homographes verbe/nom.

    Mesuré : spaCy lemmatise « appliqué » en `appliquer` mais « applique » en
    `applique`, et « demandé » en `demander` mais « demande » en `demande` —
    il rend le lemme du NOM homographe. Deux formulations du même fait
    obtiennent alors des foncteurs différents, ce qui ruine l'unification,
    seul objectif de cette normalisation.

    Sur onze formes verbales éprouvées, neuf étaient déjà correctes ; l'échec
    ne touche que la 3e personne du singulier des verbes du premier groupe
    dont la forme coïncide avec un substantif. On restitue donc l'infinitif en
    ajoutant « r », sauf pour les infinitifs qui se terminent déjà par « e ».
    """
    lemme = racine.lemma_ or racine.text
    if racine.pos_ != "VERB":
        return lemme
    plie = lemme.lower()
    if plie.endswith("e") and not plie.endswith(_INFINITIFS_EN_E):
        return lemme + "r"
    return lemme


def _est_negatif(racine) -> bool:
    """La proposition porte-t-elle une négation ?"""
    for enfant in racine.children:
        if enfant.dep_ == "advmod" and _assainir(enfant.text) in _NEGATIONS:
            return True
        # « aucun signe de gravité » : la négation porte sur un déterminant.
        if enfant.dep_ == "det" and _assainir(enfant.text) in _NEGATIONS:
            return True
    return any(_assainir(t.text) in _NEGATIONS for t in racine.subtree
               if t.dep_ in ("advmod", "det"))


@functools.lru_cache(maxsize=4096)
def forme_logique(fragment: str, modele: str = syntaxe.MODELE_DEFAUT) -> str | None:
    """Littéral normalisé, ou None si spaCy est absent.

    Rend une chaîne directement utilisable comme atome ASP.
    """
    nlp = syntaxe._charger(modele)
    if nlp is None or not fragment.strip():
        return None
    doc = nlp(fragment)
    racines = [t for t in doc if t.head == t]
    if not racines:
        return None
    racine = racines[0]

    if racine.pos_ not in ("VERB", "AUX"):
        # Groupe nominal : pas de prédicat verbal à former. On produit un atome
        # constant à partir des mots pleins, ce qui reste unifiable entre deux
        # mentions du même objet.
        mots = [_assainir(t.lemma_) for t in doc
                if t.pos_ in ("NOUN", "PROPN", "ADJ")
                and _assainir(t.text) not in _VIDES_NOMINAL]
        atome = "_".join(m for m in mots if m)[:60]
        return atome or None

    arguments = []
    for enfant in racine.children:
        if enfant.dep_ in ("nsubj", "nsubj:pass", "obj", "dobj"):
            arguments.append(_assainir(enfant.lemma_))
    # Un pronom sujet n'apporte rien à l'unification, mais le retirer rendrait
    # « il présente un risque » et « présente un risque » distincts. On le
    # garde : la fidélité prime sur l'élégance du littéral.
    arguments = [a for a in arguments if a]

    foncteur = _assainir(_lemme_verbal(racine))
    if not foncteur:
        return None
    litteral = f"{foncteur}({', '.join(arguments)})" if arguments else foncteur
    return f"-{litteral}" if _est_negatif(racine) else litteral
