"""Filtre de singularité : distinguer un enchaînement de récit d'une règle.

Le problème traité
------------------
Deux documents narratifs sur 32 produisaient systématiquement une règle fausse
— six exécutions, six règles — là où la référence est vide :

    « La foudre est tombée sur le transformateur »
        => « Le quartier s'est trouvé sans électricité »
    « L'eau s'est répandue dans la cave »
        => « La cave a été asséchée le lendemain »

Ce sont des causalités RÉELLES du monde, mais racontées, sans être avancées
comme raison de croire quoi que ce soit. Le filtre narratif du modèle les
laisse passer, de façon reproductible.

Le critère retenu, et pourquoi
------------------------------
Un récit enchaîne des événements RÉVOLUS et SINGULIERS ; une règle métier pose
un état qui déclenche une conséquence. La marque en est le temps des DEUX
verbes principaux :

    « La sauvegarde nocturne a échoué »  =>  « La restauration ne peut plus
                                              être garantie »
      passé                                  présent      -> RÈGLE, conservée

    « La foudre est tombée »             =>  « Le quartier s'est trouvé sans
                                              électricité »
      passé                                  passé        -> RÉCIT, rejeté

Il faut DEUX passés, ET au moins un auxiliaire de TEMPS. Un seul passé ne
signale rien — les règles de référence comportent souvent une prémisse au passé
composé à valeur de constat. Et deux participes ne suffisent pas non plus : le
français emploie « être + participe » aussi pour le passif présent, massif dans
le banc (« le sel est appliqué »). L'étiquette `aux:tense`, opposée à
`aux:pass`, sépare les deux.

Un ancrage temporel précis — « le lendemain », « vers vingt-deux heures » —
suffit également : il situe un événement unique, ce qu'une règle générale ne
fait jamais.

Ce que ce filtre coûte, mesuré
------------------------------
Restreint aux paires DÉPOURVUES de connecteur explicite, comme il se doit :
douze relations justes sont dans ce cas, le filtre en rejette deux, formant une
seule règle à deux conditions. Bilan sur le banc :

    faux positifs narratifs   2/32  ->  0/32
    règles justes            67/101 -> 66/101
    F1 des règles            70,2 % ->  69,5 %

La règle perdue vient de `26-chaine-froid`, où spaCy analyse « Comme » en
préposition (`case`, ADP) et non en conjonction : le court-circuit ne la prouve
donc pas et le filtre la voit. C'est une erreur du parseur, pas du mécanisme —
vérifié, l'analyse par phrase ne la corrige pas davantage.

Pourquoi un filtre symbolique plutôt qu'une instruction au modèle
-----------------------------------------------------------------
Trois corrections par le prompt ont été tentées sur ce projet, les trois
retirées après mesure. Le filtre narratif du modèle échoue ici de façon
REPRODUCTIBLE — six fois sur six — donc le hasard n'y est pour rien. Un test
morphologique déterministe ne dépend pas de la variance d'inférence.
"""
from __future__ import annotations

import functools
import unicodedata

from . import syntaxe

# Noms qui situent un événement unique dans le temps. Ce sont des repères
# déictiques, pas un vocabulaire de domaine : « en hiver » ou « en période
# d'examens » n'y figurent pas, car ils décrivent des situations GÉNÉRALES qui
# ancrent légitimement des règles métier.
_ANCRAGES = {
    "nuit", "heure", "lendemain", "veille", "hier", "matin", "soir",
    "aujourd", "demain", "minuit", "midi", "apres-midi", "moment",
}
_RELATIONS_ANCRAGE = ("obl", "obl:mod", "advmod", "nmod")


def _plier(mot: str) -> str:
    decompose = unicodedata.normalize("NFKD", mot.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def _racine(doc):
    racines = [token for token in doc if token.head == token]
    return racines[0] if racines else None


def _au_passe(token) -> bool:
    """Le verbe porte-t-il la marque du passé ?"""
    return token is not None and token.morph.get("Tense") == ["Past"]


def _passe_compose(token) -> bool:
    """Passé composé AVÉRÉ, distingué du passif présent.

    Le français emploie « être + participe » pour deux choses : le passé
    composé (« la foudre est tombée ») et la voix passive au présent (« le sel
    est appliqué »). Le trait `Tense=Past` du participe vaut dans les deux cas,
    si bien qu'un test morphologique naïf rejetait les conclusions passives —
    or elles sont massives dans le banc : « le sel est appliqué »,
    « l'avertissement est donné », « la consultation est imposée ».

    L'étiquette de l'auxiliaire tranche : `aux:tense` marque le temps composé,
    `aux:pass` la voix. On exige donc, sur la paire, au moins un auxiliaire de
    TEMPS — spaCy étiquetant parfois `aux:pass` un passé composé de verbe de
    mouvement (« est tombée »), le second membre de la paire le rattrape.
    """
    return token is not None and any(
        enfant.dep_ == "aux:tense" for enfant in token.children
    )


def _ancre_dans_le_temps(doc) -> bool:
    return any(_plier(token.lemma_) in _ANCRAGES for token in doc
               if token.dep_ in _RELATIONS_ANCRAGE)


@functools.lru_cache(maxsize=4096)
def evenement_singulier(fragment_a: str, fragment_b: str,
                        modele: str = syntaxe.MODELE_DEFAUT) -> bool:
    """La paire relate-t-elle un enchaînement d'événements singuliers ?

    Sans spaCy, rend False : le filtre s'efface et le comportement antérieur
    est retrouvé.
    """
    if syntaxe.desactive():
        return False
    nlp = syntaxe._charger(modele)
    if nlp is None:
        return False
    doc_a, doc_b = nlp(fragment_a), nlp(fragment_b)
    racine_a, racine_b = _racine(doc_a), _racine(doc_b)
    deux_au_passe = _au_passe(racine_a) and _au_passe(racine_b)
    un_temps_compose = _passe_compose(racine_a) or _passe_compose(racine_b)
    if deux_au_passe and un_temps_compose:
        return True
    return _ancre_dans_le_temps(doc_a) or _ancre_dans_le_temps(doc_b)
