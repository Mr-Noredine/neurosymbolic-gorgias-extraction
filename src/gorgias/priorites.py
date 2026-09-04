"""Première passe : isoler les énoncés de classement avant tout étiquetage.

Le problème traité, mesuré sur `08-priorite-simple` : la phrase de classement
« la remise l'emporte sur le paiement d'avance » est étiquetée comme une
Option. Elle engendre donc une règle qui n'existe pas, puis la préférence
désigne cette règle fantôme comme gagnante et se rattache au mauvais contexte.
Une seule faute d'étiquetage produit à la fois l'invention et la priorité
fausse.

Deux corrections par le prompt ont été tentées et mesurées ; les deux ont
aggravé le résultat. Ce module change de moyen : le classement est repéré
symboliquement et RETIRÉ du flux soumis au modèle, qui ne peut donc plus le
prendre pour une conclusion.

Ce qui est retiré, et ce qui ne l'est pas
-----------------------------------------
Seule la CLAUSE DE CLASSEMENT est soustraite. Le contexte qui la précède reste
dans le flux : dans « Pour un client stratégique, la remise l'emporte sur le
paiement d'avance », notre segmentation produit deux segments, et
« Pour un client stratégique » est précisément le ``When`` de la préférence.
Le retirer supprimerait l'ancrage même que l'on cherche à extraire.
"""
from __future__ import annotations

import re
import unicodedata


# Amorces lexicales du classement. Volontairement génériques : ce sont des
# tournures de comparaison, pas des règles de domaine.
_AMORCES = (
    "prime sur", "primer sur", "l'emporte sur", "lemporte sur", "emporte sur",
    "prevaut sur", "prevaut", "est prefere", "sont preferes", "est preferee",
    "sont preferees", "prefere a", "preferes a", "preferee a", "preferees a",
    "passe avant", "passent avant", "a la priorite", "ont la priorite",
    "est prioritaire", "sont prioritaires", "supplante", "prevaut contre",
    "prend le pas sur", "au detriment de",
)


# Comparatifs DISCONTINUS : « préfère le porc au poulet » place les termes
# comparés entre le verbe et sa préposition, ce qu'une amorce contiguë ne peut
# pas attraper. Mesuré : la liste contiguë couvre 29 marqueurs gold sur 31 ;
# les deux ratés relèvent tous deux de ce motif.
# Les deux termes comparés sont CAPTURÉS. Sans cela, découper le fragment sur
# la correspondance entière tranche au milieu des mots : « il préfère le porc
# au poulet » rendait ('il', 'oulet'), la correspondance ayant absorbé les
# termes eux-mêmes. Mesuré : cinq priorités perdues sur `01-courses-3-niveaux`.
_COMPARATIFS = (
    re.compile(r"\b(?:prefer\w*|privilegi\w*)\s+(.{1,60}?)\s+"
               r"(?:au|aux|a|plutot que)\s+(\S[^.;]{0,60})", re.IGNORECASE),
)


# Comparatif à PREMIER TERME ÉLIDÉ : « il les préfère aux deux autres ». Le
# terme gagnant est repris par un pronom, donc absent de la clause ; le motif
# contigu comme le comparatif discontinu exigent tous deux un terme explicite
# entre le verbe et sa préposition, et laissent donc passer cet énoncé.
#
# La conséquence n'est pas seulement une priorité manquée : le segment reste
# dans le flux, s'y fait étiqueter comme une proposition ordinaire, et sert de
# PRÉMISSE aux options énumérées. Mesuré sur `01-courses-3-niveaux` : trois
# règles de référence reçoivent « il les préfère aux deux autres » pour
# condition et deviennent fausses. Un classement laissé dans le flux pollue,
# il ne se contente pas de manquer.
#
# Ce motif sert UNIQUEMENT à reconnaître le segment pour le soustraire ; il ne
# figure pas dans `_COMPARATIFS` parce qu'on ne peut pas en tirer deux côtés —
# le gagnant est anaphorique. `cotes_du_classement` rend alors None et aucune
# priorité n'est fabriquée : on retire la pollution sans inventer de préférence.
#
# Portée mesurée : 2 marqueurs de référence, 2 occurrences sur le banc, ZÉRO
# sur les 32 récits. La variante plus large — le verbe seul, sans préposition —
# a été mesurée puis REFUSÉE : elle capture « les communications directes sont
# privilégiées » de `48-drones-autonomes`, qui est une conclusion de règle et
# non un classement, et la soustraire détruirait l'option, sa règle et sa
# priorité.
_COMPARATIFS_ELIDES = (
    re.compile(r"\b(?:prefer\w*|privilegi\w*)\s+(?:au|aux|a|plutot que)\b",
               re.IGNORECASE),
)


def _plier(texte: str) -> str:
    """Minuscule sans accents, LONGUEUR PRÉSERVÉE.

    Le repli sert aussi à localiser l'amorce dans le texte source : un
    NFKD global décalerait les indices dès qu'un caractère se décompose en
    plusieurs (les ligatures, par exemple). On plie donc caractère par
    caractère et on garde le caractère d'origine quand le repli n'en rend pas
    exactement un.
    """
    sortie = []
    for caractere in texte.lower():
        decompose = "".join(
            c for c in unicodedata.normalize("NFKD", caractere)
            if not unicodedata.combining(c)
        )
        sortie.append(decompose if len(decompose) == 1 else caractere)
    return "".join(sortie)


def empan_du_marqueur(fragment: str) -> tuple[int, int] | None:
    """Bornes de l'amorce dans le fragment. Voir la version détaillée ci-après."""
    """Bornes de l'amorce DANS le fragment, pour ancrer l'entité Marker.

    Les références ancrent le marqueur sur l'amorce seule (« l'emporte sur »),
    pas sur la clause entière : un empan trop large ne s'apparie pas au gold.
    """
    plie = _plier(fragment)
    contigue = _amorce_contigue(plie)
    if contigue:
        amorce, position = contigue
        return position, position + len(amorce)
    for motif in _COMPARATIFS:
        trouve = motif.search(plie)
        if trouve:
            # Les références bornent le marqueur sur le comparatif ENTIER :
            # « préfère le porc au poulet », termes compris.
            return trouve.start(), trouve.end()
    return None


def _amorce_contigue(plie: str) -> tuple[str, int] | None:
    """Amorce contiguë et sa position, en exigeant une FRONTIÈRE DE MOT.

    Sans cette frontière, « prefere a » correspond au préfixe de
    « préfère aux » et le découpage tranche au milieu du mot : mesuré,
    « il les préfère aux deux autres » rendait ('il les', 'ux deux autres').
    """
    for amorce in _AMORCES:
        trouve = re.search(rf"(?<!\w){re.escape(amorce)}(?!\w)", plie)
        if trouve:
            return amorce, trouve.start()
    return None


def marqueur_dans(fragment: str) -> str | None:
    """Rend l'amorce de classement présente dans le fragment, sinon None."""
    plie = _plier(fragment)
    contigue = _amorce_contigue(plie)
    if contigue:
        return contigue[0]
    for motif in _COMPARATIFS + _COMPARATIFS_ELIDES:
        trouve = motif.search(plie)
        if trouve:
            return trouve.group(0)
    return None


def isoler(texte: str, segments) -> tuple[list[int], dict[int, str]]:
    """Repère les segments qui SONT un classement.

    Rend (numéros à soustraire du flux, {numéro: amorce trouvée}). Les numéros
    sont ceux de la segmentation, à partir de 1.
    """
    retires: list[int] = []
    amorces: dict[int, str] = {}
    for numero, (debut, fin) in enumerate(segments, start=1):
        amorce = marqueur_dans(texte[debut:fin])
        if amorce is not None:
            retires.append(numero)
            amorces[numero] = amorce
    return retires, amorces


_MOT = re.compile(r"[\wÀ-ÖØ-öø-ÿ]{3,}", re.UNICODE)
_VIDES = {
    "les", "des", "une", "aux", "que", "qui", "pour", "avec", "sans", "dans",
    "sur", "par", "est", "sont", "ete", "etre", "cette", "cet", "ces", "son",
    "sa", "ses", "leur", "leurs", "plus", "moins", "tout", "tous", "toute",
}


def _radicaux(fragment: str) -> set[str]:
    """Radicaux approximatifs : les cinq premiers caractères des mots pleins.

    La nominalisation est le cas difficile : « le remplacement l'emporte sur la
    réparation » doit s'apparier à des règles concluant « elle est remplacée »
    et « elle est réparée ». Une comparaison de surface échoue ; tronquer aux
    cinq premiers caractères rapproche remplacement/remplacée (« rempl ») et
    réparation/réparée (« repar ») sans imposer de lemmatiseur.
    """
    mots = (_plier(m.group()) for m in _MOT.finditer(fragment))
    return {m[:5] for m in mots if m not in _VIDES}


def apparier(mention: str, candidats: dict[int, str]) -> int | None:
    """Numéro du candidat dont le texte recouvre le mieux la mention.

    `candidats` associe un numéro de segment à son texte. Rend None si aucun
    candidat ne partage de radical : mieux vaut ne pas produire de priorité
    que d'en produire une fausse.
    """
    cibles = _radicaux(mention)
    if not cibles:
        return None
    meilleur, score_max = None, 0
    for numero, fragment in candidats.items():
        score = len(cibles & _radicaux(fragment))
        if score > score_max:
            meilleur, score_max = numero, score
    return meilleur


def cotes_du_classement(fragment: str) -> tuple[str, str] | None:
    """Découpe « X l'emporte sur Y » en (gagnante, perdante).

    Rend None si l'amorce est absente ou si l'un des côtés est vide : sans les
    deux membres, la comparaison ne désigne rien.
    """
    plie = _plier(fragment)
    contigue = _amorce_contigue(plie)
    if contigue:
        amorce, position = contigue
        gauche = fragment[:position].strip(" ,;:")
        droite = fragment[position + len(amorce):].strip(" ,;:.")
        return (gauche, droite) if gauche and droite else None
    for motif in _COMPARATIFS:
        trouve = motif.search(plie)
        if trouve:
            # Comparatif discontinu : les termes sont DANS la correspondance,
            # on les lit dans les groupes plutôt que de découper autour.
            gauche = fragment[trouve.start(1):trouve.end(1)].strip(" ,;:")
            droite = fragment[trouve.start(2):trouve.end(2)].strip(" ,;:.")
            return (gauche, droite) if gauche and droite else None
    return None
