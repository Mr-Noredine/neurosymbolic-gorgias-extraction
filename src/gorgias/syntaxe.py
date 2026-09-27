"""Court-circuit symbolique : les subordonnées explicites ne passent pas par le LLM.

Le juge de paires détruit des relations conditionnelles pourtant évidentes —
mesuré, 14 documents sur 47 ne produisent rien alors que l'étape de proposition
avait trouvé les bons liens, soit 31 règles perdues sur 101. Trois tentatives
de correction par le prompt ont échoué.

Ce module traite autrement la moitié du problème qui est décidable
grammaticalement. Dans « Quand le sol est gelé, le sel est appliqué », la
subordonnée circonstancielle porte l'étiquette de dépendance ``advcl`` et son
introducteur (« quand ») porte ``mark`` : la condition est alors identifiable
sans aucune probabilité. On enregistre la relation directement et le LLM n'est
appelé que pour les liens dépourvus de marqueur explicite.

Plafond mesuré sur les 116 relations de référence du banc :

    fr_core_news_sm   59/116 (51 %)   0 en direction inverse    26 Mo
    fr_core_news_md   61/116 (53 %)   0 en direction inverse    63 Mo
    fr_core_news_lg   68/116 (59 %)   0 en direction inverse   613 Mo

Le second chiffre est le plus important : quand le motif se déclenche,
l'orientation est toujours juste. Le court-circuit ajoute donc du rappel sans
introduire d'erreur de direction, ce qui convient à une cible où la précision
prime. Il ne « garantit » pas la récupération intégrale : environ la moitié des
relations de référence n'est portée par aucune subordonnée explicite et reste
du ressort du modèle de langue.
"""
from __future__ import annotations

import functools
import os
import re


# `lg` retrouve neuf relations de plus que `sm` pour 613 Mo contre 26 : l'écart
# est mesuré, pas supposé, et le disque pèse peu à côté des 6,7 Go du modèle de
# langue. Le nom reste configurable pour que `sm` reste utilisable sur une
# machine contrainte, au prix de huit points de couverture.
MODELE_DEFAUT = "fr_core_news_lg"


def desactive() -> bool:
    """Interrupteur de mesure : LPP_SANS_SYNTAXE=1 rend le court-circuit inerte.

    Comparer « avec » et « sans » suppose de ne faire varier QUE ce facteur.
    Sans cet interrupteur, la seule façon d'obtenir la référence serait de
    changer de version de code ou de machine — et l'écart mesuré mélangerait
    alors deux causes.
    """
    return os.environ.get("LPP_SANS_SYNTAXE", "") not in ("", "0")


@functools.lru_cache(maxsize=2)
def _charger(nom: str):
    """Charge le modèle une seule fois. Absent, le court-circuit se désactive.

    L'analyse syntaxique est un accélérateur, pas une dépendance dure : sans
    spaCy installé, le pipeline doit continuer à fonctionner exactement comme
    avant plutôt que de refuser de démarrer.
    """
    try:
        import spacy
    except ImportError:
        return None
    try:
        # Le lemmatiseur est conservé : `predicats.py` en a besoin pour former
        # ses foncteurs, et charger un second exemplaire du modèle coûterait
        # 613 Mo de plus. Seule la reconnaissance d'entités nommées, inutile
        # ici, est désactivée.
        return spacy.load(nom, disable=["ner"])
    except OSError:
        return None


def _recouvre(debut: int, fin: int, span: tuple[int, int]) -> bool:
    return not (fin <= span[0] or debut >= span[1])


def _empan(tokens) -> tuple[int, int]:
    return (min(t.idx for t in tokens),
            max(t.idx + len(t.text) for t in tokens))


@functools.lru_cache(maxsize=4096)
def coupures_de_clauses(texte: str, debut: int, fin: int,
                        modele: str = MODELE_DEFAUT) -> frozenset[int] | None:
    """Virgules qui séparent réellement deux clauses dans un fragment.

    Une virgule sépare aussi les éléments d'une liste, une apposition ou un
    complément déplacé. Les couper tous fabriquait des chaînes de règles dans
    les textes réglementaires. Une coupure est retenue si ce qui la suit porte
    un sujet et un verbe fini autonomes. Les coordinations nominales et les
    compléments sans sujet restent dans leur clause. Les deux-points restent
    toujours des frontières typographiques.

    ``None`` signifie que spaCy n'est pas disponible : l'appelant conserve
    alors le comportement historique, afin que la dépendance reste optionnelle.
    """
    if desactive():
        return None
    nlp = _charger(modele)
    if nlp is None:
        return None
    fragment = texte[debut:fin]
    document = nlp(fragment)
    ponctuations = [t for t in document if t.text in {",", ":"}]
    # Compatibilité : une virgule reste une frontière par défaut. On ne retire
    # que celles dont l'analyse prouve le rôle INTERNE. Cette asymétrie garde
    # intact le rappel historique quand le parseur hésite.
    retenues: set[int] = {
        debut + t.idx + len(t.text) for t in ponctuations
    }
    phrase_debut = max(
        texte.rfind(".", 0, debut), texte.rfind(";", 0, debut),
        texte.rfind("\n", 0, debut),
    ) + 1
    unite = texte[phrase_debut:fin]
    protegeable = (
        fragment.count(",") >= 2
        and (
            re.search(
                r"\b(?:si\b|lorsqu|quand\b|comme\b|en\s+l['’]absence\b)",
                unite, re.I,
            )
            or re.search(r"\by\s+compris\b", unite, re.I)
        )
    )
    if not protegeable:
        return frozenset(retenues)
    utiles = [t for t in document if not (t.is_space or t.is_punct)]
    for ponctuation in ponctuations:
        position = debut + ponctuation.idx + len(ponctuation.text)
        if ponctuation.text == ":":
            continue
        suivants = [t for t in utiles if t.idx > ponctuation.idx]
        if not suivants:
            continue
        premier = suivants[0]
        if premier.lower_ in {"sauf", "hormis", "excepté", "même"}:
            continue
        # « et lorsque B, C » est bien une seconde subordonnée ; un simple
        # « ou le service » est au contraire le membre d'une coordination.
        if premier.pos_ == "CCONJ":
            apres = suivants[1] if len(suivants) > 1 else None
            if apres is not None and (
                apres.pos_ == "SCONJ"
                or apres.lower_ in {"si", "lorsque", "quand", "comme"}
            ):
                continue
            retenues.discard(position)
            continue

        prochaine = next(
            (t for t in document
             if t.idx > ponctuation.idx and t.is_punct and t.text in {",", ":"}),
            None,
        )
        limite = prochaine.idx if prochaine is not None else len(fragment)
        tranche = [t for t in utiles if ponctuation.idx < t.idx < limite]
        sujets = [t for t in tranche if t.dep_ in {"nsubj", "nsubj:pass", "expl:subj"}]
        verbes = [
            t for t in tranche
            if t.pos_ in {"VERB", "AUX"} and "Fin" in t.morph.get("VerbForm")
        ]
        # Apposition ou complément entre deux virgules.
        if ((prochaine is not None and premier.pos_ in {"ADP", "ADJ"})
                or (premier.pos_ == "ADP" and len(ponctuations) >= 2)):
            retenues.discard(position)
            continue
        # Membre nominal d'une coordination (« le pays, un territoire… »).
        premiers_nominaux = [t for t in tranche[:6] if t.pos_ in {"NOUN", "PROPN", "ADJ"}]
        if any(t.dep_ in {"conj", "appos"} for t in premiers_nominaux):
            retenues.discard(position)
            continue
        # Prédicat coordonné sans nouveau sujet à l'intérieur d'une grande
        # subordonnée ; la virgule suivante ferme cette subordonnée.
        if (prochaine is not None and verbes and not sujets
                and re.match(r"\s*(?:si\b|lorsqu|quand\b|comme\b)",
                             fragment, re.I)):
            retenues.discard(position)
            continue
        # « le projet, la modification ou la prorogation respecte… » : le
        # second groupe nominal est encore un élément du sujet coordonné, pas
        # le début d'une principale.
        avant = [t for t in utiles if t.idx < ponctuation.idx]
        sujets_avant = [
            t for t in avant
            if t.dep_ in {"nsubj", "nsubj:pass", "expl:subj"}
        ]
        if any(
            ancien.head == sujet.head
            and any(enfant.dep_ == "conj" for enfant in sujet.children)
            for ancien in sujets_avant
            for sujet in sujets
        ):
            retenues.discard(position)
    return frozenset(retenues)


@functools.lru_cache(maxsize=4096)
@functools.lru_cache(maxsize=4096)
def ouvre_une_clause(tete: str, modele: str = MODELE_DEFAUT) -> bool | None:
    """Le fragment donné ouvre-t-il une CLAUSE autonome ?

    Sert au seul cas d'une ponctuation forte suivie d'une minuscule. La règle
    de surface — « un point suivi d'une minuscule ne termine pas une phrase »
    — protège « etc. le reste » et « M. le maire », mais recolle deux phrases
    entières dès qu'un texte n'ouvre pas ses phrases par une capitale. Mesuré
    sur `data/synthetique` : 58 frontières manquées, d'où des segments portant
    deux conclusions, des empans qui ne s'apparient plus à la référence, et
    des règles fausses là où la prémisse s'accrochait au segment recollé.

    Le test remplace la surface par la grammaire : on coupe si ce qui suit
    porte un SUJET et un VERBE FINI propres, ce qui est vrai d'une phrase et
    faux d'une apposition ou d'un groupe nominal. ``None`` signale l'absence
    de spaCy : l'appelant garde alors la règle de surface, plus prudente.
    """
    nlp = _charger(modele)
    if nlp is None:
        return None
    document = nlp(tete)
    for token in document:
        if token.dep_ not in {"nsubj", "nsubj:pass"}:
            continue
        tete_verbale = token.head
        if tete_verbale.pos_ not in {"VERB", "AUX"}:
            continue
        # Au passif et aux temps composés, le sujet dépend du PARTICIPE, dont
        # la forme n'est pas finie : « les couloirs ont été rénovés » a
        # `rénovés` pour tête et `ont` pour auxiliaire. Chercher la finitude
        # sur la seule tête refusait donc toute phrase au passif — soit, dans
        # ce corpus, la moitié d'entre elles.
        formes = [tete_verbale] + [
            enfant for enfant in tete_verbale.children
            if enfant.pos_ == "AUX"
        ]
        if any("VerbForm=Fin" in str(forme.morph) for forme in formes):
            return True
    return False


def _traits_fragment(fragment: str, modele: str) -> tuple[tuple, ...] | None:
    """Analyse légère mémorisée des petits segments utilisés par les gardes.

    Une même proposition est contrôlée avant l'étiquetage, avant la proposition
    de liens puis pendant la vérification. Réanalyser ce fragment avec spaCy à
    chacun de ces passages dominait désormais le temps des documents sans LLM.
    """
    nlp = _charger(modele)
    if nlp is None:
        return None
    return tuple(
        (
            token.is_punct,
            token.is_space,
            token.pos_,
            token.dep_,
            token.lower_,
            tuple(token.morph.get("PronType")),
        )
        for token in nlp(fragment)
    )


@functools.lru_cache(maxsize=32)
def aretes_explicites(texte: str, modele: str = MODELE_DEFAUT) -> list[tuple]:
    """Empans (condition, conclusion) prouvés par une subordonnée marquée.

    Retourne une liste de couples d'empans de caractères. La direction suit la
    grammaire : la subordonnée ``advcl`` est la condition, la proposition dont
    elle dépend est la conclusion.
    """
    if desactive():
        return []
    nlp = _charger(modele)
    if nlp is None:
        return []
    aretes = []
    for token in nlp(texte):
        if token.dep_ != "advcl":
            continue
        if not any(enfant.dep_ == "mark" for enfant in token.children):
            continue
        subordonnee = list(token.subtree)
        principale = [t for t in token.head.subtree if t not in set(subordonnee)]
        if not subordonnee or not principale:
            continue
        aretes.append((_empan(subordonnee), _empan(principale)))
    return aretes


_CONDITION_TEXTUELLE = re.compile(
    r"^\s*(?:mais\s+)?(?:si\b|s['’](?:il|elle|on|ils|elles)\b|"
    r"quand\b|lorsqu|comme\b|puisque\b|étant\s+donné\s+que\b|chez\b|"
    r"dès\s+que\b|dans\s+(?:cette\s+situation|ce\s+cas)\b|"
    r"en\s+l['’]absence\s+de\b|en\s+présence\s+d|"
    r"à\s+l['’]échéance\b|en\s+cas\s+de\b|"
    r"pour\s+(?:économiser|éviter|réduire|garantir|assurer)\b)",
    re.IGNORECASE,
)
_CIRCONSTANCE_COURTE = re.compile(r"^\s*en\b", re.IGNORECASE)
_CONNECTEUR_INTERNE = re.compile(
    r"^\s*(?:que\s+s['’](?:il|elle|on|ils|elles)|que\s+si|"
    r"s['’](?:il|elle|on|ils|elles)|si|quand|lorsque|dès\s+que|"
    r"only\s+if|if|when|parce\s+que|because|provided\s+that|à)\s*$",
    re.IGNORECASE,
)
_CLASSEMENT_SANS_REGLE = re.compile(
    r"\bprioritaire\s+par\s+rapport\s+[àa]\b|"
    r"\bpr[ée]f[èe]re?\b.*\bmême\s+en\b",
    re.IGNORECASE,
)
_ARTICULATION = re.compile(r"^\s*en\s+revanche\s*$", re.IGNORECASE)
_REPRISE_TOPIQUE = re.compile(r"^\s*(?:lui|elle|eux|elles)\b", re.IGNORECASE)
_REPRISE_DEMONSTRATIVE = re.compile(
    r"^\s*(?:un\s+tel|une\s+telle|de\s+tels|de\s+telles|ce|cet|cette|ces)\b",
    re.IGNORECASE,
)
_CONDITION_COORDONNEE = re.compile(
    r"^\s*(?:(?:et|ou)\s+(?:que|si|lorsque|quand)|sans)\b", re.IGNORECASE
)
_CONDITION_NOMINALE = re.compile(
    r"^\s*(?:à\s+l['’]échéance\b|en\s+cas\s+de\b)", re.IGNORECASE
)
# L'intervalle entre deux segments d'une MÊME phrase ne contient que de la
# ponctuation faible. Un point, un point-virgule ou une ligne vide marquent au
# contraire une frontière que nulle subordination ne franchit.
_MEME_PHRASE = re.compile(r"[\s,:—–-]*")
_LIAISON_CONDITIONS = re.compile(
    r"(?:\bou\s+(?:si\b|s['’](?:il|elle|on|ils|elles)\b)|"
    r"\bet\s+à\s+la\s+condition\s+que)\s*$",
    re.IGNORECASE,
)
# CONNECTEUR DE CONSÉQUENCE ISOLÉ. « Par conséquent », « Dès lors », « Donc »
# suivis d'une virgule forment leur propre segment : la découpe sur la
# ponctuation les détache de la clause qu'ils introduisent. Ils sont l'exact
# symétrique des connecteurs de condition déjà traités ici — même classe
# fermée de connecteurs de discours, même certitude grammaticale, direction
# inverse : la conclusion est le segment qui SUIT, la prémisse celui qui
# PRÉCÈDE.
#
# La forme isolée est seule retenue, et c'est ce qui rend le motif exact.
# « dès lors QUE le sinistre relève d'un événement climatique » est une
# CONDITION et non une conséquence : l'ancrage sur la fin de segment l'exclut
# sans avoir à énumérer les tournures. Et « La restauration ne peut DONC plus
# être garantie », où le connecteur est enchâssé dans la clause, est écarté
# aussi : les références y demandent DEUX prémisses (`04-chainage-technique`),
# que rien dans la surface ne permet de compter — produire la seule prémisse
# adjacente y fabriquerait une règle fausse.
#
# Portée mesurée sur les 188 documents annotés : 37 segments dans le corpus
# synthétique, où les familles « chaine2 » et « chaine3 » — 40 documents —
# sont construites sur ce motif. ZÉRO occurrence dans les 32 récits négatifs
# et dans les lots RGPD, donc aucune pollution possible.
_CONSEQUENCE_ISOLEE = re.compile(
    r"^\s*(?:et\s+|puis\s+)?(?:donc|par\s+cons[ée]quent|d[èe]s\s+lors|"
    r"c['’]est\s+pourquoi|en\s+cons[ée]quence|de\s+ce\s+fait|"
    r"il\s+en\s+r[ée]sulte)\s*$",
    re.IGNORECASE,
)


# Subordonnée introduite : ce qu'un connecteur de conséquence doit FRANCHIR
# pour trouver sa prémisse. « une relecture externe est commandée, car la
# traduction n'a pas été relue. Par conséquent, le tirage doit être réduit. »
# — la conclusion suit de la relecture commandée, PAS de la subordonnée
# causale qui la justifie. Prendre le segment immédiatement précédent
# fabriquait ici une règle fausse ; mesuré, six sur le corpus synthétique.
#
# « car » et « parce que » figurent ici et pas dans `_CONDITION_TEXTUELLE` :
# celle-là sert à détacher une condition en tête de phrase, ce que « car » ne
# fait jamais en français.
# CAUSE POSTPOSÉE. « une relecture externe est commandée, car la traduction
# n'a pas été relue » : la subordonnée causale suit sa principale, et c'est
# elle qui la fonde. Le motif est le symétrique exact de `puisque` — déjà
# traité en tête de phrase — mais « car » et « parce que » ne s'y placent
# jamais en français, d'où une entrée distincte.
#
# « car » est une conjonction de COORDINATION : le parseur dépendanciel lui
# donne `cc`/`conj` et non `advcl`+`mark`, si bien que le court-circuit
# syntaxique ne la voit pas. C'est le seul connecteur causal courant dans ce
# cas, et il était donc absent de bout en bout.
#
# Mesuré sur les 188 documents annotés : 24 segments correspondent, et 23 sont
# CONFIRMÉS par la référence — le segment précédent y est bien une conclusion
# dont celui-ci est condition. Le seul non confirmé est dans
# `48-drones-autonomes`. ZÉRO occurrence dans les 32 récits négatifs.
_CAUSE_POSTPOSEE = re.compile(r"^\s*(?:car|parce\s+qu)\b", re.IGNORECASE)


def _ouvre_sa_phrase(texte: str, debut: int) -> bool:
    """Le segment est-il en TÊTE de sa phrase ?

    Une condition DÉTACHÉE ouvre sa phrase : « Quand la commande dépasse mille
    euros, le service applique une remise. » La même conjonction placée après
    sa principale est POSTPOSÉE et se rattache en arrière : « nous hébergeons
    dans le cloud, puisque la sauvegarde nocturne a échoué. » La distinction
    est de position, pas de vocabulaire.

    Sans elle, la subordonnée postposée recevait AUSSI une arête vers la
    phrase suivante, et fabriquait une règle qui enjambe le point. Mesuré sur
    `g0131-preference-informatique` : « Pour un budget contraint » — le
    contexte de la priorité — devenait la conclusion d'une règle fondée sur la
    cause de la phrase d'avant. Une seule arête produisait ainsi une règle
    fausse, une Option fausse à la place d'un Context, et un ancrage de
    priorité faux.
    """
    return not texte[:debut].rstrip(" \t\r\n").endswith((",", ";", ":"))\
        and texte[:debut].rstrip(" \t\r\n")[-1:] in ("", ".", "!", "?", "\n")


_SUBORDONNEE_INTRODUITE = re.compile(
    r"^\s*(?:car|parce\s+que|puisque|comme|si|s['’](?:il|elle|on|ils|elles)|"
    r"quand|lorsqu|d[èe]s\s+que|[ée]tant\s+donn[ée]\s+que|"
    r"attendu\s+que|vu\s+que)\b",
    re.IGNORECASE,
)


def est_connecteur_de_consequence(fragment: str) -> bool:
    """Le segment n'est-il QUE un connecteur de conséquence ?

    Ces segments n'énoncent rien : les références ne leur donnent aucune
    entité. Ils doivent donc être retirés du flux d'étiquetage, faute de quoi
    « Par conséquent » se fait étiqueter comme une proposition et devient un
    Context fantôme — puis, s'il sert de prémisse, une règle fausse.
    """
    return bool(_CONSEQUENCE_ISOLEE.match(fragment))


_AGE_TEMPOREL = re.compile(
    r"\b(?:avait|avaient|était|étaient)\b[^.;]{0,30}\bans\b",
    re.IGNORECASE,
)


def aretes_textuelles(texte: str, segments) -> list[tuple]:
    """Repli exact pour les connecteurs que le parseur dépendanciel rate.

    Il ne cherche aucune relation sémantique : il ne couvre que deux formes
    de surface fermées, une condition détachée juste avant sa principale et
    une condition suffixée dont le découpeur a conservé le connecteur dans
    l'intervalle entre les deux segments.
    """
    aretes = []
    for index in range(len(segments) - 1):
        gauche, droite = segments[index], segments[index + 1]
        fragment_g = texte[gauche[0]:gauche[1]]
        fragment_d = texte[droite[0]:droite[1]]
        nominale_g = bool(_CONDITION_NOMINALE.search(fragment_g))
        nominale_d = bool(_CONDITION_NOMINALE.search(fragment_d))
        en_tete = _ouvre_sa_phrase(texte, gauche[0])
        condition_detachee = en_tete and bool(
            _CONDITION_TEXTUELLE.search(fragment_g))
        if (en_tete
                and _CIRCONSTANCE_COURTE.search(fragment_g)
                and len(fragment_g.split()) <= 6
                and not _ARTICULATION.fullmatch(fragment_g)):
            condition_detachee = True
        if (condition_detachee and not nominale_g
                and not _CONDITION_COORDONNEE.search(fragment_d)
                and not _CLASSEMENT_SANS_REGLE.search(fragment_d)):
            aretes.append((gauche, droite))

        # Une condition nominale suffixée dépend de la principale qui la
        # précède. Plusieurs conditions successives se rattachent toutes à la
        # même principale, jamais les unes aux autres.
        #
        # SUFFIXÉE VEUT DIRE DANS LA MÊME PHRASE. « En cas de sinistre »
        # rattaché en arrière est une lecture correcte de « l'indemnité est
        # due, en cas de sinistre » ; elle est fausse dès qu'un point sépare
        # les deux, car le groupe ouvre alors sa propre phrase et conditionne
        # ce qui SUIT. Sur `32-tri-urgences` et `38-conseil-municipal`, cette
        # arête faisait entrer le contexte de la priorité (« En cas de
        # suspicion de fracture ») dans les conditions de la règle précédente :
        # la règle devenait fausse, et la priorité qui s'y appuyait avec elle —
        # une seule arête détruisait deux éléments de référence.
        if nominale_d and _MEME_PHRASE.fullmatch(texte[gauche[1]:droite[0]]):
            cible = index
            while cible > 0 and _CONDITION_NOMINALE.search(
                    texte[segments[cible][0]:segments[cible][1]]):
                cible -= 1
            aretes.append((droite, segments[cible]))

        # Une cause postposée fonde la principale qui la précède, dans la
        # même phrase. Plusieurs causes successives se rattachent toutes à
        # cette principale, jamais les unes aux autres.
        if (_CAUSE_POSTPOSEE.match(fragment_d)
                and _MEME_PHRASE.fullmatch(texte[gauche[1]:droite[0]])):
            cible = index
            while cible > 0 and _CAUSE_POSTPOSEE.match(
                    texte[segments[cible][0]:segments[cible][1]]):
                cible -= 1
            aretes.append((droite, segments[cible]))

        # Le connecteur de conséquence isolé enjambe : sa conclusion est le
        # segment qui le suit, sa prémisse la PRINCIPALE qui le précède — les
        # subordonnées introduites qui s'interposent sont franchies, car elles
        # fondent l'assertion précédente au lieu de l'être.
        if index > 0 and est_connecteur_de_consequence(fragment_g):
            amont = index - 1
            while amont > 0 and _SUBORDONNEE_INTRODUITE.match(
                    texte[segments[amont][0]:segments[amont][1]]):
                amont -= 1
            if not _SUBORDONNEE_INTRODUITE.match(
                    texte[segments[amont][0]:segments[amont][1]]):
                aretes.append((segments[amont], droite))

        intervalle = texte[gauche[1]:droite[0]]
        if (_CONNECTEUR_INTERNE.fullmatch(intervalle)
                and not _AGE_TEMPOREL.search(fragment_d)
                and not _CLASSEMENT_SANS_REGLE.search(fragment_g)):
            aretes.append((droite, gauche))
        # « Un investisseur tolérant au risque, lui, retient… » : la virgule
        # détache un topique que le pronom reprend explicitement. Ce n'est pas
        # une simple proximité entre deux phrases, mais une même construction.
        if (intervalle.strip() == ","
                and _REPRISE_TOPIQUE.search(fragment_d)):
            aretes.append((gauche, droite))
        # « Comme A et que B, C » : les deux segments coordonnés portent la
        # même conclusion, située juste après eux. Le parseur dépendanciel rate
        # parfois entièrement ce motif selon le vocabulaire du domaine.
        if (index + 2 < len(segments)
                and _CONDITION_TEXTUELLE.search(fragment_g)
                and _CONDITION_COORDONNEE.search(fragment_d)):
            conclusion = segments[index + 2]
            aretes.extend(((gauche, conclusion), (droite, conclusion)))

        # « Si A, B. Les autres doivent alors C. » : « alors » reprend sans
        # ambiguïté la condition explicite deux segments plus tôt.
        if (index + 2 < len(segments) and condition_detachee):
            suite = segments[index + 2]
            if re.search(r"\balors\b", texte[suite[0]:suite[1]], re.IGNORECASE):
                aretes.append((gauche, suite))

        # « lorsqu'un drone transporte du matériel. Dans ce cas, ... » : le
        # démonstratif réactive la condition précédente, il n'en crée pas une
        # nouvelle dont le span serait « Dans ce cas ».
        if (index > 0 and re.fullmatch(
                r"\s*dans\s+ce\s+cas\s*", fragment_g, re.IGNORECASE)):
            precedente = segments[index - 1]
            aretes.append((precedente, droite))

    # Le découpeur retire le connecteur de la seconde condition. Sa trace
    # demeure toutefois exactement dans l'intervalle source : « ou si » crée
    # une alternative et « et à la condition que » une conjonction. Dans les
    # deux cas, la seconde condition porte sur la même principale que l'aînée.
    for rang in range(len(segments) - 1):
        gauche, droite = segments[rang], segments[rang + 1]
        liaison = texte[max(gauche[0], gauche[1] - 12):droite[0]]
        if not _LIAISON_CONDITIONS.search(liaison):
            continue
        for condition, conclusion in list(aretes):
            if condition == gauche:
                aretes.append((droite, conclusion))

    # Une seconde phrase qui reprend explicitement la conclusion précédente
    # (« Un tel transfert… », « Cette décision… ») hérite de sa condition.
    # Le démonstratif fournit l'ancrage ; la simple proximité ne suffit pas.
    for condition, conclusion in list(aretes):
        try:
            rang = segments.index(conclusion)
        except ValueError:
            continue
        try:
            rang_condition = segments.index(condition)
        except ValueError:
            rang_condition = rang
        suivant = max(rang, rang_condition) + 1
        if suivant < len(segments):
            fragment = texte[segments[suivant][0]:segments[suivant][1]]
            if _REPRISE_DEMONSTRATIVE.search(fragment):
                aretes.append((condition, segments[suivant]))
    return aretes


def relations_prouvees(texte: str, segments, paires,
                       modele: str = MODELE_DEFAUT) -> dict:
    """Verdicts A_TO_B / B_TO_A décidés grammaticalement, pour les paires données.

    Une paire n'est tranchée que si l'un de ses segments recouvre la
    subordonnée et l'autre la principale. Les paires non couvertes sont
    absentes du dictionnaire et restent à la charge du modèle de langue.
    """
    textuelles = aretes_textuelles(texte, segments)
    # Après « X que s'il Y », Y est une condition suffixée dont le connecteur
    # a été retiré du span. Le parseur du document complet peut alors rattacher
    # à tort une condition précédente à Y et en faire une conclusion autonome.
    # La provenance du découpage permet de l'interdire sans heuristique.
    conditions_suffixees = {
        condition
        for condition, conclusion in textuelles
        if condition[0] > conclusion[0]
        and _CONNECTEUR_INTERNE.fullmatch(texte[conclusion[1]:condition[0]])
    }
    conditions_temporelles = {
        segments[index]
        for index in range(1, len(segments))
        if _AGE_TEMPOREL.search(texte[segments[index][0]:segments[index][1]])
        and _CONNECTEUR_INTERNE.fullmatch(
            texte[segments[index - 1][1]:segments[index][0]]
        )
    }
    aretes = [
        arete for arete in aretes_explicites(texte, modele)
        if arete[1] not in conditions_suffixees
    ] + textuelles
    if not aretes:
        return {}
    verdicts = {}
    for paire in paires:
        gauche, droite = paire
        span_g, span_d = segments[gauche - 1], segments[droite - 1]
        for condition, conclusion in aretes:
            fragment_conclusion = texte[conclusion[0]:conclusion[1]]
            if _CLASSEMENT_SANS_REGLE.search(fragment_conclusion):
                continue
            if (span_g not in conditions_temporelles
                    and _recouvre(*condition, span_g)
                    and span_d not in conditions_suffixees
                    and _recouvre(*conclusion, span_d)):
                verdicts[paire] = "A_TO_B"
                break
            if (span_d not in conditions_temporelles
                    and _recouvre(*condition, span_d)
                    and span_g not in conditions_suffixees
                    and _recouvre(*conclusion, span_g)):
                verdicts[paire] = "B_TO_A"
                break
    return verdicts


def peut_conclure(texte: str, debut: int, fin: int,
                  modele: str = MODELE_DEFAUT) -> bool:
    """Ce segment peut-il être la CONCLUSION d'une règle ?

    Deux formes ne le peuvent pas, pour des raisons de grammaire et non de
    contenu. Le test ne porte que sur la conclusion : les deux formes sont au
    contraire des conditions parfaitement légitimes.

    (a) UNE SUBORDONNÉE MARQUÉE. « Puisque l'exposition est critique » énonce
        une prémisse ; une clause introduite par une conjonction de
        subordination ne peut pas être ce qu'on en conclut. C'est le miroir
        exact du court-circuit : là où lui dit « la subordonnée EST la
        condition », celui-ci dit « la subordonnée n'est PAS la conclusion ».

    (b) UN ADJOINT CONCESSIF. « même en hiver », « même à budget contraint » —
        « même » suivi d'une préposition introduit une circonstance concédée,
        sans verbe ni prédication. Rien n'y est affirmé, donc rien ne peut en
        être conclu.

    Mesuré sur le banc : 7 règles FAUSSES retirées, ZÉRO règle de référence
    détruite — la garantie est exacte, pas statistique. Le retournement de ces
    relations a été essayé d'abord et mesuré INUTILE : les trois cas de (a)
    sont des paires de paraphrases (« l'exposition est critique » contre « est
    jugée critique »), et aucune direction n'y est juste. On rejette donc au
    lieu de retourner.
    """
    if desactive():
        return True
    fragment = texte[debut:fin]
    if _CONDITION_TEXTUELLE.search(fragment):
        return False
    if (_CIRCONSTANCE_COURTE.search(fragment)
            and len(fragment.split()) <= 6
            and not _ARTICULATION.fullmatch(fragment)):
        return False
    traits = _traits_fragment(fragment, modele)
    if traits is None:
        return True
    jetons = [t for t in traits if not (t[0] or t[1])]
    if not jetons:
        return True
    # La DÉPENDANCE plutôt que la seule catégorie : spaCy étiquette « Comme »
    # tantôt SCONJ, tantôt ADP selon le contexte — incohérence qui a mordu ce
    # projet plusieurs fois. `dep_ == "mark"` dit la FONCTION du mot, et elle
    # ne varie pas : le segment est une subordonnée quelle que soit l'étiquette
    # de forme. Mesuré : une règle fausse de plus retirée (`45-telecom`), zéro
    # règle de référence détruite.
    if jetons[0][2] == "SCONJ" or jetons[0][3] == "mark":
        return False
    if (jetons[0][4] == "même" and len(jetons) > 1
            and jetons[1][2] == "ADP"):
        return False
    # (c) UNE CONJONCTION DE COORDINATION EN TÊTE. « car elles permettent une
    #     réaction plus rapide » : `car` introduit la RAISON de ce qui précède,
    #     donc une prémisse. `Mais si…` de même. Ces segments existent parce que
    #     la découpe typographique les isole ; ils ne sont pas autonomes.
    if jetons[0][2] == "CCONJ":
        return False
    # (d) UN PRONOM RELATIF EN TÊTE. « qui peut ensuite les retransmettre » est
    #     un modifieur, pas une proposition autonome : il n'affirme rien par
    #     lui-même. Le trait `PronType=Rel` distingue « qui » de « il », qui
    #     ouvre au contraire une conclusion parfaitement valide.
    if jetons[0][2] == "PRON" and "Rel" in jetons[0][5]:
        return False
    return True


# Particules d'EXCEPTION et de CONCESSION. Ce sont des mots grammaticaux, au
# même titre que les amorces de comparaison de `priorites.py` : ils marquent un
# rapport logique, pas un domaine.
#
# La version purement structurelle — n'importe quelle ADP ou ADV suivie d'un
# SCONJ — a été mesurée puis REFUSÉE : elle capture « dès que la communication
# le permet », qui est une condition de référence légitime. C'est la valeur de
# la particule qui compte, pas la structure.
_CONCESSIFS = ("sauf", "même", "hormis", "excepté", "quoique", "bien")


def peut_conditionner(texte: str, debut: int, fin: int,
                      modele: str = MODELE_DEFAUT) -> bool:
    """Ce segment peut-il être la CONDITION d'une règle ?

    Une subordonnée d'EXCEPTION ou de CONCESSION ne le peut pas. « sauf
    lorsqu'il constitue le seul moyen disponible » ne conditionne pas la règle,
    elle la DÉFAIT ; « même lorsque leur transmission entraîne une consommation
    importante » dit que la règle tient MALGRÉ cela. Dans les deux cas le
    formalisme l'exprime par une priorité, jamais par une prémisse — en faire
    une condition inverse le sens de la règle.

    Mesuré : 2 règles fausses retirées, ZÉRO condition de référence détruite.
    Aucune condition du banc ne s'ouvre sur ces particules.
    """
    if desactive():
        return True
    fragment = texte[debut:fin]
    traits = _traits_fragment(fragment, modele)
    if traits is None:
        return True
    jetons = [t for t in traits if not (t[0] or t[1])]
    if not jetons:
        return True
    # Un CONNECTEUR SEUL ne conditionne rien. « Toutefois » isolé en segment
    # par la ponctuation n'affirme aucun fait : c'est une articulation du
    # discours, pas une prémisse. Mesuré : une règle fausse retirée
    # (`48-drones-autonomes`), zéro condition de référence détruite — aucune
    # ne se réduit à un adverbe.
    if len(jetons) == 1 and jetons[0][2] in ("ADV", "CCONJ"):
        return False
    return not (len(jetons) > 1
                and jetons[0][4] in _CONCESSIFS
                and jetons[1][2] == "SCONJ")


@functools.lru_cache(maxsize=4096)
def points_de_coupure(texte: str, debut: int, fin: int,
                      modele: str = MODELE_DEFAUT) -> list[int]:
    """Positions où scinder un segment portant des subordonnées COORDONNÉES.

    « Comme la parcelle borde un monument classé ET QUE la hauteur projetée
    dépasse douze mètres, l'avis est requis » énonce DEUX conditions. La
    segmentation les gardait dans un seul segment, si bien que le pipeline ne
    pouvait produire qu'une entité là où la référence en attend deux.

    Mesuré sur les 15 règles de référence à plusieurs conditions : 9 avaient
    leurs conditions fusionnées, 1 seulement après cette découpe. La couverture
    des 268 spans de référence reste intégrale — la coupure n'ampute rien.

    Le segment est analysé ISOLÉMENT, pas dans son document : mesuré, spaCy
    rattache les coordonnées différemment selon l'empan qu'on lui donne, et
    l'analyse du document entier ne séparait que 4 cas sur 9. Analyser l'unité
    qu'on veut découper est à la fois plus fiable et plus économe.
    """
    if desactive():
        return []
    nlp = _charger(modele)
    if nlp is None:
        return []
    fragment = texte[debut:fin]
    points = []
    for token in nlp(fragment):
        if token.dep_ != "conj":
            continue
        if not any(enfant.dep_ == "mark" for enfant in token.children):
            continue
        sous_arbre = sorted(token.subtree, key=lambda t: t.idx)
        coordonnants = [e for e in token.children if e.dep_ == "cc"]
        # Couper avant le coordonnant (« et ») quand il précède la subordonnée,
        # afin qu'il ne reste pas en tête du second morceau.
        ouverture = min([e.idx for e in coordonnants] + [sous_arbre[0].idx])
        if 0 < ouverture < len(fragment):
            points.append(debut + ouverture)
    return sorted(set(points))


# Un subordonnant coupé de la proposition qu'il introduit : « Si, et dans la
# mesure où », « si, en violation du présent règlement », « Si nécessaire ».
_SUBORDONNANT = re.compile(
    r"\b(?:si|s['’]il|s['’]ils|lorsque|lorsqu['’]|quand|où|dès\s+que|"
    r"à\s+moins\s+que|pour\s+autant\s+que)\b", re.I)

# Longueur au-delà de laquelle un segment n'est plus un connecteur tronqué
# mais une proposition que l'analyseur a mal étiquetée. Mesuré : la clause
# « lorsqu'un ordre provenant de la station de contrôle ENTRE en contradiction
# … » (120 caractères) est une CONDITION DE RÉFÉRENCE que spaCy dépouille de
# son verbe — il prend « entre » pour la préposition. Les connecteurs tronqués
# réellement observés font 13 à 37 caractères. Le plafond les sépare sans
# ambiguïté et rend le mécanisme inerte sur le banc.
_LONGUEUR_CONNECTEUR = 48


def connecteurs_tronques(texte: str, segments,
                         modele: str = MODELE_DEFAUT) -> set[int]:
    """Indices des segments qui ne sont qu'un subordonnant sans sa proposition.

    « Si, et dans la mesure où, il n'est pas possible de … » : la virgule
    interne au connecteur déclenche la découpe, et « Si, et dans la mesure où »
    devient un segment à part entière. Étiqueté, il fabrique une entité fausse,
    puis une règle fausse — et il PRIVE la proposition suivante du subordonnant
    qui portait sa conditionnalité.

    Le critère est grammatical : le segment contient un subordonnant, mais NI
    sujet NI verbe conjugué. Il n'y a donc pas de prédication : ce n'est pas
    une proposition.

    L'analyse porte sur le DOCUMENT ENTIER, une seule fois, et non sur chaque
    fragment isolé — mesuré, « Quand le fournisseur CHANGE de recette » perd
    son verbe dès qu'on l'analyse seul (« change » devient un nom), et le
    mécanisme aurait amputé une condition de référence du banc.

    Mesuré avant d'être écrit : 0 segment touché sur les 48 cas du banc, 0 sur
    les 32 récits négatifs, 0 sur le premier lot hors échantillon, 4 sur le
    second — tous des fragments qui ne recouvrent AUCUN empan de référence.
    """
    # `LPP_SANS_CONNECTEUR=1` rend le mécanisme inerte, pour la comparaison
    # contrôlée : l'inférence n'étant pas reproductible, un seul tirage de
    # chaque côté ne tranche rien, et changer de version de code mêlerait
    # deux causes.
    if desactive() or os.environ.get("LPP_SANS_CONNECTEUR", "") not in ("", "0"):
        return set()
    nlp = _charger(modele)
    if nlp is None:
        return set()
    document = None
    trouves: set[int] = set()
    for rang, (debut, fin) in enumerate(segments):
        if fin - debut > _LONGUEUR_CONNECTEUR:
            continue
        if not _SUBORDONNANT.search(texte, debut, fin):
            continue
        if document is None:
            document = nlp(texte)
        jetons = [j for j in document
                  if j.idx >= debut and j.idx + len(j.text) <= fin]
        if any(j.dep_ in ("nsubj", "nsubj:pass", "expl:subj") for j in jetons):
            continue
        if any(j.pos_ in ("VERB", "AUX") and "Fin" in j.morph.get("VerbForm")
               for j in jetons):
            continue
        trouves.add(rang)
    return trouves
