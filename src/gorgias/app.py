from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path

from . import predicats
from . import priorites
from . import singularite
from . import syntaxe
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TEXT_BOUND_LINE = re.compile(
    r"^(T\d+)\t(Context|Option|Marker) (\d+) (\d+)\t(.*)$"
)
_IMPLICIT_ATTRIBUTE = re.compile(r"^A\d+\tImplicit (T\d+) True$")
_EVENT_LINE = re.compile(
    r"^(E\d+)\t(rule|prefer|meta_prefer):(T\d+)((?: [A-Za-z]+:[TE]\d+)*)$"
)
_ATTRIBUTE_LINE = re.compile(
    r"^(A\d+)\t(Modality|Negated|Implicit) (T\d+)(?: (\S+))?$"
)
_ROLE = re.compile(r"([A-Za-z]+):([TE]\d+)")


def _build_accent_classes() -> dict[str, str]:
    """Map each ASCII letter to every Latin letter sharing it as base form."""
    groups: dict[str, set[str]] = {}
    for code_point in range(0x41, 0x250):
        character = chr(code_point)
        if not character.isalpha():
            continue
        base = unicodedata.normalize("NFD", character)[0].lower()
        if len(base) == 1 and base.isascii() and base.isalpha():
            groups.setdefault(base, {base}).add(character.lower())
    return {base: "".join(sorted(variants)) for base, variants in groups.items()}


_ACCENT_CLASSES = _build_accent_classes()


def _character_pattern(character: str) -> str:
    """Return a pattern tolerant to the accent and quote variants of a model."""
    if character in "'’‘`":
        return "['’‘`]"
    if character in '"“”':
        return '["“”]'
    if character in "-‐‑‒–—−":
        return "[-‐‑‒–—−]"

    base = unicodedata.normalize("NFD", character)[0].lower()
    variants = _ACCENT_CLASSES.get(base)
    if variants and len(variants) > 1:
        return f"[{variants}]"
    return re.escape(character)


def _span_candidates(source_text: str, annotated_span: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    start = source_text.find(annotated_span)
    while start != -1:
        candidates.append((start, start + len(annotated_span)))
        start = source_text.find(annotated_span, start + 1)
    if candidates:
        return candidates

    pattern_parts: list[str] = []
    previous_was_space = False
    for character in annotated_span:
        if character in " \t":
            if not previous_was_space:
                pattern_parts.append(r"[ \t]+")
            previous_was_space = True
            continue

        previous_was_space = False
        pattern_parts.append(_character_pattern(character))

    if not pattern_parts:
        return []

    flexible_pattern = re.compile("".join(pattern_parts), re.IGNORECASE)
    candidates = [
        (match.start(), match.end())
        for match in flexible_pattern.finditer(source_text)
    ]
    if candidates:
        return candidates

    return _fuzzy_span_candidates(source_text, annotated_span)


_WORD = re.compile(r"\w+", re.UNICODE)

# Un span reformulé n'est accepté que si l'alignement reste très proche du
# texte source : ces seuils sont volontairement stricts, un span mal aligné
# étant bien pire qu'un échec visible.
_MIN_FUZZY_WORDS = 5
_MIN_WORD_COVERAGE = 0.8
_MIN_LENGTH_RATIO = 0.7
_MAX_LENGTH_RATIO = 1.4


def _fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _fuzzy_span_candidates(
    source_text: str, annotated_span: str
) -> list[tuple[int, int]]:
    """Locate a span the model paraphrased slightly, or refuse to guess.

    Last-resort tier: it only fires once the exact and accent-tolerant
    searches have failed. It aligns the span word by word onto the source and
    gives up unless the alignment covers almost every word, so a genuinely
    invented span still raises rather than silently landing somewhere wrong.
    """
    source_words = [
        (match.start(), match.end(), _fold(match.group()))
        for match in _WORD.finditer(source_text)
    ]
    span_words = [_fold(match.group()) for match in _WORD.finditer(annotated_span)]

    if len(span_words) < _MIN_FUZZY_WORDS or not source_words:
        return []

    matcher = SequenceMatcher(
        None, [word for _, _, word in source_words], span_words, autojunk=False
    )
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return []

    matched_words = sum(block.size for block in blocks)
    if matched_words / len(span_words) < _MIN_WORD_COVERAGE:
        return []

    first_word = blocks[0].a
    last_word = blocks[-1].a + blocks[-1].size - 1
    covered_words = last_word - first_word + 1
    if not (
        _MIN_LENGTH_RATIO * len(span_words)
        <= covered_words
        <= _MAX_LENGTH_RATIO * len(span_words)
    ):
        return []

    return [(source_words[first_word][0], source_words[last_word][1])]


def repair_annotation_offsets(annotations: str, source_text: str) -> str:
    """Repair text-bound offsets and guarantee exact brat source spans."""
    lines = annotations.splitlines()
    implicit_entities = {
        match.group(1)
        for line in lines
        if (match := _IMPLICIT_ATTRIBUTE.fullmatch(line))
    }
    repaired_lines: list[str] = []

    for line in lines:
        if not line.startswith("T"):
            repaired_lines.append(line)
            continue

        match = _TEXT_BOUND_LINE.fullmatch(line)
        if not match:
            raise ValueError(f"annotation textuelle brat invalide : {line!r}")

        entity_id, entity_type, raw_start, raw_end, raw_span = match.groups()
        claimed_start = int(raw_start)
        claimed_end = int(raw_end)
        annotated_span = raw_span.rstrip()

        if entity_id in implicit_entities:
            repaired_lines.append(line)
            continue
        if not annotated_span:
            raise ValueError(f"le segment {entity_id} est vide")

        if (
            0 <= claimed_start <= claimed_end <= len(source_text)
            and source_text[claimed_start:claimed_end] == annotated_span
        ):
            actual_start, actual_end = claimed_start, claimed_end
            exact_span = annotated_span
        else:
            candidates = _span_candidates(source_text, annotated_span)
            if not candidates:
                raise ValueError(
                    f"le segment de {entity_id} est introuvable dans le texte source : "
                    f"{annotated_span!r}"
                )
            actual_start, actual_end = min(
                candidates,
                key=lambda offsets: abs(offsets[0] - claimed_start),
            )
            exact_span = source_text[actual_start:actual_end]

        repaired_lines.append(
            f"{entity_id}\t{entity_type} {actual_start} {actual_end}\t{exact_span}"
        )

    return "\n".join(repaired_lines)


def validate_annotation_graph(annotations: str) -> None:
    """Reject annotations whose events reference missing or ill-typed ids."""
    entity_types: dict[str, str] = {}
    event_kinds: dict[str, str] = {}
    events: list[tuple[str, str, str, list[tuple[str, str]]]] = []
    attributes: list[tuple[str, str]] = []

    for line in annotations.splitlines():
        if line.startswith("T"):
            match = _TEXT_BOUND_LINE.fullmatch(line)
            if not match:
                raise ValueError(f"ligne T invalide : {line!r}")
            entity_id, entity_type = match.group(1), match.group(2)
            if entity_id in entity_types:
                raise ValueError(
                    f"l'identifiant {entity_id} est utilisé par deux segments "
                    "différents ; numérote chaque annotation séparément"
                )
            entity_types[entity_id] = entity_type
        elif line.startswith("E"):
            match = _EVENT_LINE.fullmatch(line)
            if not match:
                misplaced = re.match(r"^E\d+\t(Context|Option|Marker) ", line)
                if misplaced:
                    raise ValueError(
                        f"{line.split(chr(9))[0]} porte le type "
                        f"{misplaced.group(1)}, qui est un type d'entité : il "
                        "doit être sur une ligne T, pas E. Les lignes E ne "
                        "portent que rule, prefer ou meta_prefer"
                    )
                raise ValueError(f"ligne E invalide : {line!r}")
            event_id, kind, trigger, roles = match.groups()
            if event_id in event_kinds:
                raise ValueError(
                    f"l'identifiant {event_id} est utilisé par deux événements"
                )
            event_kinds[event_id] = kind
            events.append((event_id, kind, trigger, _ROLE.findall(roles)))
        else:
            match = _ATTRIBUTE_LINE.fullmatch(line)
            if not match:
                raise ValueError(f"ligne A invalide : {line!r}")
            attributes.append((match.group(1), match.group(3)))

    if "Option" not in entity_types.values():
        raise ValueError(
            "aucune entité de type Option : le texte doit contenir au moins une "
            "conclusion"
        )

    for attribute_id, target in attributes:
        if target not in entity_types:
            raise ValueError(f"{attribute_id} référence {target} qui n'existe pas")

    for event_id, kind, trigger, roles in events:
        if trigger not in entity_types:
            raise ValueError(f"{event_id} référence {trigger} qui n'existe pas")

        for role, target in roles:
            if target == event_id:
                raise ValueError(
                    f"{event_id} se référence lui-même via {role}: ; un événement "
                    "ne peut désigner que des événements déjà définis avant lui"
                )
            if target.startswith("T") and target not in entity_types:
                raise ValueError(
                    f"{event_id} référence {target} qui n'existe pas"
                )
            if target.startswith("E") and target not in event_kinds:
                raise ValueError(
                    f"{event_id} référence l'événement {target} qui n'existe pas ; "
                    "n'utilise que des événements déjà définis"
                )

        expected_trigger = "Option" if kind == "rule" else "Marker"
        if entity_types[trigger] != expected_trigger:
            raise ValueError(
                f"{event_id} est un {kind} : son déclencheur {trigger} doit être "
                f"de type {expected_trigger}, or il est de type "
                f"{entity_types[trigger]}"
            )

        role_names = [role for role, _ in roles]
        if kind == "rule":
            if role_names.count("Effect") != 1:
                raise ValueError(f"{event_id} doit avoir exactement un Effect")
            if role_names.count("Condition") < 1:
                raise ValueError(f"{event_id} doit avoir au moins un Condition")
            for role, target in roles:
                if role == "Effect" and entity_types[target] != "Option":
                    raise ValueError(
                        f"{event_id} : l'Effect {target} doit être une Option, or "
                        f"il est de type {entity_types[target]}"
                    )
                if role == "Condition" and entity_types[target] == "Marker":
                    raise ValueError(
                        f"{event_id} : la Condition {target} ne peut pas être un "
                        "Marker"
                    )
            continue

        for role, target in roles:
            if role == "When" and entity_types.get(target) != "Context":
                raise ValueError(
                    f"{event_id} : le When {target} doit être un Context, or il "
                    f"est de type {entity_types.get(target)}"
                )

        ranked = {role: target for role, target in roles if role in {"Winner", "Loser"}}
        if len(ranked) != 2:
            raise ValueError(f"{event_id} doit avoir un Winner et un Loser")
        if ranked["Winner"] == ranked["Loser"]:
            raise ValueError(
                f"{event_id} classe {ranked['Winner']} contre lui-même"
            )

        expected = {"rule"} if kind == "prefer" else {"prefer", "meta_prefer"}
        for role, target in ranked.items():
            actual = event_kinds.get(target, "")
            if actual not in expected:
                raise ValueError(
                    f"{event_id} est un {kind} : son {role} doit être un événement "
                    f"{' ou '.join(sorted(expected))}, or {target} est un {actual}"
                )


# ============================================================================
# Théorie commune à toutes les étapes
#
# Formulation neutre du problème de décision tel que le définissent Kakas,
# Moraïtis & Spanoudakis, « Gorgias: Applying Argumentation » (2018), § 2 :
# OPTIONS, SCENARIO INFORMATION, SCENARIO-BASED PREFERENCES. Le niveau d'une
# priorité y est la profondeur de raffinement du scénario (§ 3.1, algorithme 1)
# et non un palier fixé d'avance : c'est le code, pas le modèle, qui le calcule.
# ============================================================================

# Noyau théorique envoyé en tête de CHAQUE appel : il est donc tenu court, et
# le détail opératoire vit dans les instructions de chaque étape.
_THEORY = """You analyse the reasoning structure of a text.

Any text that reasons describes a DECISION PROBLEM made of three things:
OPTIONS (what the reasoning concludes: decisions, verdicts, positions),
SCENARIO INFORMATION (what it depends on: facts, conditions, circumstances),
and PREFERENCES between options in a given situation.

Adding information to a situation refines it. When a text refines a situation
and thereby changes which option wins, it states a priority; the depth of the
refinement gives its level. You never compute levels and never write
identifiers: they are derived afterwards from what you report.

Annotate only what takes part in the inference. Identifying and administrative
material, headings and pure narration take part in none.
"""

SEED = 42

# Nombre de tirages par appel. `seed` et `top_k=1` ne suffisent pas à rendre
# l'inférence reproductible : mesuré, un appel sur six rend une réponse
# différente à prompt strictement identique — deux logits presque égaux
# s'inversent. Au-delà de 1, on vote séparément sur chaque étiquette et chaque
# relation, ce qui rend le résultat plus stable au prix du nombre d'appels.
VOTES = 1

# Journal des appels : quatre empreintes par appel permettent d'isoler la
# couche qui varie entre deux exécutions — entrée, sérialisation du prompt,
# réponse du modèle, ou interprétation de cette réponse.
TRACE: list[dict] = []
# Compteur d'appels au LLM, lu par le harnais de mesure. Un appel = une
# invocation effective, tentatives de reprise comprises.
COMPTEUR_APPELS: dict[str, int] = {"total": 0}

TRACE_ENABLED = False


def _empreinte(valeur) -> str:
    if not isinstance(valeur, str):
        valeur = json.dumps(valeur, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"))
    return hashlib.sha256(valeur.encode("utf-8")).hexdigest()[:16]


def _clean_json_payload(raw: str) -> str:
    without_think = _THINK_BLOCK.sub("", raw).strip()
    blocks = re.findall(
        r"\x60{3}(?:json)?[ \t]*\r?\n(.*?)\x60{3}",
        without_think,
        re.DOTALL | re.IGNORECASE,
    )
    if blocks:
        without_think = blocks[0].strip()
    start = without_think.find("{")
    end = without_think.rfind("}")
    if start != -1 and end > start:
        return without_think[start : end + 1]
    return without_think


class StageError(ValueError):
    """Une étape n'a pas produit de résultat exploitable."""



def _vote_majoritaire(valeurs, repli=None):
    """Vote sur une décision atomique, jamais sur une réponse JSON entière."""
    if not valeurs:
        return repli
    comptes = Counter(valeurs)
    valeur, nombre = comptes.most_common(1)[0]
    return valeur if nombre > len(valeurs) / 2 else repli


def _vote_stage(key: str, bulletins: list[object]) -> object:
    """Fusionne des réponses interprétées décision par décision.

    Deux JSON sémantiquement identiques peuvent différer par l'ordre des
    champs, et un seul élément différent rendait auparavant tout le bulletin
    différent. Le vote porte maintenant sur une étiquette, une relation ou
    une ligne de préférence à la fois.
    """
    if len(bulletins) == 1:
        return bulletins[0]

    if key == "labels":
        numeros = sorted({n for b in bulletins for n in b})
        sortie = {}
        for numero in numeros:
            marques = set()
            for marque in ("P", "M"):
                voix = [marque in b.get(numero, frozenset()) for b in bulletins]
                if sum(voix) > len(voix) / 2:
                    marques.add(marque)
            sortie[numero] = frozenset(marques or {"N"})
        return sortie

    if key == "support":
        conclusions = sorted({n for b in bulletins for n in b})
        sortie = {}
        for conclusion in conclusions:
            candidats = sorted(
                {g for b in bulletins for g in b.get(conclusion, [])}
            )
            sortie[conclusion] = [
                ground
                for ground in candidats
                if sum(ground in b.get(conclusion, []) for b in bulletins)
                > len(bulletins) / 2
            ]
        return sortie

    if key in {"relations", "narrative"}:
        numeros = sorted({n for b in bulletins for n in b})
        repli = "NONE" if key == "relations" else "Y"
        return {
            numero: _vote_majoritaire(
                [b[numero] for b in bulletins if numero in b], repli
            )
            for numero in numeros
        }

    if key == "enumerations":
        numeros = sorted({n for b in bulletins for n in b})
        sortie = {}
        for numero in numeros:
            valeur = _vote_majoritaire(
                [tuple(b.get(numero, [])) for b in bulletins], ()
            )
            if valeur:
                sortie[numero] = list(valeur)
        return sortie

    if key == "preferences":
        normalises = [
            {
                json.dumps(ligne, ensure_ascii=False, sort_keys=True): ligne
                for ligne in bulletin
            }
            for bulletin in bulletins
        ]
        toutes = sorted({signature for b in normalises for signature in b})
        return [
            next(b[signature] for b in normalises if signature in b)
            for signature in toutes
            if sum(signature in b for b in normalises) > len(normalises) / 2
        ]

    return _vote_majoritaire(
        [json.dumps(b, ensure_ascii=False, sort_keys=True) for b in bulletins]
    )


def _realigner(resultat: dict, attendus: set[int]) -> dict:
    """Répare la RENUMÉROTATION d'un lot par le modèle.

    Les segments sont présentés avec leurs numéros du document ; le modèle
    recommence pourtant souvent la numérotation à 1. Mesuré sur
    `09-raffinement-explicite` : présenté le seul segment
    « 9. même en hiver », le modèle répond `{"n":1,…}`. Le contrôle de complétude
    rejette la réponse, les trois reprises échouent de façon identique, et
    TOUT le lot d'étiquettes est perdu — sur ce document, la totalité de
    l'étape 1, soit trois appels de pure perte.

    La réparation n'est tentée que lorsqu'elle est DÉMONTRABLE, jamais
    devinée : autant d'entrées que de numéros demandés, et des clés qui
    forment exactement 1..k. Les items étant présentés dans l'ordre des
    numéros demandés, l'appariement positionnel est alors exact. Toute autre
    divergence est laissée à l'échec, qui reste le comportement sûr.
    """
    if not attendus or set(resultat) == attendus:
        return resultat
    obtenues = sorted(resultat)
    if len(obtenues) != len(attendus):
        return resultat
    if obtenues != list(range(1, len(obtenues) + 1)):
        return resultat
    return dict(zip(sorted(attendus), (resultat[cle] for cle in obtenues)))


def _ask_stage(
    model: Any,
    instructions: str,
    payload: str,
    key: str,
    interpret,
    attempts: int = 3,
    expected: Sequence[int] | None = None,
    partiel: bool = False,
) -> object:
    """Produit un ou plusieurs bulletins complets, puis vote atomiquement.

    `partiel` autorise, en dernier recours, un bulletin INCOMPLET plutôt que
    rien. Une réponse à laquelle il manque un numéro faisait jusqu'ici perdre
    le lot entier : dix paires jugées, une entrée absente, dix verdicts jetés.
    Là où une entrée manquante équivaut à « aucune information sur ce
    numéro » — étiquettes, rattachements, orientation des paires — garder les
    autres est strictement meilleur que tout perdre. Les appelants pour qui
    l'absence de verdict a un sens conservateur (le filtre narratif) doivent
    traiter les numéros absents eux-mêmes.
    """
    bulletins = []
    erreurs = []
    attendus = set(expected or ())

    for _ in range(max(1, VOTES)):
        produit = False
        meilleur: tuple[set[int], dict] | None = None
        messages: list = [
            SystemMessage(content=f"{_THEORY}\n\n{instructions}"),
            HumanMessage(content=payload),
        ]
        last_error = "aucune réponse"
        for attempt in range(attempts):
            COMPTEUR_APPELS["total"] += 1
            COMPTEUR_APPELS[key] = COMPTEUR_APPELS.get(key, 0) + 1
            response = model.invoke(messages)
            if TRACE_ENABLED:
                TRACE.append(
                    {
                        "cle": key,
                        "entree": _empreinte(payload),
                        "prompt": _empreinte(
                            "\u0000".join(m.content for m in messages)
                        ),
                        "reponse": _empreinte(response.content),
                    }
                )
            try:
                document = json.loads(_clean_json_payload(response.content))
                if not isinstance(document, dict) or key not in document:
                    raise ValueError(f"la clé JSON attendue est \"{key}\"")
                resultat = interpret(document[key])
                if attendus and isinstance(resultat, dict):
                    resultat = _realigner(resultat, attendus)
                    couverts = attendus & set(resultat)
                    if meilleur is None or len(couverts) > len(meilleur[0]):
                        meilleur = (couverts, resultat)
                    manquants = sorted(attendus - set(resultat))
                    if manquants:
                        raise ValueError(
                            "réponse incomplète, numéros manquants : "
                            + ", ".join(map(str, manquants))
                        )
                if TRACE_ENABLED:
                    TRACE[-1]["interpretation"] = _empreinte(repr(resultat))
                bulletins.append(resultat)
                produit = True
                break
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as error:
                last_error = str(error)
                if attempt == attempts - 1:
                    break
                messages.extend(
                    [
                        response,
                        HumanMessage(
                            content=(
                                f"Your answer was rejected: {last_error}\n"
                                "Answer again with JSON only, no prose and no "
                                f"code fence, with the key \"{key}\". Include "
                                "one entry for every requested number and copy "
                                "every span character for character from the "
                                "text sent above."
                            )
                        ),
                    ]
                )
        else:
            last_error = "aucune tentative exécutée"
        if not produit and partiel and meilleur is not None and meilleur[0]:
            bulletins.append({numero: valeur
                              for numero, valeur in meilleur[1].items()
                              if numero in attendus})
            produit = True
        if not produit:
            erreurs.append(last_error)

    quorum = max(1, VOTES // 2 + 1)
    if len(bulletins) < quorum:
        detail = "; ".join(erreurs) or "pas assez de bulletins complets"
        raise StageError(
            f"{len(bulletins)}/{VOTES} bulletin(s) exploitable(s), "
            f"quorum {quorum} requis : {detail}"
        )
    return _vote_stage(key, bulletins)


# ============================================================================
# Résolution des spans dans le texte source
# ============================================================================


_SPAN_EDGES = " \t\r\n.,;:!?«»\"'“”‘’()[]"


def locate_span(source_text: str, span: str, offset: int = 0) -> tuple[int, int] | None:
    """Renvoie les bornes exactes du span dans la source, ou None.

    La ponctuation de bord est retirée avant la recherche : sans cela deux
    spans identiques dont l'un porte le point final produisent deux entités
    distinctes au même endroit.
    """
    cleaned = span.strip().strip(_SPAN_EDGES)
    if not cleaned:
        return None
    candidates = _span_candidates(source_text, cleaned)
    if not candidates:
        return None
    start, end = candidates[0]
    return start + offset, end + offset


# ============================================================================
# Assemblage : le code attribue tous les identifiants et tous les niveaux
# ============================================================================


class Assembler:
    """Construit un .ann brat valide à partir de spans et de relations.

    Le modèle ne produit jamais d'identifiant : c'est cette classe qui les
    attribue, ce qui rend structurellement impossibles les références
    pendantes, les auto-références et les confusions entre numéros T et E.
    """

    def __init__(self, source_text: str) -> None:
        self._source = source_text
        self._entities: dict[tuple[int, int, str], None] = {}
        self._events: list[dict] = []
        self._complements: set[
            tuple[tuple[int, int, str], tuple[int, int, str]]
        ] = set()
        self._diagnostics: list[str] = []
        self._attributes: list[tuple[str, tuple[int, int, str], str]] = []

    def entity(self, kind: str, bounds: tuple[int, int]) -> tuple[int, int, str]:
        key = (bounds[0], bounds[1], kind)
        self._entities.setdefault(key, None)
        return key

    def modality(self, option_key: tuple[int, int, str], value: str) -> None:
        self._attributes.append(("Modality", option_key, value))

    def rule(self, option_key, condition_keys) -> int:
        roles = [("Condition", ("T", key)) for key in condition_keys]
        roles.append(("Effect", ("T", option_key)))
        self._events.append(
            {"kind": "rule", "trigger": option_key, "roles": roles}
        )
        return len(self._events) - 1

    def complement(self, first_key, second_key) -> None:
        """Déclare deux options incompatibles, sans orienter leur préférence."""
        if (
            first_key == second_key
            or first_key[2] != "Option"
            or second_key[2] != "Option"
        ):
            return
        first, second = sorted(
            (first_key, second_key), key=lambda key: (key[0], key[1], key[2])
        )
        self._complements.add((first, second))

    def diagnostic(self, message: str) -> None:
        self._diagnostics.append(message.replace("\n", " ").strip())

    def priority(self, kind, marker_key, winner, loser, when_key) -> int:
        """`when_key` accepte un contexte, ou plusieurs.

        Le scénario d'une préférence est souvent une CONJONCTION : dans
        « Quand le sol est gelé et que la pluie tombe, le sable est préféré au
        sel », la priorité ne vaut que si les deux conditions tiennent. N'en
        retenir qu'une la déclenche trop largement ; fabriquer un contexte
        neuf qui les redit produit un atome orphelin que rien n'active — la
        préférence ne se déclenche alors JAMAIS, ce qu'a révélé la comparaison
        par modèles stables sur `11-combinaison`.
        """
        roles = [("Winner", ("E", winner)), ("Loser", ("E", loser))]
        cles = ([when_key] if isinstance(when_key, tuple) and len(when_key) == 3
                else list(when_key or ()))
        for cle in cles:
            if cle is not None:
                roles.append(("When", ("T", cle)))
        self._events.append(
            {"kind": kind, "trigger": marker_key, "roles": roles}
        )
        return len(self._events) - 1

    def render(self) -> str:
        # Un Marker n'a de raison d'être que s'il déclenche une priorité. Celui
        # qu'aucun événement n'utilise est un faux positif de l'étape 3 : on le
        # retire plutôt que de livrer une entité orpheline.
        triggers = {event["trigger"] for event in self._events}
        for key in [k for k in self._entities if k[2] == "Marker" and k not in triggers]:
            del self._entities[key]

        order = sorted(self._entities, key=lambda key: (key[0], key[1], key[2]))
        tid = {key: f"T{index + 1}" for index, key in enumerate(order)}

        ranking = {"rule": 0, "prefer": 1, "meta_prefer": 2}
        event_order = sorted(
            range(len(self._events)),
            key=lambda index: (ranking[self._events[index]["kind"]], index),
        )
        eid = {index: f"E{rank + 1}" for rank, index in enumerate(event_order)}

        lines = [
            f"{tid[key]}\t{key[2]} {key[0]} {key[1]}\t{self._source[key[0]:key[1]]}"
            for key in order
        ]

        for index in event_order:
            event = self._events[index]
            parts = [f"{event['kind']}:{tid[event['trigger']]}"]
            for role, (space, target) in event["roles"]:
                parts.append(
                    f"{role}:{tid[target] if space == 'T' else eid[target]}"
                )
            lines.append(f"{eid[index]}\t{' '.join(parts)}")

        for number, (name, key, value) in enumerate(self._attributes, start=1):
            lines.append(f"A{number}\t{name} {tid[key]} {value}")

        return "\n".join(lines)

    def render_lpp(self) -> str:
        """Rend les composants LPP/Gorgias dans un format canonique.

        Les noms, références et produits cartésiens sont générés ici. Le
        modèle n'écrit donc ni identifiant LPP, ni relation ``complement``.
        """
        entity_order = sorted(
            (key for key in self._entities if key[2] in {"Context", "Option"}),
            key=lambda key: (key[0], key[1], key[2]),
        )
        counters = {"Context": 0, "Option": 0}
        literal_ids = {}
        prefixes = {"Context": "c", "Option": "o"}
        for key in entity_order:
            counters[key[2]] += 1
            literal_ids[key] = f"{prefixes[key[2]]}{counters[key[2]]}"

        event_ids = {}
        event_counters = {"rule": 0, "prefer": 0, "meta_prefer": 0}
        event_prefixes = {"rule": "r", "prefer": "p", "meta_prefer": "mp"}
        ranking = {"rule": 0, "prefer": 1, "meta_prefer": 2}
        event_order = sorted(
            range(len(self._events)),
            key=lambda index: (ranking[self._events[index]["kind"]], index),
        )
        for index in event_order:
            kind = self._events[index]["kind"]
            event_counters[kind] += 1
            event_ids[index] = f"{event_prefixes[kind]}{event_counters[kind]}"

        lines = [
            "% Composants LPP/Gorgias normalisés",
            "% Les libellés conservent le texte source exact.",
        ]
        lines.extend(f"% {message}" for message in self._diagnostics if message)
        for key in entity_order:
            predicate = "scenario" if key[2] == "Context" else "option"
            label = json.dumps(
                self._source[key[0]:key[1]], ensure_ascii=False
            )
            # La forme logique normalisée s'AJOUTE au libellé source, elle ne
            # le remplace pas : les références sont annotées par empans de
            # caractères et toute la mesure repose sur eux. Elle rend
            # l'unification possible — deux mentions du même fait convergent
            # vers le même littéral — sans rien casser en aval.
            forme = predicats.forme_logique(self._source[key[0]:key[1]])
            if forme:
                lines.append(
                    f"{predicate}({literal_ids[key]}, {forme}, {label})."
                )
            else:
                lines.append(f"{predicate}({literal_ids[key]}, {label}).")

        for first, second in sorted(
            self._complements,
            key=lambda pair: (literal_ids.get(pair[0], ""), literal_ids.get(pair[1], "")),
        ):
            if first in literal_ids and second in literal_ids:
                lines.append(
                    f"complement({literal_ids[first]}, {literal_ids[second]})."
                )

        for index in event_order:
            event = self._events[index]
            roles = event["roles"]
            when = [
                literal_ids[target]
                for role, (space, target) in roles
                if role == "When" and space == "T" and target in literal_ids
            ]
            if event["kind"] == "rule":
                conditions = [
                    literal_ids[target]
                    for role, (space, target) in roles
                    if role == "Condition" and space == "T" and target in literal_ids
                ]
                effect = literal_ids[event["trigger"]]
                lines.append(
                    f"rule({event_ids[index]}, [{', '.join(conditions)}], {effect})."
                )
                continue

            ranked = {
                role: event_ids[target]
                for role, (space, target) in roles
                if role in {"Winner", "Loser"} and space == "E" and target in event_ids
            }
            predicate = event["kind"]
            lines.append(
                f"{predicate}({event_ids[index]}, [{', '.join(when)}], "
                f"{ranked['Winner']}, {ranked['Loser']})."
            )
        return "\n".join(lines)




# ============================================================================
# Étape 0 — découpage en propositions, sans modèle
#
# Le texte est découpé en Python avant tout appel. Le modèle ne produit donc
# plus de spans : il choisit une étiquette pour des segments qui existent déjà.
# Trois conséquences : un span ne peut plus être inventé, la couverture du
# document est garantie, et la tâche devient une classification fermée — ce
# qu'un petit modèle réussit bien mieux qu'une extraction libre.
# ============================================================================

_HARD_BREAK = re.compile(r"\n+")
_SENTENCE_BREAK = re.compile(r"([.!?;])([\"'»)\]]*)(\s+)")
_SOFT_BREAK = re.compile(r"([,:])(\s+)")
_INLINE_CONDITION_BREAK = re.compile(
    r"\b(?:et\s+à\s+la\s+condition\s+que|"
    r"que\s+s['’](?:il|elle|on|ils|elles)|que\s+si|"
    r"s['’](?:il|elle|on|ils|elles)|si|quand|lorsque|dès\s+que|"
    r"only\s+if|if|when|parce\s+que|because|provided\s+that|"
    r"à\s+(?=qui\b))\b",
    re.IGNORECASE,
)
_NOMINAL_SUFFIX_BREAK = re.compile(
    r"(?<!\w)(?:à\s+l['’]échéance\b|en\s+cas\s+de\b)", re.IGNORECASE
)
_EXTENSION_DE_PORTEE = re.compile(r"\by\s+compris\s*$", re.IGNORECASE)
_THEN_BREAK = re.compile(r"\b(?:alors|then)\b", re.IGNORECASE)
_AND_WITHOUT_BREAK = re.compile(r"\bet\s+(?=sans\b)", re.IGNORECASE)
_CONDITIONAL_START = re.compile(
    r"^\s*(?:mais\s+)?(?:si|s['’]|if|quand|lorsque|when)\b",
    re.IGNORECASE,
)
_ABBREVIATION = re.compile(
    r"(?:\b\w|\b\d+|\bno|\bart|\bal|\bp|\bpp|\bcf|\bvs)\.?$", re.IGNORECASE
)

# Seuil calibré contre l'annotation de référence : au-delà, condition et
# conclusion restent collées dans un même segment et aucune règle ne peut se
# construire ; en deçà, on fragmente des propositions que la référence garde
# entières. À 80, 22 des 27 spans de référence restent couverts à 90 % par un
# seul segment. Le découpage se fait sur la ponctuation seule : une liste de
# connecteurs a été essayée et dégradait la couverture (22 -> 18).
# Deux seuils en interaction : MIN_CLAUSE décide OÙ couper, MIN_SEGMENT ce
# qu'on GARDE. Mal réglés, ils se combattent — un contexte court comme
# « En hiver » (8 car.) était correctement détaché puis jeté. Calibrés
# conjointement sur les références : le plafond de rappel passe de 81 % à
# 90 %, pour 65 segments au lieu de 55 sur le benchmark.
MIN_CLAUSE = 6
MIN_SEGMENT = 6


# Longueur du fragment soumis à l'analyse grammaticale quand une ponctuation
# forte est suivie d'une minuscule. Une clause française tient largement dans
# cette fenêtre, et la borner garde le coût de l'analyse constant.
_FENETRE_DE_CLAUSE = 160


def _is_real_break(text: str, end: int, after: int) -> bool:
    if _ABBREVIATION.search(text[:end].rstrip()):
        return False
    tail = text[after : after + 2].lstrip()
    if not tail or not tail[0].islower():
        return True
    # Ponctuation forte suivie d'une MINUSCULE. La règle de surface refusait
    # la frontière, ce qui protège « etc. le reste » mais recolle deux phrases
    # entières dès qu'un texte n'ouvre pas ses phrases par une capitale. On
    # tranche donc grammaticalement : la frontière est réelle si ce qui suit
    # porte un sujet et un verbe fini propres.
    #
    # Le cas est INEXISTANT sur du texte réel — zéro occurrence dans les 48
    # cas, les 32 récits et les trois lots RGPD exploités — donc l'analyse ne
    # s'exécute jamais sur eux et le comportement mesuré y est inchangé. Sur
    # `data/synthetique`, elle récupère 58 frontières de phrase.
    suite = text[after : after + _FENETRE_DE_CLAUSE]
    coupure = min(
        [position for position in (suite.find("."), suite.find("\n"))
         if position != -1] or [len(suite)]
    )
    verdict = syntaxe.ouvre_une_clause(suite[:coupure].strip())
    return bool(verdict)


def _split_soft(text: str, start: int, end: int,
                ponctuation: bool = True) -> list[tuple[int, int]]:
    """Coupe une proposition à chaque virgule ou deux-points séparant deux
    propositions assez substantielles.

    La coupure est systématique et non conditionnée à une longueur : « Quand
    une commande dépasse mille euros, le service applique une remise » tient en
    70 caractères et porte pourtant une condition ET sa conclusion. Tant qu'un
    segment porte les deux, aucune étiquette ne peut être juste et le rappel
    reste plafonné. Mesuré sur les références du benchmark : ce plafond passe
    de 59 % à 80 %.
    """
    if ponctuation:
        candidates = [
            match.end(1)
            for match in _SOFT_BREAK.finditer(text, start, end)
            if match.end(1) - start >= MIN_CLAUSE
            and end - match.end() >= MIN_CLAUSE
        ]
        grammaticales = syntaxe.coupures_de_clauses(text, start, end)
        cuts = [
            cut
            for cut in candidates
            if grammaticales is None or cut in grammaticales
        ]
        if cuts:
            bornes = [start] + cuts + [end]
            return [
                morceau
                for index in range(len(bornes) - 1)
                for morceau in _split_soft(
                    text, bornes[index], bornes[index + 1], ponctuation=False
                )
            ]

    # Certaines conditions ne sont séparées par aucune ponctuation :
    # « X n'est retenu que si Y » ou « X because Y ». Le connecteur est de la
    # colle discursive, pas une partie nécessaire du span ; on garde donc les
    # deux clauses exactes situées de part et d'autre.
    for match in _INLINE_CONDITION_BREAK.finditer(text, start, end):
        # « y compris lorsque… » étend la portée de la principale ; il ne la
        # conditionne pas. Retirer « lorsque » transformerait précisément
        # cette extension en prémisse autonome.
        if _EXTENSION_DE_PORTEE.search(text[start:match.start()]):
            continue
        if (
            match.start() - start >= MIN_CLAUSE
            and end - match.end() >= MIN_CLAUSE
        ):
            return (
                _split_soft(text, start, match.start(), ponctuation)
                + _split_soft(text, match.end(), end, ponctuation)
            )

    # Conditions nominales suffixées : elles n'ont pas de verbe, mais jouent
    # le même rôle que « si… » dans « prend fin à l'échéance…, en cas de… ».
    for match in _NOMINAL_SUFFIX_BREAK.finditer(text, start, end):
        if match.start() - start >= MIN_CLAUSE and end - match.start() >= MIN_CLAUSE:
            return (
                _split_soft(text, start, match.start(), ponctuation)
                + _split_soft(text, match.start(), end, ponctuation)
            )
    # Testé et ÉCARTÉ : couper aussi quand le subordonnant est EN TÊTE de
    # proposition (« Lorsque X, Y »), pour l'exclure de l'empan comme il l'est
    # partout ailleurs. Le banc et le jeu hors échantillon atteignent déjà
    # 317/317 et 21/21 des empans de référence au critère du scoreur — le
    # recouvrement de 50 % rend les sept lettres du connecteur indifférentes.
    # Le garder, en revanche, INFORME : `peut_conclure` rejette un segment
    # dont le premier jeton est SCONJ, ce qui interdit à une condition de
    # devenir conclusion. La coupure aurait détruit ce signal sans rien gagner.

    # Deux conditions nominales coordonnées gardent chacune leur span :
    # « Chez un patient fébrile ET SANS signe de gravité, ... ».
    if text[start:end].lstrip().lower().startswith("chez "):
        match = _AND_WITHOUT_BREAK.search(text, start, end)
        if (match is not None and match.start() - start >= MIN_CLAUSE
                and end - match.end() >= MIN_CLAUSE):
            return [(start, match.start()), (match.end(), end)]

    # Même traitement pour « si X alors Y » sans virgule, mais seulement
    # lorsque le début de la proposition est explicitement conditionnel.
    fragment = text[start:end]
    if _CONDITIONAL_START.search(fragment):
        match = _THEN_BREAK.search(text, start, end)
        if (
            match is not None
            and match.start() - start >= MIN_CLAUSE
            and end - match.end() >= MIN_CLAUSE
        ):
            return [(start, match.start()), (match.end(), end)]

    return [(start, end)]


# L'espace INSÉCABLE (U+00A0) est massif dans les textes réglementaires —
# EUR-Lex en met entre le numéro de paragraphe et son texte. Il n'était pas
# rogné, si bien qu'un empan pouvait commencer par un blanc invisible.
_BLANCS = " \t\r\n\u00a0\u202f"

# « 4. », « 2) », « a) », « iv. » en tête de segment : numérotation, pas
# proposition. Le nombre est borné à trois chiffres pour ne pas confondre une
# énumération avec une année (« 2026. L'année suivante » reste intact), et les
# lettres sont minuscules pour épargner « M. Dupont ».
_ENUMERATION_EN_TETE = re.compile(
    r"[ \t\u00a0\u202f]*(?:\d{1,3}|[a-z]|[ivxl]{1,5})[.)°][ \t\u00a0\u202f]+(?=\S)"
)

# Têtes qui n'OUVRENT PAS une proposition : elles rattachent un complément ou
# une apposition à ce qui précède. « …, y compris la fourniture d'un service, »
# et « …, entre autres, » étaient découpés en propositions autonomes, étiquetés,
# puis câblés comme conditions — les faux positifs qui dominaient encore le jeu
# hors échantillon.
#
# La liste est GRAMMATICALE, pas thématique : ce sont des mots-outils du
# français. Les têtes d'EXCEPTION (« sauf », « hormis », « sous réserve ») en
# sont volontairement absentes — elles introduisent une vraie condition
# négative, et les fusionner détruirait une exception. Mesuré : les inclure
# faisait avaler la clause « sauf lorsqu'il constitue le seul moyen
# disponible… » de `48-drones-autonomes`.
_INCISE_EN_TETE = re.compile(
    # Pas d'ancre `^` : le motif est appliqué avec `match(text, start, end)`,
    # qui ancre déjà à `start` — un `^` y exigerait l'offset zéro du document.
    r"(?:y\s+compris|entre\s+autres|notamment|à\s+savoir|le\s+cas\s+échéant|"
    r"en\s+particulier|(?:ou|et)\s+(?:de\s|d['’]|à\s)|d['’])",
    re.I,
)

# Ce qui peut séparer une incise de ce qu'elle complète : une ponctuation
# faible, rien de plus. Un point ou un point-virgule interdit la fusion.
_LIAISON_FAIBLE = frozenset(" ,:\t\r\n  ")


# Même motif ANCRÉ EN DÉBUT DE LIGNE, pour l'effacer AVANT toute analyse.
_ENUMERATION_DE_LIGNE = re.compile(
    r"(?m)^[ \t  ]*(?:\d{1,3}|[a-z]|[ivxl]{1,5})[.)°]"
    r"(?=[ \t  ]+\S)"
)


def neutraliser_enumerations(texte: str) -> str:
    """Efface les numéros de paragraphe en début de ligne, sans bouger un offset.

    Rogner l'empan ne suffisait pas : le numéro reste dans la chaîne analysée
    par spaCy, et il en déforme l'arbre de dépendances AVANT que la découpe
    n'ait lieu. Sur « 12)   Quand le sol gèle, … » le point de coupure se
    déplaçait, et « Quand » — le subordonnant qui porte toute la
    conditionnalité — passait du côté de la principale.

    La substitution remplace chaque caractère par une ESPACE : la longueur est
    conservée, donc tous les offsets du document restent exacts et les empans
    écrits dans le `.ann` désignent le même texte que le fichier source.
    """
    return _ENUMERATION_DE_LIGNE.sub(lambda m: " " * len(m.group()), texte)


def _segments_typographiques(text: str) -> list[tuple[int, int]]:
    """Découpe par la seule ponctuation, AVANT la scission des coordonnées.

    Isolé de `segment_text` parce que `_freres_coordonnes` a besoin de savoir
    quels segments finaux proviennent d'une même unité typographique : c'est
    ce qui distingue deux conditions coordonnées de deux propositions
    voisines.
    """
    blocks: list[tuple[int, int]] = []
    cursor = 0
    for match in _HARD_BREAK.finditer(text):
        if match.start() > cursor:
            blocks.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        blocks.append((cursor, len(text)))

    segments: list[tuple[int, int]] = []
    for block_start, block_end in blocks:
        # Les blancs de tête ne sont pas de la proposition, et ils ne sont pas
        # inoffensifs : `_split_soft` mesure la longueur de ce qui PRÉCÈDE un
        # marqueur conditionnel pour décider d'y couper. Six espaces résiduelles
        # — ce que laisse la neutralisation d'un « 12) » — suffisaient à faire
        # passer ce test, produisant un segment fait de vide et rejetant
        # « Quand » du côté de la principale.
        while block_start < block_end and text[block_start] in _BLANCS:
            block_start += 1
        start = block_start
        for match in _SENTENCE_BREAK.finditer(text, block_start, block_end):
            if not _is_real_break(text, match.start(1), match.end()):
                continue
            if match.end(2) - start >= MIN_SEGMENT:
                segments.extend(_split_soft(text, start, match.end(2)))
            start = match.end()
        if block_end - start >= MIN_SEGMENT:
            segments.extend(_split_soft(text, start, block_end))
    return segments


def segment_text(text: str) -> list[tuple[int, int]]:
    """Découpe le texte en propositions et renvoie leurs bornes exactes."""
    text = neutraliser_enumerations(text)
    segments = _segments_typographiques(text)

    # Scinder les subordonnées COORDONNÉES : « Comme A et que B, alors C »
    # énonce deux conditions, que la découpe typographique laissait dans un
    # seul segment. Mesuré : 9 des 15 règles de référence à plusieurs
    # conditions étaient inatteignables pour cette raison, 1 seulement après.
    # La couverture des 268 spans de référence reste intégrale.
    coordonnees: list[tuple[int, int]] = []
    for start, end in segments:
        bornes = [start] + syntaxe.points_de_coupure(text, start, end) + [end]
        coordonnees.extend((bornes[i], bornes[i + 1])
                           for i in range(len(bornes) - 1))

    trimmed: list[tuple[int, int]] = []
    for start, end in coordonnees:
        while start < end and text[start] in _BLANCS:
            start += 1
        # Un MARQUEUR D'ÉNUMÉRATION en tête n'appartient pas à la proposition.
        # « 4.   Au moment de déterminer… » : le numéro de paragraphe restait
        # collé au premier segment, et son premier jeton devenait « 4 » au lieu
        # de « Lorsque ». Tous les tests grammaticaux s'en trouvaient aveuglés
        # — `dep_ == "mark"`, `peut_conclure`, le court-circuit — et les empans
        # étaient décalés par rapport à la référence.
        #
        # Le banc ne pouvait pas le montrer : sa prose n'est pas numérotée. Le
        # jeu HORS ÉCHANTILLON l'a révélé au premier essai, sur du texte
        # réglementaire réel — la forme que prend presque tout texte normatif.
        # Mesuré : 10 segments concernés hors échantillon, ZÉRO sur les 509
        # segments du banc, et ni « 12 heures » ni « 2026. L'année » ne sont
        # entamés.
        marqueur = _ENUMERATION_EN_TETE.match(text, start, end)
        if marqueur:
            start = marqueur.end()
        while end > start and text[end - 1] in _BLANCS + ".,;:":
            end -= 1
        if end - start >= MIN_SEGMENT:
            trimmed.append((start, end))

    # RECOLLER LES INCISES. Une apposition n'est pas une proposition : elle ne
    # peut ni conclure ni conditionner, et l'étiqueter fabrique une entité
    # fausse puis une règle fausse. La virgule qui la sépare est réintégrée
    # dans l'empan, comme le fait la référence.
    #
    # Mesuré avant d'être écrit : ZÉRO fusion sur les 48 documents du banc et
    # les 32 récits négatifs — le mécanisme y est inerte, il ne peut donc pas
    # y avoir dégradé quoi que ce soit. Hors échantillon : 6 fusions, 37
    # segments ramenés à 31, empans de référence exactement retrouvés 5 → 8,
    # couverture inchangée à 21/21.
    fusionnes: list[tuple[int, int]] = []
    for start, end in trimmed:
        if (
            fusionnes
            and _INCISE_EN_TETE.match(text, start, end)
            and set(text[fusionnes[-1][1]:start]) <= _LIAISON_FAIBLE
        ):
            fusionnes[-1] = (fusionnes[-1][0], end)
            continue
        fusionnes.append((start, end))

    # RENDRE SON SUBORDONNANT À LA PROPOSITION. « Si, et dans la mesure où, il
    # n'est pas possible de… » : la virgule INTERNE au connecteur déclenche la
    # découpe, et « Si, et dans la mesure où » devient un segment. Il fabrique
    # une entité fausse, une règle fausse, et prive la proposition suivante du
    # mot qui portait sa conditionnalité — `peut_conditionner` ne voit plus
    # rien. La fusion se fait donc vers l'AVANT, contrairement aux incises.
    #
    # Mesuré avant d'être écrit : 0 segment touché sur les 48 cas et les 32
    # récits, 0 sur le premier lot hors échantillon, 4 sur le second — tous des
    # fragments ne recouvrant AUCUN empan de référence.
    tronques = syntaxe.connecteurs_tronques(text, fusionnes)
    if not tronques:
        return fusionnes
    recolles: list[tuple[int, int]] = []
    reporte: int | None = None
    for rang, (start, end) in enumerate(fusionnes):
        if reporte is not None:
            start = reporte
            reporte = None
        # Le connecteur ne rejoint sa proposition que si rien de plus qu'une
        # ponctuation faible ne les sépare — un point les rend étrangers.
        if (rang in tronques and rang + 1 < len(fusionnes)
                and set(text[end:fusionnes[rang + 1][0]]) <= _LIAISON_FAIBLE):
            reporte = start
            continue
        recolles.append((start, end))
    return recolles


# ============================================================================
# Instructions des étapes
# ============================================================================

_LABEL_INSTRUCTIONS = """STEP 1 - WHICH SEGMENTS TAKE PART IN THE REASONING

The text has been cut into numbered segments. Answer TWO independent yes/no
questions for EACH segment:

  "p" - PROPOSITION: does the segment state something that can be true or
        false and that takes part in the reasoning - a fact, a condition, a
        circumstance, a decision, a judgement, a conclusion?

  "m" - MARKER: does the segment contain an explicit cue of priority between
        two conclusions ("prefers", "takes precedence over", "prevails over",
        "wins over", "overrides", or its equivalent in the language used)?

A segment may be BOTH a proposition and contain a marker. Do not discard its
propositional content merely because it also says "takes precedence over".
Answer "N" to both questions for a heading, numbering, identification,
administrative detail, pure narration, an incidental example, or quoted
material nobody takes up.

Do NOT try to decide here whether a proposition is a conclusion or a ground.
That depends on how it relates to the others, and it is worked out at the next
step. A single proposition can be the conclusion of one inference and the
ground of another; asking the question here would have no correct answer.

A segment stating what someone does, decides, must do, may do, or what holds
in a situation IS a proposition, however plain its wording. "N" is for
segments that state nothing assessable: a heading, a number, a name, a
reference.

When in doubt about "p", answer "Y": a proposition wrongly kept costs nothing
if no relation attaches to it, whereas one wrongly dropped is lost for good.

Answer with JSON only:
{"labels":[{"n":1,"p":"Y","m":"N"},{"n":2,"p":"N","m":"N"},
           {"n":3,"p":"Y","m":"Y"}]}
One entry per segment, using the numbers shown, in order."""


_ENUMERATION_INSTRUCTIONS = """STEP 0b - ENUMERATED ALTERNATIVES

Some segments list several alternatives at once: "he may buy lamb, pork,
chicken or fish" offers four, not one. Each of them is a distinct possible
decision, and they must be separated before anything else.

Report ONLY the segments that enumerate alternatives OFFERED AS A CHOICE, and
for each, the items of the list, copied character for character.

EVERY ITEM YOU REPORT MUST APPEAR INSIDE THE SEGMENT YOU NUMBER. A list may
already have been cut across several segments: then report, for each segment,
only the items that segment itself contains, and skip a segment holding just
one item. Never gather into one entry items that live in different segments.

This is NOT an enumeration to report:
  - a list of facts or circumstances that hold together ("the road is icy and
    the tyres are worn") - those are joint conditions, not alternatives;
  - a coordination inside a single decision ("he takes his coat and leaves");
  - two clauses joined by "and" that each state something different.

Most segments enumerate nothing. Report an empty list rather than force one.

Answer with JSON only:
{"enumerations":[{"n":3,"items":["du porc","du poulet","du poisson"]}]}
Use only the numbers shown, and copy the items from the segment itself."""


_SUPPORT_INSTRUCTIONS = """STEP 2 - WHAT THE TEXT PUTS FORWARD TO REACH EACH PROPOSITION

You are given numbered propositions from one text, then the numbers of those
to examine. For EACH one, say which OTHER propositions the text puts forward
as grounds for reaching it - the "if" side of which it is the "then".

Ask, for an ordered pair: does the text present A as a reason to accept B?
If so, A is a ground of B. Direction matters: A grounding B is not B
grounding A. Consequence cues ("therefore", "so", "hence", "it follows"),
condition cues ("if", "when", "provided that") and justification cues
("because", "since", "as") point to the direction, each in its own way.

The surest cue is a subordinate clause of condition or cause attached to a
main clause: in "when X, Y" / "if X, Y" / "Y because X" / "since X, Y", the
subordinate X grounds the main clause Y. Take that link every time you see it,
in whatever language the text uses.

A proposition merely sitting nearby, or belonging to another inference, is not
a ground. Proximity alone is never a reason. Conversely, do not answer empty
out of caution: a conclusion the text actually justifies must receive its
grounds.

Grounds usually come BEFORE the conclusion they support, though a text may
also state the conclusion first and justify it after. A ground placed after
its conclusion is the less common case: propose one only when the text plainly
calls for it, never merely because the two propositions sit side by side.

A proposition may be the ground of one inference while itself resting on
others: that is a normal chain, report both.
Never make two propositions ground each other.

Answer with JSON only, one entry per proposition asked:
{"support":[{"c":4,"g":[1,2]},{"c":9,"g":[]}]}
Use only the numbers shown."""



# --- Décomposition par paires -------------------------------------------
# Le modèle ne décide plus « combien de relations existent » : le code énumère
# les paires candidates et il rend une étiquette par paire. Deux juges séparés
# — l'un cherche le fondement, l'autre teste l'explication narrative — pour
# que l'avertissement narratif n'influence pas la recherche elle-même.

_PAIRE_INSTRUCTIONS = """Determine the relation between two propositions in their context.

Each item gives A, B, and the exact PASSAGE from A to B. Use the passage to
read intervening cues, but classify only the directed relation between A and B.

A relation A -> B holds when A is put forward as a reason, a condition, a
justification or a ground from which B may be concluded.

Chronology, closeness in the text, the flow of a narrative, and a connective
such as "however" are not enough to establish a relation.

Classes:
  A_TO_B     A is a ground for B
  B_TO_A     B is a ground for A
  NONE       neither grounds the other
  AMBIGUOUS  the text does not settle it

Decide from the given text alone. Never supply a premise the text does not
state.

  A: "The disk is full."      B: "The backup failed."        -> A_TO_B
  A: "Paul enters the shop."  B: "However Marie stays out."  -> NONE

Answer with JSON only, one entry per numbered pair:
{"relations":[{"n":1,"r":"A_TO_B"},{"n":2,"r":"NONE"}]}"""


_NARRATIF_INSTRUCTIONS = """For each numbered pair, one question only: can the apparent connection
between the two propositions be explained by the telling alone? Use the
PASSAGE to see the surrounding narrative, but judge only A against B.

Answer YES when what links them is only their order in time, their closeness
in the text, the flow of a description or a narrative, or a connective marking
a turn in the account.

Answer NO when one proposition is put forward as a reason to accept the other.

  A: "She arrived at nine."   B: "She then opened the window."  -> YES
  A: "The disk is full."      B: "The backup failed."           -> NO

Answer with JSON only, one entry per numbered pair:
{"narrative":[{"n":1,"r":"YES"},{"n":2,"r":"NO"}]}"""


_RANKING_INSTRUCTIONS = """STEP 3 - THE PREFERENCES BETWEEN CONCLUSIONS

You are given numbered conclusions from one text, numbered pieces of context,
and the segments that carry a PRIORITY CUE. Report the places where the text
states that, in a given situation, one conclusion WINS over another.

Every preference you report must rest on one of the segments listed under
PRIORITY CUES: those are the only places where the text ranks anything. If
that list says "(none)", there is no preference to report.

Under "marker_n", give the number of the PRIORITY CUE segment used. Under
"marker", copy the ranking expression ALONE out of that same segment - a few
words, never the whole segment. In "the discount takes precedence over the
advance payment", the marker is "takes precedence over".

NOT a preference: "however", "but", "though", "nevertheless", a change of
subject. And two cases stated side by side are NOT a preference: "if A then X,
if B then Y" describes two different situations, not a conflict. A preference
needs two conclusions in COMPETITION and the text saying which one wins.

Most texts contain no preference. That is the normal case: answer
{"preferences":[]}. Never invent one.

You are NOT deciding any level or ranking of preferences against each other.
You report one row per ranking the text states, and nothing else.

For each row give:
  "preferred"   the conclusions the text retains in that situation
  "contrasted"  the conclusions it sets aside there
  "complement"  true ONLY when the text designates the contrasted set by an
                anaphora such as "the others", "the other two", "les autres"
                or "les deux autres". In that case leave "contrasted" empty:
                Python will compute the complement inside the active parent
                option set. Never use every option in the document.
  "delta"       the context pieces this situation ADDS - only the new ones
  "reactivates" the context pieces already stated earlier that the text
                explicitly brings back ("even in winter", "in that case",
                "still"). Leave empty when the text brings nothing back.
  "marker"      the ranking expression alone

TWO OPPOSITE ROWS ARE NORMAL AND EXPECTED. "In winter A wins over B" and "if
the goods are local B wins over A even in winter" are two rows, not a
contradiction: they hold in different situations. Report both. Never drop one
because it seems to conflict with the other - it is precisely that pair that
the text is building.

Answer with JSON only:
{"preferences":[{"preferred":[2],"contrasted":[5],"delta":[7],
                 "complement":false,"reactivates":[],"marker_n":9,
                 "marker":"prime sur"}]}
Use only the numbers shown."""


# ============================================================================
# Orchestration
# ============================================================================

DEFAULT_MODEL = os.environ.get("GORGIAS_MODEL", "qwen3:8b")
DEFAULT_BASE_URL = os.environ.get("GORGIAS_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_TIMEOUT = float(os.environ.get("GORGIAS_TIMEOUT", "300"))
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = int(os.environ.get("GORGIAS_NUM_PREDICT", "512"))
OUTPUT_FORMATS = {"brat", "lpp"}
LABEL_BATCH = 30          # segments étiquetés par appel
LONG_DOCUMENT_THRESHOLD = 60
LONG_DOCUMENT_LABEL_BATCH = 8
SUPPORT_BATCH = 4         # conclusions rattachées par appel
PAIR_WINDOW = 6           # écart maximal entre deux propositions d'une paire
PAIR_BATCH = 10           # paires jugées par appel
SUPPORT_WINDOW = 12       # segments amont proposés comme prémisses candidates
SUPPORT_LOOKAHEAD = 3
RANKING_BATCH = 10
_MAX_MARKER_WORDS = 5

# L'étape d'énumération est une détection de listes alternatives. Sans aucune
# marque de choix ou de liste dans le lot, son résultat valide est forcément
# vide : appeler le modèle dans ce cas consommait auparavant un tour complet
# sur presque chaque document.
_ENUMERATION_SURFACE_CUE = re.compile(
    r"\b(?:ou|soit|either|or|choix|options?|alternatives?)\b",
    re.IGNORECASE | re.DOTALL,
)


def _numbered(source: str, spans, numbers) -> str:
    return "\n".join(
        f"{number}. {source[start:end]}" for number, (start, end) in zip(numbers, spans)
    )


def _yes(value) -> bool:
    return str(value).strip().upper()[:1] in {"Y", "O", "1", "T"}


def _read_labels(items) -> dict[int, frozenset[str]]:
    if not isinstance(items, list):
        raise ValueError("\"labels\" doit être une liste")
    labels: dict[int, frozenset[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        if "p" in item or "m" in item:
            tags = {
                tag
                for tag, field in (("P", "p"), ("M", "m"))
                if _yes(item.get(field))
            }
            labels[number] = frozenset(tags or {"N"})
            continue

        # Compatibilité avec les anciennes réponses durant une relance ou
        # avec un modèle qui reproduit encore l'ancien exemple.
        tag = str(item.get("t", "")).strip().upper()[:1]
        if tag in {"P", "M", "N"}:
            labels[number] = frozenset({tag})
    if not labels:
        raise ValueError("aucune étiquette exploitable")
    return labels


def _read_support(items) -> dict[int, list[int]]:
    if not isinstance(items, list):
        raise ValueError("\"support\" doit être une liste")
    par_conclusion: dict[int, list[int]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            conclusion = int(item.get("c"))
        except (TypeError, ValueError):
            continue
        grounds = []
        for value in item.get("g") or []:
            try:
                grounds.append(int(value))
            except (TypeError, ValueError):
                continue
        par_conclusion[conclusion] = grounds
    if not par_conclusion:
        raise ValueError("aucun rattachement exploitable")
    return par_conclusion


_COMPLEMENT_REFERENCE = re.compile(
    r"(?:^\s*COMPLEMENT\s*$|\b(?:autres?|others?|remaining|restants?|reste)\b)",
    re.IGNORECASE,
)


def _read_rankings(items) -> list[dict]:
    """Lit les lignes de la table de préférences scénarisées (phase B)."""
    if not isinstance(items, list):
        raise ValueError("\"preferences\" doit être une liste")

    def liste(valeurs):
        if valeurs is None:
            return []
        return valeurs if isinstance(valeurs, list) else [valeurs]

    def entiers(valeurs):
        sortie = []
        for valeur in liste(valeurs):
            try:
                sortie.append(int(valeur))
            except (TypeError, ValueError):
                continue
        return sortie

    lignes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        def entiers_ou_mentions(valeurs):
            sortie = []
            for valeur in liste(valeurs):
                if isinstance(valeur, str) and valeur.strip():
                    try:
                        sortie.append(int(valeur))
                    except ValueError:
                        sortie.append(valeur.strip())
                else:
                    try:
                        sortie.append(int(valeur))
                    except (TypeError, ValueError):
                        continue
            return sortie

        preferees = entiers_ou_mentions(item.get("preferred"))
        brutes_ecartees = liste(item.get("contrasted"))
        complement = _yes(item.get("complement")) or any(
            isinstance(valeur, str) and _COMPLEMENT_REFERENCE.search(valeur)
            for valeur in brutes_ecartees
        )
        ecartees = entiers_ou_mentions(
            [
                valeur
                for valeur in brutes_ecartees
                if not (
                    isinstance(valeur, str)
                    and _COMPLEMENT_REFERENCE.search(valeur)
                )
            ]
        )
        if not preferees or (not ecartees and not complement):
            continue
        marqueur = item.get("marker")
        try:
            numero_marqueur = int(item.get("marker_n"))
        except (TypeError, ValueError):
            numero_marqueur = None
        lignes.append(
            {
                "preferred": preferees,
                "contrasted": ecartees,
                "complement": complement,
                "delta": entiers(item.get("delta")),
                "reactivates": entiers(item.get("reactivates")),
                "marker_n": numero_marqueur,
                "marker": marqueur.strip() if isinstance(marqueur, str) else "",
            }
        )
    return lignes




def _read_enumerations(items) -> dict[int, list[str]]:
    if not isinstance(items, list):
        raise ValueError("\"enumerations\" doit être une liste")
    trouvees: dict[int, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            numero = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        morceaux = [
            m.strip() for m in (item.get("items") or [])
            if isinstance(m, str) and m.strip()
        ]
        if len(morceaux) >= 2:
            trouvees[numero] = morceaux
    return trouvees


def _split_enumerations(
    texte, segments, enumerations, return_scopes: bool = False
):
    """Éclate les segments énumérant des alternatives en un segment par item.

    Une alternative co-listée (« du porc, du poulet ou du poisson ») porte
    plusieurs options dans un seul segment : tant qu'elles n'en forment qu'un,
    aucune ne peut être désignée séparément dans une préférence. L'éclatement
    a lieu avant l'étiquetage, si bien que tout l'aval travaille comme
    d'habitude sur des numéros de segments.
    """
    affines: list[tuple[int, int]] = []
    scopes: list[frozenset[int]] = []
    for index, (debut, fin) in enumerate(segments, start=1):
        morceaux = enumerations.get(index)
        if not morceaux:
            affines.append((debut, fin))
            continue
        bornes = []
        curseur = debut
        for morceau in morceaux:
            trouve = _span_candidates(texte[curseur:fin], morceau)
            if not trouve:
                continue
            bornes.append((curseur + trouve[0][0], curseur + trouve[0][1]))
            curseur += trouve[0][1]
        # On n'éclate que si tous les items ont été retrouvés dans l'ordre :
        # un éclatement partiel perdrait du texte sans prévenir.
        if len(bornes) == len(morceaux):
            premier = len(affines) + 1
            affines.extend(bornes)
            scopes.append(
                frozenset(range(premier, premier + len(bornes)))
            )
        else:
            affines.append((debut, fin))
    return (affines, scopes) if return_scopes else affines


def _bloc_du_segment(texte, segments, numero: int) -> int:
    """Numéro du bloc textuel, les sauts de ligne délimitant les passages."""
    if not 1 <= numero <= len(segments):
        return -1
    return len(_HARD_BREAK.findall(texte[: segments[numero - 1][0]]))


def _resolve_complement_scope(
    preferred,
    marker_number,
    option_scopes,
    previous_preferences,
    option_numbers,
    texte,
    segments,
):
    """Résout O_parent pour calculer O_parent \\ O_preferred.

    Priorité à l'ensemble préféré du scénario parent le plus proche, puis à
    la dernière énumération d'alternatives du même bloc. Aucune portée globale
    au document n'est autorisée : l'absence ou l'égalité de deux candidats
    produit ``AMBIGUOUS``.
    """
    preferred = set(preferred)
    options = set(option_numbers)
    marker_block = _bloc_du_segment(texte, segments, marker_number)
    candidats = []

    for index, preference in enumerate(previous_preferences):
        parent = set(preference["preferred"]) & options
        parent_marker = preference.get("marker_n")
        if (
            preferred < parent
            and isinstance(parent_marker, int)
            and parent_marker < marker_number
            and _bloc_du_segment(texte, segments, parent_marker) == marker_block
        ):
            candidats.append(
                (marker_number - parent_marker, 0, -index, frozenset(parent))
            )

    for index, scope in enumerate(option_scopes):
        active = set(scope) & options
        if not active or not preferred < active:
            continue
        last = max(scope)
        if (
            last < marker_number
            and _bloc_du_segment(texte, segments, last) == marker_block
        ):
            candidats.append(
                (marker_number - last, 1, -index, frozenset(active))
            )

    if not candidats:
        return None, "AMBIGUOUS"
    candidats.sort(key=lambda item: item[:3])
    meilleur = candidats[0]
    egalite = [
        candidat for candidat in candidats
        if candidat[:2] == meilleur[:2] and candidat[3] != meilleur[3]
    ]
    if egalite:
        return None, "AMBIGUOUS"
    return sorted(meilleur[3] - preferred), "RESOLVED"


_MOTS_VIDES = {
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "l", "d",
    "the", "a", "an", "of", "to", "and", "or", "et", "ou", "ce", "cette",
    "nous", "il", "elle", "on", "we", "it", "he", "she",
}


def _mots_pleins(fragment: str) -> set[str]:
    return {
        _fold(mot.group())
        for mot in _WORD.finditer(fragment)
        if _fold(mot.group()) not in _MOTS_VIDES and len(mot.group()) > 1
    }


def _lier_mention(texte, mention, candidats, segments) -> int | None:
    """Relie une mention (« le porc ») à l'option déjà posée qui la reprend.

    Les options classées par une préférence sont souvent reprises à
    l'intérieur de la clause de classement, sous une forme différente de leur
    première apparition. On apparie sur les mots pleins partagés.
    """
    cherches = _mots_pleins(mention)
    if not cherches:
        return None
    meilleur, score, etendue = None, 0.0, 0
    for numero in candidats:
        debut, fin = segments[numero - 1]
        communs = cherches & _mots_pleins(texte[debut:fin])
        if not communs:
            continue
        rapport = len(communs) / len(cherches)
        # À rapport égal, la mention la plus courte l'emporte : un long
        # segment contenant le mot n'est pas une reprise de l'option, il
        # ne fait que la mentionner au passage.
        if rapport > score or (rapport == score and fin - debut < etendue):
            meilleur, score, etendue = numero, rapport, fin - debut
    return meilleur if score >= 0.5 else None



def _read_relations(items) -> dict[int, str]:
    if not isinstance(items, list):
        raise ValueError("\"relations\" doit être une liste")
    lues: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            numero = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        etiquette = str(item.get("r", "")).strip().upper()
        if etiquette in {"A_TO_B", "B_TO_A", "NONE", "AMBIGUOUS"}:
            lues[numero] = etiquette
    if not lues:
        raise ValueError("aucune relation exploitable")
    return lues


def _read_narratif(items) -> dict[int, str]:
    if not isinstance(items, list):
        raise ValueError("\"narrative\" doit être une liste")
    lues: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            numero = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        etiquette = str(item.get("r", "")).strip().upper()[:1]
        if etiquette in {"Y", "N"}:
            lues[numero] = etiquette
    if not lues:
        raise ValueError("aucun jugement narratif exploitable")
    return lues


def _index_paragraphes(texte: str, segments) -> list[int]:
    """Numéro de paragraphe de chaque segment, indexé comme les segments (1-based).

    Un paragraphe est un bloc séparé par une ligne vide. Le découpage est
    typographique et déterministe : il ne coûte aucune inférence.
    """
    frontieres = []
    position = 0
    for bloc in texte.split("\n\n"):
        if bloc.strip():
            debut = texte.index(bloc, position)
            frontieres.append((debut, debut + len(bloc)))
            position = debut + len(bloc)
    index = [0] * (len(segments) + 1)
    for numero, (debut, _) in enumerate(segments, start=1):
        for rang, (bloc_debut, bloc_fin) in enumerate(frontieres):
            if bloc_debut <= debut < bloc_fin:
                index[numero] = rang
                break
    return index


def _freres_coordonnes(texte: str) -> set[tuple[int, int]]:
    """Paires de segments issues d'une MÊME subordonnée coordonnée.

    « Comme A ET QUE B, alors C » énonce deux conditions d'une seule règle.
    `segment_text` les sépare déjà — c'est la découpe des coordonnées — mais
    jette aussitôt l'information qu'elles sont sœurs, si bien que la suite du
    pipeline les traite comme deux propositions sans lien.

    Deux conséquences mesurées, toutes deux corrigées par ce qui suit :
      - la seconde condition est PERDUE (`25-vol-retarde`, `26-chaine-froid`) ;
      - pire, la syntaxe prouve parfois « A => B » entre les deux sœurs, ce qui
        fabrique une règle là où il n'y a qu'une conjonction de conditions.

    Les frontières sont celles que `points_de_coupure` a posées, jamais une
    liste de coordonnants : la détection reste grammaticale.
    """
    freres: set[tuple[int, int]] = set()
    if syntaxe.desactive():
        return freres
    numero = 0
    for debut, fin in _segments_typographiques(texte):
        coupures = syntaxe.points_de_coupure(texte, debut, fin)
        bornes = [debut] + coupures + [fin]
        precedent = None
        for rang in range(len(bornes) - 1):
            a, b = bornes[rang], bornes[rang + 1]
            while a < b and texte[a] in " \t\r\n":
                a += 1
            while b > a and texte[b - 1] in " \t\r\n.,;:":
                b -= 1
            if b - a < MIN_SEGMENT:
                precedent = None
                continue
            numero += 1
            if precedent is not None and rang > 0:
                freres.add((precedent, numero))
            precedent = numero
    return freres


_LIAISON_ALTERNATIVE = re.compile(
    r"\bou\s+(?:si\b|s['’](?:il|elle|on|ils|elles)\b)\s*$", re.IGNORECASE
)
_CONDITION_NOMINALE_SURFACE = re.compile(
    r"^\s*(?:à\s+l['’]échéance\b|en\s+cas\s+de\b)", re.IGNORECASE
)


def _alternatives_coordonnees(texte: str, segments) -> set[tuple[int, int]]:
    """Conditions voisines reliées par OU, à développer en règles séparées.

    Le schéma brat/LPP représente une conjonction par plusieurs ``Condition``
    sur une règle, mais n'a pas d'opérateur de disjonction. « si A ou si B »
    doit donc devenir deux règles de même effet. Les conditions nominales
    suffixées d'une même conclusion (« à l'échéance…, en cas de… ») suivent la
    même convention d'annotation.
    """
    alternatives: set[tuple[int, int]] = set()
    for numero in range(1, len(segments)):
        gauche = segments[numero - 1]
        droite = segments[numero]
        liaison = texte[max(gauche[0], gauche[1] - 12):droite[0]]
        fragment_g = texte[gauche[0]:gauche[1]]
        fragment_d = texte[droite[0]:droite[1]]
        if (_LIAISON_ALTERNATIVE.search(liaison)
                or (_CONDITION_NOMINALE_SURFACE.search(fragment_g)
                    and _CONDITION_NOMINALE_SURFACE.search(fragment_d))):
            alternatives.add((numero, numero + 1))
    return alternatives


def _variantes_conditions(premisses, alternatives) -> list[list[int]]:
    """Développe les composantes alternatives, en conservant les conjonctions."""
    presentes = list(dict.fromkeys(premisses))
    graphe = {p: set() for p in presentes}
    for gauche, droite in alternatives:
        if gauche in graphe and droite in graphe:
            graphe[gauche].add(droite)
            graphe[droite].add(gauche)
    composantes: list[list[int]] = []
    visites: set[int] = set()
    for premisse in presentes:
        if premisse in visites or not graphe[premisse]:
            continue
        pile = [premisse]
        composante = []
        while pile:
            courant = pile.pop()
            if courant in visites:
                continue
            visites.add(courant)
            composante.append(courant)
            pile.extend(graphe[courant] - visites)
        composantes.append(sorted(composante))
    communs = [p for p in presentes if p not in visites]
    variantes = [communs]
    for composante in composantes:
        variantes = [base + [choix] for base in variantes for choix in composante]
    return variantes or [communs]


_LOCUTEUR = re.compile(r"^([A-ZÉÈÀÂÎÔÛ][A-ZÉÈÀÂÎÔÛ\-]+)\s*\.", re.MULTILINE)


def _index_locuteurs(texte: str, segments) -> list[int]:
    """Numéro de LOCUTEUR de chaque segment, indexé comme les segments.

    Un dialogue est repéré par ses étiquettes de tour (« ANNA. », « BORIS. ») :
    une capitale suivie d'un point en tête de ligne. Hors dialogue, tous les
    segments reçoivent le même numéro et le garde-fou est inerte.

    Les tours d'un MÊME locuteur portent le même numéro : mesuré sur le banc,
    une règle de référence peut réunir deux prémisses énoncées dans deux tours
    distincts d'ANNA (`05-dialogue`), mais aucune ne franchit un changement de
    locuteur.
    """
    noms: dict[str, int] = {}
    frontieres: list[tuple[int, int]] = []
    for marque in _LOCUTEUR.finditer(texte):
        nom = marque.group(1)
        frontieres.append((marque.start(), noms.setdefault(nom, len(noms))))
    index = [0] * (len(segments) + 1)
    for numero, (debut, _) in enumerate(segments, start=1):
        courant = 0
        for position, rang in frontieres:
            if position <= debut:
                courant = rang
            else:
                break
        index[numero] = courant
    return index


def _meme_bloc(paires, index) -> list[tuple[int, int]]:
    """Retient les paires dont les deux membres sont dans le même bloc.

    Le bloc est donné par `index` : paragraphe (`_index_paragraphes`) ou
    locuteur (`_index_locuteurs`). Les deux frontières relèvent du même
    principe — un auteur qui va à la ligne clôt son raisonnement, un locuteur
    qui cède la parole aussi — et se composent en appliquant le filtre deux
    fois.

    LOCUTEUR, mesuré sur les 5 dialogues du banc : 11 relations de référence,
    dont ZÉRO franchit un changement de locuteur ; 8 relations produites, dont
    4 le franchissent et les QUATRE sont fausses. Le garde-fou retire donc
    4 règles fausses et aucune juste — c'est un gain de précision pur, exact
    et non statistique. La conclusion d'un interlocuteur n'est pas la prémisse
    d'un autre : c'est une position concurrente, ce que la priorité capture.
    """
    return [p for p in paires if index[p[0]] == index[p[1]]]


def _meme_paragraphe(paires, paragraphe) -> list[tuple[int, int]]:
    """Retient les paires dont les deux membres sont dans le même paragraphe.

    Un auteur qui va à la ligne clôt son raisonnement : une prémisse et sa
    conclusion ne se répartissent pas de part et d'autre d'un blanc. La fenêtre
    de portée, calibrée sur des documents d'un seul paragraphe de six à sept
    segments, relie sinon des paragraphes entiers dès que le texte s'allonge.

    Mesuré sur un texte de 47 segments et 11 paragraphes : 170 des 261 paires à
    portée (65 %) traversaient une frontière, et les fausses règles observées
    en venaient. INERTE SUR LE BANC PAR CONSTRUCTION — aucun des 79 documents
    de mesure n'a plus d'un paragraphe (852 paires positives et 298 négatives,
    toutes intra-paragraphe) : le garde-fou ne peut donc rien y dégrader, mais
    le banc ne peut rien y démontrer non plus. Il a été instrumenté sur des
    documents longs, hors banc.
    """
    return [p for p in paires if paragraphe[p[0]] == paragraphe[p[1]]]


def _paires_candidates(propositions, fenetre) -> list[tuple[int, int]]:
    """Énumère les paires non ordonnées à portée. La direction est demandée
    au modèle, pas déduite de l'ordre : le code ne préjuge de rien."""
    ordonnees = sorted(propositions)
    return [
        (gauche, droite)
        for index, gauche in enumerate(ordonnees)
        for droite in ordonnees[index + 1 :]
        if droite - gauche <= fenetre
    ]


def _presenter(texte, segments, paires, numeros) -> str:
    return "\n".join(
        f"{numero}. A: {texte[segments[g - 1][0]:segments[g - 1][1]]}\n"
        f"   B: {texte[segments[d - 1][0]:segments[d - 1][1]]}\n"
        f"   PASSAGE: {texte[segments[g - 1][0]:segments[d - 1][1]]}"
        for numero, (g, d) in zip(numeros, paires)
    )


def _compile_preferences(assembler, lignes, rule_of_option, key_of) -> int:
    """Compile les lignes de préférence en prefer/meta_prefer (phases C et D).

    Le modèle ne fournit que l'incrément de contexte de chaque ligne. Le
    scénario cumulatif est reconstruit ici : une ligne qui réactive
    explicitement un contexte déjà posé ("même en hiver") hérite du scénario
    de la ligne où ce contexte figurait. La méta-priorité n'est jamais
    demandée, elle se déduit de l'inclusion stricte des scénarios cumulés.
    """
    scenarios: list[frozenset] = []
    for index, ligne in enumerate(lignes):
        parent = None
        for autre in range(index):
            if ligne["reactivates"] & scenarios[autre]:
                parent = autre
        scenarios.append(
            (scenarios[parent] if parent is not None else frozenset())
            | ligne["delta"]
            | ligne["reactivates"]
        )

    def contextes_de(numeros):
        """TOUS les Contexts du scénario, dans l'ordre du texte.

        Une Option n'y a pas sa place : le validateur refuse l'annotation
        entière si on l'y met. On les rend tous parce que le scénario d'une
        préférence est souvent une conjonction — n'en retenir qu'un la
        déclenche dès que ce seul contexte tient.
        """
        return [cle for numero in sorted(numeros)
                if (cle := key_of.get(numero)) is not None and cle[2] == "Context"]

    evenement: dict[int, list[int]] = {}
    # Deux lignes de préférence peuvent classer la même paire de règles dans le
    # même contexte : le produit cartésien en produirait alors des doublons.
    deja: dict[tuple, int] = {}
    produites = 0
    for index, ligne in enumerate(lignes):
        quand = contextes_de(ligne["delta"]) or contextes_de(scenarios[index])
        # Un groupe préféré face à un groupe écarté se développe en produit
        # cartésien : c'est au code de le faire, pas au modèle. Chaque paire
        # ainsi classée est aussi une paire ``complement(o', o)`` au sens de
        # l'algorithme 1 : les deux options sont en conflit, indépendamment de
        # l'orientation de la préférence.
        for gagnante in ligne["preferred"]:
            for perdante in ligne["contrasted"]:
                assembler.complement(key_of[gagnante], key_of[perdante])
                regle_g = rule_of_option.get(key_of[gagnante])
                regle_p = rule_of_option.get(key_of[perdante])
                if regle_g is None or regle_p is None or regle_g == regle_p:
                    continue
                # `quand` est une liste de contextes : la figer pour la clé.
                signature = (regle_g, regle_p, tuple(quand))
                if signature in deja:
                    evenement.setdefault(index, []).append(deja[signature])
                    continue
                deja[signature] = assembler.priority(
                    "prefer", ligne["marker"], regle_g, regle_p, quand
                )
                evenement.setdefault(index, []).append(deja[signature])
                produites += 1

    for index, ligne in enumerate(lignes):
        for autre in range(len(lignes)):
            if autre == index or autre not in evenement or index not in evenement:
                continue
            inverse = (
                set(ligne["preferred"]) == set(lignes[autre]["contrasted"])
                and set(ligne["contrasted"]) == set(lignes[autre]["preferred"])
            )
            signature = (evenement[index][0], evenement[autre][0])
            if inverse and scenarios[autre] < scenarios[index] and signature not in deja:
                deja[signature] = 1
                assembler.priority(
                    "meta_prefer",
                    ligne["marker"],
                    evenement[index][0],
                    evenement[autre][0],
                    contextes_de(scenarios[index] - scenarios[autre]),
                )
                produites += 1
    return produites



# ============================================================================
# Étape 2 — trois implémentations mesurables
#
# `groupe`   : une question par lot de conclusions (« lesquelles de ces
#              propositions fondent celle-ci ? »).
# `paires`   : décomposition par paires avec deux juges séparés, telle que
#              recommandée pour isoler la décision narrative de la recherche
#              de fondement.
# `hybride`  : le groupe propose avec un rappel élevé ; les deux juges ne
#              vérifient ensuite que ces candidats. Défaut.
#
# L'étage entièrement par paires était trop conservateur et asséchait le
# graphe. L'hybride conserve la capacité de proposition du groupe tout en
# rendant chaque lien réfutable séparément.
# ============================================================================


def _relations_par_groupe(
    model, texte, segments, propositions, say, examiner=None
) -> dict:
    """Interroge par lot de conclusions : rappel plus élevé, décision moins nette."""
    grounds: dict[int, list[int]] = {}
    # Les prémisses candidates sont bornées au paragraphe des conclusions
    # examinées. La fenêtre amont vaut 12 segments, le double de PAIR_WINDOW :
    # sur un document d'un seul paragraphe elle est sans effet, mais dès que le
    # texte s'allonge elle inonde le prompt de segments d'autres paragraphes.
    # Mesuré sur un texte de 47 segments : 200 candidats présentés, 96 après
    # contrainte (-52 %), dans l'étage qui pèse 46 % du temps total.
    paragraphe = _index_paragraphes(texte, segments)
    locuteur = _index_locuteurs(texte, segments)
    cibles = propositions if examiner is None else examiner
    for start in range(0, len(cibles), SUPPORT_BATCH):
        group = cibles[start : start + SUPPORT_BATCH]
        candidates = sorted(
            {
                autre
                for number in group
                for autre in range(
                    max(1, number - SUPPORT_WINDOW),
                    min(len(segments), number + SUPPORT_LOOKAHEAD) + 1,
                )
                if autre in propositions
                and paragraphe[autre] == paragraphe[number]
                and locuteur[autre] == locuteur[number]
            }
        )
        if not candidates:
            continue
        payload = (
            "PROPOSITIONS:\n"
            + _numbered(texte, [segments[i - 1] for i in candidates], candidates)
            + "\n\nEXAMINE: "
            + ", ".join(str(number) for number in group)
        )
        try:
            retenus = _ask_stage(
                model,
                _SUPPORT_INSTRUCTIONS,
                payload,
                "support",
                _read_support,
                expected=group,
                partiel=True,
            )
        except StageError:
            continue
        for number in group:
            premisses = [
                autre
                for autre in dict.fromkeys(retenus.get(number, []))
                if autre in propositions and autre != number
            ]
            if premisses:
                grounds[number] = premisses
    say(f"étape 2 (groupe) : {sum(len(v) for v in grounds.values())} relation(s)")
    return grounds


def _verifier_paires(
    model,
    texte,
    segments,
    paires,
    say,
) -> dict:
    """Oriente des paires candidates, puis élimine les liens narratifs.

    Un routeur statistique Text-JEPA/SIGReg servait ici d'éclaireur ; il a été
    supprimé. Ses caractéristiques étaient démontrées inertes : `lexical+jepa`
    égalait `lexical-only` au quatrième chiffre, les embeddings n'étant qu'une
    application linéaire du vecteur lexical qui alimentait déjà la même tête
    linéaire. Voir reports/analyse_erreurs_hybride.md.
    """
    orientees: dict[tuple[int, int], str] = {}
    a_soumettre = []

    # Deux conditions coordonnées ne s'infèrent pas l'une l'autre. La paire est
    # retirée AVANT les juges : mesuré, la syntaxe prouve parfois « Comme le
    # retard dépasse trois heures » => « et que la cause relève du
    # transporteur », fabriquant une règle là où il n'y a qu'une conjonction.
    freres = _freres_coordonnes(texte)
    alternatives = _alternatives_coordonnees(texte, segments)
    paires = [p for p in paires if p not in freres and p not in alternatives]

    # Court-circuit symbolique. Une subordonnée circonstancielle introduite par
    # un marqueur explicite ("quand", "si", "parce que") porte sa direction dans
    # sa grammaire : la faire juger par un modèle de langue n'ajoute qu'un
    # risque. Mesuré sur les 116 relations de référence, ce motif en couvre 51 %
    # avec ZÉRO erreur de direction — d'où la dispense des deux juges, y compris
    # du filtre narratif : une conditionnelle explicite n'est pas un lien de
    # récit. Sans spaCy installé, prouvees est vide et le comportement est
    # inchangé.
    prouvees = syntaxe.relations_prouvees(texte, segments, paires)
    if prouvees:
        orientees.update(prouvees)
        say(f"         {len(prouvees)} relation(s) prouvée(s) par la syntaxe, "
            "dispensée(s) des juges")
    # Filtre de singularité. Une paire dépourvue de connecteur explicite dont
    # les DEUX verbes sont au passé, ou qui porte un ancrage temporel précis
    # (« le lendemain », « vers vingt-deux heures »), relate un enchaînement
    # d'événements révolus — un récit, pas une règle. Mesuré : les deux seuls
    # documents narratifs pollués du banc le sont par ce motif, de façon
    # reproductible (six exécutions, six règles fausses), et le filtre narratif
    # du modèle les laisse passer. Coût mesuré : une règle juste sur 67.
    # Les paires PROUVÉES par la syntaxe n'y sont pas soumises : leur
    # connecteur atteste déjà l'intention argumentative.
    ecartees_recit = 0
    for paire in paires:
        if paire in prouvees:
            continue
        gauche, droite = paire
        if singularite.evenement_singulier(
                texte[segments[gauche - 1][0]:segments[gauche - 1][1]],
                texte[segments[droite - 1][0]:segments[droite - 1][1]]):
            ecartees_recit += 1
            continue
        a_soumettre.append(paire)
    if ecartees_recit:
        say(f"         {ecartees_recit} paire(s) écartée(s) : événements "
            "singuliers, pas de règle")

    for start in range(0, len(a_soumettre), PAIR_BATCH):
        lot = a_soumettre[start : start + PAIR_BATCH]
        numeros = list(range(1, len(lot) + 1))
        try:
            verdicts = _ask_stage(
                model, _PAIRE_INSTRUCTIONS,
                _presenter(texte, segments, lot, numeros),
                "relations", _read_relations,
                expected=numeros,
                partiel=True,
            )
        except StageError:
            continue
        for numero, paire in zip(numeros, lot):
            if verdicts.get(numero) in {"A_TO_B", "B_TO_A"}:
                orientees[paire] = verdicts[numero]

    # Les relations prouvées grammaticalement sont exclues du filtre narratif :
    # une subordonnée conditionnelle explicite n'est pas un effet de récit.
    #
    # La dispense NE VA PAS PLUS LOIN, et c'est mesuré. Étendre l'exemption aux
    # paires déjà orientées A_TO_B dont la prémisse s'ouvre sur un introducteur
    # récupérait bien les 4 documents visés (+4 règles justes) mais produisait
    # 8 règles de plus : précision marginale 50 %, contre 73 % pour le système
    # et une cible proche de 100 %. F1 sémantique 74,8 % -> 74,2 %, et un récit
    # pollué sur 32. Le mécanisme a donc été mesuré puis refusé.
    retenues = sorted(p for p in orientees if p not in prouvees)
    rejetees = 0
    non_verifiees = 0
    for start in range(0, len(retenues), PAIR_BATCH):
        lot = retenues[start : start + PAIR_BATCH]
        numeros = list(range(1, len(lot) + 1))
        try:
            verdicts = _ask_stage(
                model, _NARRATIF_INSTRUCTIONS,
                _presenter(texte, segments, lot, numeros),
                "narrative", _read_narratif,
                expected=numeros,
                partiel=True,
            )
        except StageError:
            for paire in lot:
                if paire in orientees:
                    del orientees[paire]
                    non_verifiees += 1
            continue
        for numero, paire in zip(numeros, lot):
            verdict = verdicts.get(numero)
            if verdict == "Y":
                del orientees[paire]
                rejetees += 1
            elif verdict is None and paire in orientees:
                # Un bulletin partiel ne dispense pas de la réfutation : une
                # paire que le juge narratif n'a pas examinée reste NON
                # vérifiée, et une paire non vérifiée ne passe pas.
                del orientees[paire]
                non_verifiees += 1
    say(
        f"         {len(orientees)} retenue(s), "
        f"{rejetees} narrative(s), {non_verifiees} non vérifiée(s)"
    )

    grounds: dict[int, list[int]] = {}
    certaines: dict[int, set[int]] = {}
    refusees = 0
    for paire, etiquette in orientees.items():
        gauche, droite = paire
        premisse, conclusion = (
            (gauche, droite) if etiquette == "A_TO_B" else (droite, gauche)
        )
        # CE QUI NE PEUT PAS ÊTRE UNE CONCLUSION. Une subordonnée marquée
        # (« Puisque l'exposition est critique ») énonce une prémisse, et un
        # adjoint concessif (« même en hiver ») n'affirme rien : ni l'une ni
        # l'autre ne peut être ce qu'on conclut. Le test ne porte QUE sur la
        # conclusion — les deux formes sont des conditions légitimes.
        #
        # Mesuré : 7 règles fausses retirées, ZÉRO juste détruite, précision
        # des règles 80 % -> 87 %. Le retournement de ces relations a été
        # essayé et mesuré inutile (0 juste sur 3), d'où le rejet.
        if not syntaxe.peut_conclure(texte, *segments[conclusion - 1]):
            refusees += 1
            continue
        # Et symétriquement : une subordonnée d'exception ou de concession
        # (« sauf lorsque… », « même lorsque… ») ne CONDITIONNE pas la règle,
        # elle la défait. En faire une prémisse inverse le sens.
        if not syntaxe.peut_conditionner(texte, *segments[premisse - 1]):
            refusees += 1
            continue
        grounds.setdefault(conclusion, []).append(premisse)
        if paire in prouvees:
            certaines.setdefault(conclusion, set()).add(premisse)
    if refusees:
        say(f"         {refusees} relation(s) refusée(s) : la conclusion est "
            "une subordonnée ou un adjoint concessif")

    # PURETÉ DES CONCLUSIONS PROUVÉES. Quand la grammaire a déjà donné la
    # condition d'une conclusion, une prémisse SUPPLÉMENTAIRE non prouvée pour
    # la même conclusion est écartée : une subordonnée marquée énonce sa
    # condition, elle ne la complète pas.
    #
    # Le défaut visé n'est pas un manque de rappel mais une POLLUTION : sur
    # `37-arbitrage-sportif` et `43-forets`, la condition juste et prouvée était
    # bien présente, le modèle en ajoutait une seconde, et la règle entière
    # devenait fausse — emportant avec elle la priorité qui s'y appuyait. Une
    # prémisse parasite détruit deux éléments de référence.
    #
    # Mesuré hors ligne sur le banc de 48 documents : règles justes 73 -> 75,
    # PRIORITÉS justes 15 -> 17, règles produites INCHANGÉES (102). La précision
    # et le rappel montent ensemble — le mécanisme ne relâche rien, il élague.
    # Les priorités atteignent ainsi leur plafond structurel : seules 17 des 46
    # ont leurs DEUX règles justes, et une priorité ne peut exister sans elles.
    #
    # Les règles à plusieurs conditions légitimes ne sont pas touchées : la
    # découpe des subordonnées coordonnées donne à chacune son propre `mark`,
    # donc toutes sont prouvées et aucune n'est élaguée.
    for conclusion, premisses in grounds.items():
        prouvees_ici = certaines.get(conclusion)
        if not prouvees_ici:
            continue
        gardees = [p for p in premisses if p in prouvees_ici]
        if gardees and len(gardees) < len(premisses):
            grounds[conclusion] = gardees

    # CONJONCTION DE CONDITIONS. « Comme A et que B, alors C » énonce DEUX
    # conditions d'une seule règle. La découpe des coordonnées sépare A et B
    # mais perd leur fraternité : la seconde condition n'est alors rattachée à
    # rien et la règle sort amputée.
    #
    # La propagation ne part que d'une condition PROUVÉE, jamais d'une simple
    # suggestion du modèle. Mesuré sur le banc : 5 cas où la référence réclame
    # bien la sœur, ZÉRO où elle la refuse — la garantie est exacte, pas
    # statistique.
    for conclusion, premisses in grounds.items():
        prouvees_ici = certaines.get(conclusion, set())
        for aine, cadette in freres - alternatives:
            if aine in prouvees_ici and cadette not in premisses:
                premisses.append(cadette)
            elif cadette in prouvees_ici and aine not in premisses:
                premisses.append(aine)
    return grounds


def _relations_par_paires(model, texte, segments, propositions, say) -> dict:
    """Une décision par paire, puis réfutation narrative sur les seules retenues."""
    paires = _meme_bloc(
        _meme_bloc(_paires_candidates(propositions, PAIR_WINDOW),
                   _index_paragraphes(texte, segments)),
        _index_locuteurs(texte, segments),
    )
    say(f"étape 2 (paires) : {len(paires)} paire(s) candidate(s)")
    return _verifier_paires(model, texte, segments, paires, say)


def _relations_hybrides(model, texte, segments, propositions, say) -> dict:
    """Union des relations prouvées et des propositions du modèle, puis vérification.

    Les relations grammaticalement prouvées sont calculées sur TOUTES les paires
    à portée, indépendamment de ce que le modèle propose. C'est une correction :
    elles étaient auparavant calculées à l'intérieur de la vérification, donc
    uniquement sur les paires que le modèle avait déjà suggérées. Le mécanisme
    déterministe ne pouvait alors que rattraper une paire rejetée par les juges,
    jamais introduire un lien que le modèle avait manqué — alors que compenser
    ses oublis était sa raison d'être.

    Mesuré : huit documents totalement muets portaient des relations prouvées
    inexploitées, jusqu'à quatre sur `33-deneigement` et `35-semis`. Le motif
    n'a produit aucune erreur de direction sur les 116 relations de référence,
    et son introduction avait laissé les négatifs inchangés.
    """
    paragraphe = _index_paragraphes(texte, segments)
    locuteur = _index_locuteurs(texte, segments)
    a_portee = _meme_bloc(
        _meme_bloc(_paires_candidates(propositions, PAIR_WINDOW), paragraphe),
        locuteur,
    )
    # Les relations prouvées sont filtrées elles aussi. Une subordonnée et sa
    # principale appartiennent à la même phrase, donc au même paragraphe : une
    # arête `advcl` qui franchit un blanc est un artefact d'analyse, jamais une
    # relation. Mesuré sur le texte « drones » : les deux seules arêtes prouvées
    # traversantes étaient les deux fausses.
    prouvees = {
        paire: verdict
        for paire, verdict in syntaxe.relations_prouvees(
            texte, segments, a_portee).items()
        if paragraphe[paire[0]] == paragraphe[paire[1]]
    }
    # Une conclusion déjà déterminée par un connecteur explicite n'a pas à
    # repasser par l'étape de proposition. `_verifier_paires` écarte de toute
    # façon les prémisses non prouvées ajoutées à une telle conclusion (règle
    # de pureté ci-dessus) : ces appels étaient donc coûteux et sans effet sur
    # la sortie. Les propositions restent toutes disponibles comme prémisses
    # candidates pour les autres conclusions.
    conclusions_prouvees = set()
    premisses_prouvees = set()
    for (gauche, droite), verdict in prouvees.items():
        premisse, conclusion = (
            (gauche, droite) if verdict == "A_TO_B" else (droite, gauche)
        )
        if (syntaxe.peut_conclure(texte, *segments[conclusion - 1])
                and syntaxe.peut_conditionner(texte, *segments[premisse - 1])):
            conclusions_prouvees.add(conclusion)
            premisses_prouvees.add(premisse)
    a_examiner = [
        n for n in propositions
        if n not in conclusions_prouvees
        and n not in premisses_prouvees
        and syntaxe.peut_conclure(texte, *segments[n - 1])
    ]
    proposees = _relations_par_groupe(
        model, texte, segments, propositions, lambda message: None,
        examiner=a_examiner,
    )
    du_modele = {
        tuple(sorted((premisse, conclusion)))
        for conclusion, premisses in proposees.items()
        for premisse in premisses
        if premisse != conclusion
    }
    paires = _meme_bloc(_meme_bloc(sorted(du_modele), paragraphe), locuteur)
    paires = sorted(set(paires) | set(prouvees))
    inedites = len(set(prouvees) - du_modele)
    say(
        "étape 2 (hybride) : "
        f"{sum(map(len, proposees.values()))} proposition(s) de lien, "
        f"{len(prouvees)} prouvée(s) par la syntaxe dont {inedites} inédite(s), "
        f"{len(conclusions_prouvees)} conclusion(s) dispensée(s) de proposition, "
        f"{len(paires)} paire(s) à vérifier"
    )
    return _verifier_paires(model, texte, segments, paires, say)


ETAGES_RELATION = {
    "groupe": _relations_par_groupe,
    "hybride": _relations_hybrides,
    "paires": _relations_par_paires,
}
# Le module JEPA/SIGReg a été supprimé : l'analyse d'erreurs a montré ses
# caractéristiques inertes, pour un coût d'appels supérieur.
DEFAULT_ETAGE = "hybride"


def _coeurs_physiques() -> int:
    """Cœurs physiques, et non fils logiques.

    Mesuré sur un Ryzen 5 PRO 4650U (6 cœurs, 12 fils), avec un modèle
    de 4 milliards de paramètres sur CPU :
    6 fils rendent 8,14 tokens/s, 12 fils 6,05 — l'hyperthreading fait perdre
    25 % du débit parce que les deux fils d'un cœur se disputent la même unité
    vectorielle. La génération étant le poste dominant du pipeline, le réglage
    par défaut d'`os.cpu_count()` coûtait un tiers du temps total.
    """
    force = os.environ.get("GORGIAS_NUM_THREAD")
    if force:
        try:
            return max(1, int(force))
        except ValueError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            coeurs = set()
            physique = noyau = None
            for ligne in source:
                if ligne.startswith("physical id"):
                    physique = ligne.split(":")[1].strip()
                elif ligne.startswith("core id"):
                    noyau = ligne.split(":")[1].strip()
                    coeurs.add((physique, noyau))
            if coeurs:
                return len(coeurs)
    except OSError:
        pass
    return os.cpu_count() or 1


def _create_model(
    model_name: str,
    num_ctx: int,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
):
    """Charge le client Ollama uniquement lorsqu'une inférence est demandée."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        client_kwargs={"timeout": timeout},
        temperature=0,
        reasoning=False,
        num_ctx=num_ctx,
        format="json",
        seed=SEED,
        top_k=1,
        num_thread=_coeurs_physiques(),
        # Toutes les étapes rendent de petits objets JSON fermés. Sans plafond,
        # une réponse dégénérée peut monopoliser le CPU jusqu'au timeout de
        # cinq minutes alors que le plus gros lot valide tient sous 512 tokens.
        num_predict=DEFAULT_NUM_PREDICT,
    )


def annotate(
    texte: str,
    model_name: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    report=None,
    etage: str = DEFAULT_ETAGE,
    votes: int | None = None,
    model=None,
    output_format: str = "brat",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Extrait le graphe et le rend en brat ou en composants LPP/Gorgias."""
    global VOTES
    if votes is not None:
        VOTES = max(1, votes)
    if etage not in ETAGES_RELATION:
        raise ValueError(f"étage de relation inconnu : {etage!r}")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"format de sortie inconnu : {output_format!r}")
    model = model or _create_model(model_name, num_ctx, base_url, timeout)
    say = report if report is not None else (lambda message: None)

    # Une seule fois, en amont : toute la suite — découpe, syntaxe, empans —
    # travaille sur un texte débarrassé de sa numérotation. La longueur étant
    # conservée, les offsets restent ceux du fichier source.
    texte = neutraliser_enumerations(texte)

    segments = segment_text(texte)
    if not segments:
        raise ValueError("le texte ne contient aucune proposition exploitable")
    say(f"étape 0 : {len(segments)} segments découpés (sans modèle)")

    enumerations: dict[int, list[str]] = {}
    for start in range(0, len(segments), LABEL_BATCH):
        chunk = segments[start : start + LABEL_BATCH]
        numbers = list(range(start + 1, start + len(chunk) + 1))
        surface = texte[chunk[0][0]:chunk[-1][1]]
        if not _ENUMERATION_SURFACE_CUE.search(surface):
            continue
        try:
            enumerations.update(
                _ask_stage(
                    model,
                    _ENUMERATION_INSTRUCTIONS,
                    _numbered(texte, chunk, numbers),
                    "enumerations",
                    _read_enumerations,
                )
            )
        except StageError:
            continue
    option_scopes: list[frozenset[int]] = []
    if enumerations:
        avant = len(segments)
        segments, option_scopes = _split_enumerations(
            texte, segments, enumerations, return_scopes=True
        )
        say(
            f"étape 0b : {len(enumerations)} énumération(s) signalée(s), "
            f"{len(segments) - avant} segment(s) gagné(s) par éclatement"
        )

    # --- Passe 1 : isoler les énoncés de classement --------------------------
    # Mesuré : soumis au même flux que le reste, un classement est étiqueté
    # comme une Option, engendre une règle fantôme, et la préférence désigne
    # ensuite cette règle comme gagnante. Deux corrections par le prompt ont
    # échoué. On le retire donc du flux plutôt que de demander au modèle de
    # l'ignorer. Seule la clause de classement part : le contexte qui la
    # précède reste, c'est le When de la priorité.
    classements, _ = priorites.isoler(texte, segments)
    if classements:
        say(f"passe 1 : {len(classements)} énoncé(s) de classement isolé(s) "
            "et soustrait(s) du flux")

    # Même traitement pour les REPRISES de contexte : « même en hiver »,
    # « même à budget contraint ». Elles n'énoncent aucun fait neuf, ne
    # conditionnent rien et n'affirment rien — les références ne leur donnent
    # aucune entité. Laissées dans le flux, elles se font étiqueter comme des
    # propositions ordinaires et deviennent des Context fantômes.
    #
    # Mesuré sur les 188 documents annotés : 20 segments correspondent, tous
    # dans un document de raffinement, AUCUN ne recouvre une entité de
    # référence, et il n'y en a aucun dans les 32 récits ni les quatre lots
    # RGPD. Leur rôle argumentatif est ailleurs : c'est cette reprise qui fait
    # d'une priorité le raffinement d'une autre, ce dont l'étape 3 se sert.
    reprises = [
        numero for numero, (debut, fin) in enumerate(segments, start=1)
        if numero not in classements
        and (priorites.est_reprise_de_contexte(texte[debut:fin])
             # Un connecteur de conséquence isolé — « Par conséquent »,
             # « Dès lors » — n'énonce rien non plus : il porte la direction
             # de l'inférence, que l'étape 2 lit dans sa position, et les
             # références ne lui donnent aucune entité.
             or syntaxe.est_connecteur_de_consequence(texte[debut:fin]))
    ]
    hors_flux = set(classements) | set(reprises)
    if reprises:
        say(f"passe 1 : {len(reprises)} reprise(s) de contexte soustraite(s) "
            "du flux")

    # Déterminer les extrémités grammaticalement certaines AVANT l'étiquetage.
    # Elles sont nécessairement propositionnelles et n'ont donc rien à gagner
    # à passer par un classifieur génératif.
    tous = list(range(1, len(segments) + 1))
    paires_toutes = _meme_bloc(
        _meme_bloc(
            _paires_candidates(tous, PAIR_WINDOW),
            _index_paragraphes(texte, segments),
        ),
        _index_locuteurs(texte, segments),
    )
    recuperees = set()
    for (gauche, droite), verdict in syntaxe.relations_prouvees(
            texte, segments, paires_toutes).items():
        premisse, conclusion = (
            (gauche, droite) if verdict == "A_TO_B" else (droite, gauche)
        )
        if conclusion in hors_flux:
            if syntaxe.peut_conditionner(texte, *segments[premisse - 1]):
                recuperees.add(premisse)
            continue
        if (syntaxe.peut_conclure(texte, *segments[conclusion - 1])
                and syntaxe.peut_conditionner(texte, *segments[premisse - 1])):
            recuperees.update((premisse, conclusion))
    recuperees -= hors_flux

    # --- Étape 1 : étiquetage, par lots (passe 2, sur le flux nettoyé) -------
    labels: dict[int, frozenset[str]] = {}
    # Une réponse pour 30 étiquettes peut dépasser `num_predict=512` sur un
    # document long : le JSON est alors tronqué et les trois reprises coûtent
    # cher avant d'abandonner tout le lot. Les documents du banc ont au plus
    # 50 segments et gardent donc le réglage mesuré. Au-delà de 60, des lots
    # plus courts bornent la taille de sortie sans modifier le raisonnement.
    label_batch = (
        LONG_DOCUMENT_LABEL_BATCH
        if len(segments) > LONG_DOCUMENT_THRESHOLD
        else LABEL_BATCH
    )
    for start in range(0, len(segments), label_batch):
        chunk_complet = segments[start : start + label_batch]
        numeros_complets = list(range(start + 1, start + len(chunk_complet) + 1))
        garde = [(n, sp) for n, sp in zip(numeros_complets, chunk_complet)
                 if n not in hors_flux and n not in recuperees]
        if not garde:
            continue
        numbers = [n for n, _ in garde]
        chunk = [sp for _, sp in garde]
        try:
            labels.update(
                _ask_stage(
                    model,
                    _LABEL_INSTRUCTIONS,
                    _numbered(texte, chunk, numbers),
                    "labels",
                    _read_labels,
                    expected=numbers,
                    partiel=True,
                )
            )
        except StageError as error:
            say(f"  lot {start // label_batch + 1} : étiquetage abandonné ({error})")

    propositions = sorted(recuperees | {
        i for i in range(1, len(segments) + 1)
        if "P" in labels.get(i, frozenset())
    })
    # Les classements isolés sont des marqueurs par construction : ils n'ont
    # pas été soumis au modèle, donc ils ne peuvent pas porter d'étiquette.
    markers = sorted(set(classements) | {
        i for i in range(1, len(segments) + 1)
        if "M" in labels.get(i, frozenset())
    })
    if recuperees:
        say(
            "étape 1b : "
            f"{len(recuperees)} proposition(s) classée(s) sans modèle "
            "par connecteur explicite"
        )
    say(f"étape 1 : {len(propositions)} proposition(s), {len(markers)} marqueur(s)")
    if not propositions:
        say("étape 1 : aucune proposition argumentative — sortie vide")
        return ""

    # --- Étape 2 : rattachement des prémisses -------------------------------
    # La fenêtre est calculée sur la liste globale des segments : une prémisse
    # peut donc être loin en amont sans que rien ne soit perdu, ce qu'un
    # découpage en blocs indépendants ne permettait pas.
    relation_stage = ETAGES_RELATION[etage]
    grounds = relation_stage(model, texte, segments, propositions, say)

    # Deux propositions ne peuvent pas se fonder l'une l'autre : on tranche sur
    # l'ordre du texte, l'antérieure fondant la postérieure.
    for conclusion, premisses in list(grounds.items()):
        grounds[conclusion] = [
            premisse
            for premisse in premisses
            if conclusion not in grounds.get(premisse, []) or premisse < conclusion
        ]

    # Le rôle est dérivé du graphe, pas demandé au modèle : une proposition
    # conclue par une règle est une Option, une proposition qui ne fait que
    # fonder est un Context. Vérifié sans contre-exemple sur les références.
    conclusions = {number for number, prem in grounds.items() if prem}
    utilisees = {p for prem in grounds.values() for p in prem}

    # Deux options dérivées du même scénario de base appartiennent au même
    # univers actif O¹, même si une énumération a déjà été fragmentée par les
    # virgules. Cette reconstruction suit la définition SP = <S ; O> de
    # l'article et évite de prendre toutes les options du document.
    par_scenario: dict[frozenset[int], set[int]] = {}
    for conclusion in conclusions:
        scenario = frozenset(grounds.get(conclusion, ()))
        if scenario:
            par_scenario.setdefault(scenario, set()).add(conclusion)
    option_scopes.extend(
        frozenset(options)
        for options in par_scenario.values()
        if len(options) >= 2
    )

    assembler = Assembler(texte)
    key_of: dict[int, tuple[int, int, str]] = {}
    for number in propositions:
        if number in conclusions:
            key_of[number] = assembler.entity("Option", segments[number - 1])
        elif number in utilisees:
            key_of[number] = assembler.entity("Context", segments[number - 1])

    # Le contexte de scénario d'une priorité n'appartient à AUCUNE règle : dans
    # « En cas d'arrêt de ligne, le remplacement l'emporte sur la réparation »,
    # le When est une restriction autonome. Il n'était donc ni conclusion ni
    # prémisse utilisée, et n'obtenait jamais d'entité — d'où des priorités
    # compilées sans ancrage. On enregistre la proposition non-conclusion la
    # plus proche EN AMONT de chaque classement isolé.
    # Le segment IMMÉDIATEMENT amont est essayé D'ABORD, même si l'étiquetage
    # ne l'a pas retenu comme proposition. C'est le défaut que corrige cette
    # boucle : « En zone de pente », « Pour un renouvellement annuel » sont des
    # circonstants SANS VERBE, que l'étape 1 écarte parce qu'ils n'énoncent
    # rien de vrai ou faux par eux-mêmes — alors qu'ils portent tout l'ancrage
    # de la priorité. Faute de les voir, la compilation se rabattait sur le
    # contexte d'une règle voisine et produisait une priorité fausse : sur
    # `43-forets` et `31-tarification`, la priorité était construite, ancrée au
    # mauvais endroit, et comptait donc à la fois comme manquée et comme
    # fausse.
    #
    # Mesuré : le segment précédant un classement isolé est un Context de
    # référence 27 fois sur 32.
    freres_coordonnes = _freres_coordonnes(texte)
    for numero in classements:
        adjacent = numero - 1
        # « Quand A ET QUE B » est UNE conjonction de conditions, et son ancre
        # est sa TÊTE, pas sa queue. Sur `11-combinaison`, le segment adjacent
        # au classement est « et que la pluie tombe » : le prendre pour
        # contexte produisait une entité fausse ET une priorité fausse, là où
        # la référence désigne « le sol est gelé ». On remonte donc à l'aînée.
        if (adjacent - 1, adjacent) in freres_coordonnes:
            adjacent -= 1
        # Une reprise de contexte n'est pas le When : elle le rappelle. On la
        # franchit pour atteindre le circonstant qui porte réellement
        # l'ancrage.
        while adjacent >= 1 and adjacent in reprises:
            adjacent -= 1
        if (adjacent >= 1 and adjacent not in conclusions
                and adjacent not in key_of):
            debut, fin = segments[adjacent - 1]
            # Un contexte RÉPÉTÉ ne donne qu'une entité, comme dans les
            # références : « En hiver » énoncé deux fois est le même contexte,
            # et la priorité pointe sur sa PREMIÈRE énonciation.
            fragment = texte[debut:fin]
            reprise = next(
                (cle for cle in key_of.values()
                 if cle[2] == "Context" and texte[cle[0]:cle[1]] == fragment),
                None,
            )
            if reprise is None:
                # Chercher aussi parmi les SEGMENTS antérieurs identiques, même
                # sans entité : « En hiver » ouvre `09-raffinement-explicite`
                # sans verbe, donc l'étiquetage ne le retient pas et la reprise
                # ci-dessus ne pouvait rien trouver. On créait alors un doublon
                # à la seconde occurrence, là où la référence annote la
                # première.
                premier = next(
                    (i for i in range(1, adjacent)
                     if texte[segments[i - 1][0]:segments[i - 1][1]] == fragment),
                    None,
                )
                if premier is not None:
                    reprise = assembler.entity("Context", segments[premier - 1])
                    key_of[premier] = reprise
            key_of[adjacent] = reprise or assembler.entity(
                "Context", segments[adjacent - 1]
            )
            continue
        amont = [n for n in propositions
                 if n < numero and n not in conclusions and n not in key_of]
        if amont:
            n = amont[-1]
            key_of[n] = assembler.entity("Context", segments[n - 1])

    rule_of_option: dict[tuple[int, int, str], int] = {}
    alternatives = _alternatives_coordonnees(texte, segments)
    regles_produites = 0
    for number in sorted(conclusions):
        for variante in _variantes_conditions(grounds[number], alternatives):
            conditions = [key_of[p] for p in variante if p in key_of]
            if conditions:
                regle = assembler.rule(key_of[number], conditions)
                rule_of_option.setdefault(key_of[number], regle)
                regles_produites += 1
    say(
        f"étape 2 : {regles_produites} règle(s) — "
        f"{len(conclusions)} conclusion(s), "
        f"{len(utilisees - conclusions)} prémisse(s) pure(s)"
    )
    if not rule_of_option:
        # Une abstention est une sortie métier valide, pas une panne. Continuer
        # créerait des Context sans Option, puis le validateur signalerait à
        # tort un graphe cassé. Sans règle, aucune préférence compilable ne
        # peut exister non plus.
        say("étape 2 : aucune règle suffisamment sûre — sortie vide")
        return ""

    # --- Étape 3 : préférences (leur absence est normale) -------------------
    ordonnees = sorted(conclusions)
    contexts = sorted(n for n in key_of if n not in conclusions)
    lignes: list[dict] = []

    # Priorités déterministes, construites AVANT d'interroger le modèle.
    # Mesuré : lorsque le modèle ancrait sa propre ligne sur le même segment,
    # c'est elle qui l'emportait — avec gagnante et perdante inversées. Le
    # chemin symbolique, qui apparie 6 mentions sur 6 y compris les
    # nominalisations, doit passer en premier.
    couverts: set[int] = set()

    def fragment_de(numero: int) -> str:
        return texte[segments[numero - 1][0]:segments[numero - 1][1]]

    candidats = {n: fragment_de(n) for n in conclusions}
    # UNE RÈGLE SE DÉSIGNE PAR SA TÊTE OU PAR SON CORPS. Une préférence range
    # des règles, et le texte les nomme comme il veut : « la demande d'un
    # supérieur l'emporte sur son propre besoin » désigne la première règle par
    # sa conclusion (« l'employé prête l'objet ») et la seconde par sa
    # CONDITION (« s'il en a lui-même besoin »), jamais par sa conclusion
    # (« il le garde »). Apparier sur les seules conclusions laissait donc la
    # perdante introuvable et la priorité entière tombait.
    #
    # Le repli n'est essayé que là où l'appariement sur les conclusions ne rend
    # RIEN : il ne peut donc pas déplacer un appariement déjà trouvé. Mesuré
    # sur les 188 documents annotés, 140 clauses de classement : 125 se
    # résolvent sur la conclusion seule, 2 ne se résolvent qu'avec les
    # conditions, et ZÉRO se résout à tort.
    candidats_etendus = {
        n: " ".join([fragment_de(n)]
                    + [fragment_de(premisse) for premisse in grounds.get(n, ())])
        for n in conclusions
    }

    def apparier_regle(mention: str | None) -> int | None:
        if not mention:
            return None
        trouve = priorites.apparier(mention, candidats)
        if trouve is not None:
            return trouve
        return priorites.apparier(mention, candidats_etendus)

    for numero in classements:
        debut, fin = segments[numero - 1]
        fragment = texte[debut:fin]
        cotes = priorites.cotes_du_classement(fragment)
        if cotes is not None:
            gagnante = apparier_regle(cotes[0])
            perdante = apparier_regle(cotes[1])
        else:
            # COMPARATIF À PERDANTE ÉLIDÉE. « il préfère le poulet même en
            # hiver » nomme la gagnante et laisse la perdante implicite. Elle
            # ne se lit pas dans la clause : c'est la gagnante de la priorité
            # que cet énoncé raffine, donc une ligne DÉJÀ POSÉE où la gagnante
            # d'ici figure comme écartée. Sans ce chemin, le troisième niveau
            # de `01-courses-3-niveaux` est perdu tout entier — un marqueur,
            # une préférence et la seule méta-préférence du document.
            gagnante = apparier_regle(
                priorites.cote_gagnante_raffinee(fragment))
            perdante = next(
                (ligne["preferred"][0] for ligne in reversed(lignes)
                 if gagnante in ligne["contrasted"]),
                None,
            ) if gagnante is not None else None
        if gagnante is None or perdante is None or gagnante == perdante:
            continue
        empan = priorites.empan_du_marqueur(fragment)
        if empan is None:
            continue
        bornes_marqueur = (debut + empan[0], debut + empan[1])

        # LE SCÉNARIO D'UNE PRIORITÉ SE LIT DANS SA PROPRE PHRASE. Chercher le
        # dernier Context où qu'il soit en amont faisait ancrer la priorité sur
        # la condition d'une règle énoncée plus haut : sur `03-pret-objet`, la
        # priorité était compilée avec « s'il en a lui-même besoin » pour
        # scénario, c'est-à-dire la situation exactement inverse de celle que
        # le texte décrit.
        #
        # Mesuré sur les 140 priorités de référence de `data/cas` et
        # `data/synthetique` : 135 ancrages sont dans la phrase du marqueur.
        # Les 5 autres sont des REPRISES — « En hiver » énoncé deux fois, la
        # référence pointant sur la première énonciation — et la boucle
        # d'ancrage ci-dessus les ramène déjà à leur première occurrence. La
        # contrainte est donc exacte sur les 140, et non pas satisfaite 135
        # fois sur 140.
        ouverture, _ = priorites.bornes_de_phrase(texte, debut, fin)
        amont = [c for c in contexts
                 if c < numero and segments[c - 1][0] >= ouverture]
        if amont:
            delta = frozenset([amont[-1]])
        else:
            # Aucun circonstant dans la phrase : c'est alors le TERME GAUCHE du
            # classement qui décrit la situation. « Une demande venant d'un
            # supérieur hiérarchique l'emporte sur son propre besoin » ne pose
            # aucune circonstance à part — la circonstance EST le terme
            # comparé, et la référence lui donne son entité Context.
            #
            # Mesuré : sur les 134 clauses de classement de `data/cas` et
            # `data/synthetique`, 133 ont un circonstant dans leur phrase et
            # une seule n'en a pas — celle-là. Ce repli ne peut donc pas
            # déplacer un ancrage existant, il ne comble que le vide.
            gauche = texte[debut:bornes_marqueur[0]].rstrip(" ,;:\t")
            if len(gauche.split()) >= 2:
                key_of[numero] = assembler.entity(
                    "Context", (debut, debut + len(gauche))
                )
                delta = frozenset([numero])
            else:
                delta = frozenset()

        # RAFFINEMENT D'UNE PRIORITÉ DÉJÀ POSÉE. Deux classements qui rangent
        # la même paire en sens INVERSE ne se contredisent pas : ils valent
        # dans des situations différentes, et le texte dit laquelle est la plus
        # spécifique. C'est là, et nulle part ailleurs, que naît une
        # méta-préférence — `_compile_preferences` la déduit de l'inclusion
        # stricte des scénarios cumulés, encore faut-il que l'inclusion soit
        # établie. Le chemin déterministe ne posait jamais de `reactivates` :
        # aucune méta-préférence n'était donc compilable, quel que soit le
        # texte.
        #
        # L'inversion NE SUFFIT PAS à conclure au raffinement. La famille
        # « frères » du corpus synthétique le montre : « En hiver, le train
        # passe avant la voiture. Aux heures de pointe, la voiture prime sur le
        # train. » range la même paire en sens inverse dans deux situations
        # SŒURS, et n'attend aucune méta-préférence. Il faut donc, en plus,
        # que la phrase du classement annonce le raffinement — adversative en
        # ouverture, ou reprise concessive en clôture.
        #
        # Mesuré sur les 188 documents annotés : 25 documents attendent une
        # méta-préférence et 25 déclenchent l'annonce ; 202 n'en attendent
        # aucune et AUCUN ne la déclenche.
        reactivates = frozenset()
        if priorites.annonce_un_raffinement(
                priorites.phrase_autour(texte, debut, fin)):
            raffinee = next(
                (ligne for ligne in reversed(lignes)
                 if set(ligne["preferred"]) == {perdante}
                 and set(ligne["contrasted"]) == {gagnante}),
                None,
            )
            if raffinee is not None:
                reactivates = frozenset(raffinee["delta"])
        lignes.append({
            "preferred": [gagnante], "contrasted": [perdante],
            "delta": delta, "reactivates": reactivates,
            "marker_n": numero, "complement": False,
            # Ancrer sur l'amorce seule, pas sur la clause : les références
            # bornent le Marker au repère lui-même.
            "marker": assembler.entity("Marker", bornes_marqueur),
        })
        couverts.add(numero)
        say(f"passe 1 : priorité déterministe {gagnante} > {perdante}"
            + (f", contexte {sorted(delta)[0]}" if delta else ", sans contexte")
            + (f", raffine le scénario {sorted(reactivates)}"
               if reactivates else ""))
    for start in range(0, len(ordonnees), RANKING_BATCH):
        group = ordonnees[start : start + RANKING_BATCH]
        nearby = sorted(
            {c for number in group
             for c in contexts if abs(c - number) <= SUPPORT_WINDOW}
        )
        proches = sorted(
            {number for cible in group for number in markers
             if number not in couverts
             and abs(number - cible) <= SUPPORT_WINDOW}
        )
        if not proches:
            continue
        payload = (
            "CONCLUSIONS:\n"
            + _numbered(texte, [segments[i - 1] for i in group], group)
            + "\n\nCONTEXT:\n"
            + _numbered(texte, [segments[i - 1] for i in nearby], nearby)
            + "\n\nPRIORITY CUES FOUND IN THE TEXT:\n"
            + (
                _numbered(texte, [segments[i - 1] for i in proches], proches)
                if proches
                else "(none)"
            )
        )
        try:
            brutes = _ask_stage(
                model, _RANKING_INSTRUCTIONS, payload, "preferences", _read_rankings
            )
        except StageError:
            continue
        for ligne in brutes:
            mots = ligne["marker"].split()
            if not 0 < len(mots) <= _MAX_MARKER_WORDS:
                continue
            numero_marqueur = ligne.get("marker_n")
            if numero_marqueur not in proches:
                # Anciennes réponses sans marker_n : on n'infère l'ancre que
                # si un seul segment candidat contient effectivement le span.
                candidats_marqueur = []
                for numero in proches:
                    debut, fin = segments[numero - 1]
                    if locate_span(
                        texte[debut:fin], ligne["marker"], offset=debut
                    ) is not None:
                        candidats_marqueur.append(numero)
                if len(candidats_marqueur) != 1:
                    continue
                numero_marqueur = candidats_marqueur[0]
            debut, fin = segments[numero_marqueur - 1]
            bornes = locate_span(
                texte[debut:fin], ligne["marker"], offset=debut
            )
            if bornes is None:
                continue
            options = [n for n in key_of if key_of[n][2] == "Option"]

            def resoudre(valeurs):
                sortie = []
                for valeur in valeurs:
                    if isinstance(valeur, int) and valeur in key_of:
                        sortie.append(valeur)
                        continue
                    if isinstance(valeur, str):
                        lie = _lier_mention(texte, valeur, options, segments)
                        if lie is not None:
                            sortie.append(lie)
                return list(dict.fromkeys(sortie))

            preferees = resoudre(ligne["preferred"])
            ecartees = resoudre(ligne["contrasted"])
            if preferees and ligne.get("complement"):
                ecartees, statut = _resolve_complement_scope(
                    preferees,
                    numero_marqueur,
                    option_scopes,
                    lignes,
                    options,
                    texte,
                    segments,
                )
                if statut == "AMBIGUOUS":
                    assembler.diagnostic(
                        "AMBIGUOUS complement: univers parent local introuvable "
                        f"pour le segment {numero_marqueur}"
                    )
                    say(
                        "  complément AMBIGUOUS : aucun univers parent "
                        f"local et unique pour le marqueur {numero_marqueur}"
                    )
                    continue
            if not preferees or not ecartees or set(preferees) & set(ecartees):
                continue
            if numero_marqueur in couverts:
                continue  # déjà tranché symboliquement, et plus fiablement
            lignes.append(
                {
                    "preferred": preferees,
                    "contrasted": ecartees,
                    "delta": frozenset(
                        n for n in ligne["delta"] if n in key_of
                    ),
                    "reactivates": frozenset(
                        n for n in ligne["reactivates"] if n in key_of
                    ),
                    "marker_n": numero_marqueur,
                    "complement": bool(ligne.get("complement")),
                    "marker": assembler.entity("Marker", bornes),
                }
            )

    # Une priorité doit s'ancrer sur un classement REPÉRÉ SYMBOLIQUEMENT.
    # Mesuré sur les 47 documents : là où aucun classement n'est détecté, le
    # modèle produit 8 priorités et AUCUNE n'est juste ; là où il en existe un,
    # 15 sur 20 le sont. Exiger l'ancrage supprime donc 8 priorités fausses
    # sans en perdre une seule de juste — précision 54 % -> 75 %, rappel
    # inchangé. La détection couvre 30 des 31 marqueurs de référence, ce qui
    # rend l'exigence peu coûteuse.
    ecartees = [l for l in lignes if l.get("marker_n") not in classements]
    if ecartees:
        say(f"étape 3 : {len(ecartees)} préférence(s) écartée(s), "
            "aucun classement repéré à cet endroit")
        lignes = [l for l in lignes if l.get("marker_n") in classements]

    if lignes:
        nombre = _compile_preferences(assembler, lignes, rule_of_option, key_of)
        say(
            f"étape 3 : {len(lignes)} ligne(s) de préférence -> "
            f"{nombre} priorité(s) compilée(s)"
        )
    else:
        say("étape 3 : aucune préférence dans le texte (cas normal)")

    annotations = assembler.render()
    if not annotations.strip():
        say("étape 4 : aucune annotation retenue — sortie vide")
        return ""
    validate_annotation_graph(annotations)
    say("étape 4 : graphe assemblé et validé")
    return assembler.render_lpp() if output_format == "lpp" else annotations


def output_path_for(input_path: Path, output_format: str = "brat") -> Path:
    if input_path.suffix.lower() != ".txt":
        raise ValueError("le fichier d'entrée doit avoir l'extension .txt")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"format de sortie inconnu : {output_format!r}")
    return input_path.with_suffix(".ann" if output_format == "brat" else ".lpp")


def annotate_file(
    input_path: Path,
    model_name: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    report=None,
    etage: str = DEFAULT_ETAGE,
    votes: int | None = None,
    output_format: str = "brat",
    output_path: Path | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    output_path = output_path or output_path_for(input_path, output_format)
    if not input_path.is_file():
        raise ValueError(f"fichier introuvable : {input_path}")
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_input == resolved_output:
        raise ValueError("la sortie ne peut pas écraser le fichier source")
    repository_data = Path(__file__).resolve().parents[2] / "data"
    if repository_data.is_dir() and (
        resolved_input.is_relative_to(repository_data.resolve())
        or resolved_output.is_relative_to(repository_data.resolve())
    ):
        raise ValueError(
            "refus d'écrire dans le banc de référence data/ ; travaillez sur "
            "une copie ou choisissez --sortie hors du dépôt de données"
        )

    with input_path.open("r", encoding="utf-8", newline="") as source_file:
        texte = source_file.read()
    if not texte:
        raise ValueError("le fichier d'entrée est vide")

    annotations = annotate(
        texte,
        model_name,
        num_ctx,
        report,
        etage,
        votes,
        output_format=output_format,
        base_url=base_url,
        timeout=timeout,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Écriture atomique : une interruption ne doit jamais laisser un fichier
    # partiel que le système aval considérerait comme une extraction valide.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output_path.parent,
        prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
    ) as annotation_file:
        temporary_path = Path(annotation_file.name)
        annotation_file.write(annotations)
        if annotations:
            annotation_file.write("\n")
        annotation_file.flush()
        os.fsync(annotation_file.fileno())
    try:
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrait un graphe LPP/Gorgias depuis un fichier texte."
    )
    parser.add_argument("fichier", type=Path, help="fichier .txt à annoter")
    parser.add_argument(
        "--modele",
        default=DEFAULT_MODEL,
        help=f"modèle Ollama à utiliser (défaut : {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--contexte",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=f"taille de la fenêtre de contexte (défaut : {DEFAULT_NUM_CTX})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=("URL du serveur Ollama, local ou distant "
              f"(défaut : {DEFAULT_BASE_URL}; variable GORGIAS_OLLAMA_URL)"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"délai maximal d'un appel Ollama en secondes (défaut : {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--etage",
        choices=sorted(ETAGES_RELATION),
        default=DEFAULT_ETAGE,
        help=(
            "implémentation de l'étape 2 : \"groupe\" interroge par lot de "
            "conclusions, \"hybride\" vérifie ensuite ses candidats par paire, "
            "\"paires\" énumère toutes les paires à portée "
            f"(défaut : {DEFAULT_ETAGE})"
        ),
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=VOTES,
        help=(
            "tirages par appel, vote par étiquette/relation ; au-delà de 1, "
            "absorbe une partie de la variance au prix du temps de calcul "
            f"(défaut : {VOTES})"
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=sorted(OUTPUT_FORMATS),
        default="brat",
        help="sortie brat .ann ou composants LPP/Gorgias .lpp (défaut : brat)",
    )
    parser.add_argument(
        "--sortie", type=Path, default=None,
        help="chemin de sortie explicite (sinon, écrit à côté de l'entrée)",
    )
    parser.add_argument(
        "--silencieux",
        action="store_true",
        help="n'affiche pas la progression des étapes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = None if args.silencieux else (lambda m: print(m, file=sys.stderr))
    try:
        output_path = annotate_file(
            input_path=args.fichier,
            model_name=args.modele,
            num_ctx=args.contexte,
            report=report,
            etage=args.etage,
            votes=args.votes,
            output_format=args.output_format,
            output_path=args.sortie,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Erreur pendant l'appel à Ollama : {error}", file=sys.stderr)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
