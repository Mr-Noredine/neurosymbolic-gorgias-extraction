"""Sélection MÉCANIQUE du jeu hors échantillon, depuis EUR-Lex.

La règle de sélection était décrite en prose dans `SOURCE.json`. Une prose ne
se rejoue pas : ce script l'exécute, de sorte que le jeu soit reconstructible
par un tiers et que l'on puisse en tirer un LOT SUIVANT sans arbitraire.

    python3 data/horschantillon/selectionner.py --debut 1  --nombre 10
    python3 data/horschantillon/selectionner.py --debut 11 --nombre 10

La règle, inchangée : paragraphe NUMÉROTÉ d'article, 160 à 420 caractères,
contenant un marqueur conditionnel, phrase autonome — pris dans l'ORDRE du
document. Aucun choix humain n'intervient : seul le rang varie.

Dépendance : la bibliothèque standard seule (`html.parser`), pour que la
reproduction ne réclame rien d'autre que Python.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

URL = ("https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/"
       "?uri=CELEX:32016R0679")

MIN_CAR, MAX_CAR = 160, 420
# Marqueurs conditionnels — les mêmes mots-outils que le pipeline reconnaît.
CONDITIONNEL = re.compile(
    r"\b(?:si|s'il|s'ils|s'elle|s'elles|lorsque|lorsqu'|quand|dès\s+que|"
    r"en\s+cas\s+de|à\s+moins\s+que)\b", re.I)
# « 1.   texte » : un paragraphe numéroté d'article.
NUMEROTE = re.compile(r"^\s*(\d{1,2})\.\s+\S")


class Texte(HTMLParser):
    """Extrait le texte des paragraphes, en ignorant scripts et styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphes: list[str] = []
        self._tampon: list[str] = []
        self._muet = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._muet += 1
        elif tag == "p":
            self._tampon = []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._muet = max(0, self._muet - 1)
        elif tag == "p":
            self.paragraphes.append("".join(self._tampon))
            self._tampon = []

    def handle_data(self, data):
        if not self._muet:
            self._tampon.append(data)


def phrase_autonome(texte: str) -> bool:
    """Une phrase entière, pas un fragment d'énumération à puces."""
    return texte.rstrip().endswith(".") and ";" not in texte[-2:]


def selectionner(brut: str) -> list[str]:
    analyseur = Texte()
    analyseur.feed(brut)
    retenus: list[str] = []
    for paragraphe in analyseur.paragraphes:
        texte = html.unescape(paragraphe).strip()
        # EUR-Lex emploie des espaces insécables : elles font partie du texte
        # verbatim et sont conservées telles quelles.
        if not NUMEROTE.match(texte.replace(" ", " ")):
            continue
        if not MIN_CAR <= len(texte) <= MAX_CAR:
            continue
        if not CONDITIONNEL.search(texte.replace(" ", " ")):
            continue
        if not phrase_autonome(texte):
            continue
        if texte in retenus:
            continue
        retenus.append(texte)
    return retenus


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--debut", type=int, default=1,
                         help="rang du premier paragraphe retenu (1 = le tout premier)")
    parseur.add_argument("--nombre", type=int, default=10)
    parseur.add_argument("--sortie", type=Path, default=None,
                         help="dossier où écrire les .txt ; sans lui, affichage seul")
    parseur.add_argument("--prefixe", default="hs")
    parseur.add_argument(
        "--source-html", type=Path, default=None,
        help="copie HTML locale déjà gelée ; évite qu'une évolution d'EUR-Lex change le lot",
    )
    arguments = parseur.parse_args()

    if arguments.source_html is not None:
        octets = arguments.source_html.read_bytes()
    else:
        requete = urllib.request.Request(
            URL,
            headers={
                "User-Agent": "neurosymbolic-gorgias-extraction/1.0 (+benchmark reproductible)",
                "Accept-Language": "fr",
            },
        )
        with urllib.request.urlopen(requete, timeout=180) as reponse:
            octets = reponse.read()
    brut = octets.decode("utf-8", errors="replace")
    empreinte = hashlib.sha256(octets).hexdigest()

    retenus = selectionner(brut)
    print(f"paragraphes retenus par la règle : {len(retenus)}")
    print(f"sha256 de la page source : {empreinte[:16]}")

    tranche = retenus[arguments.debut - 1: arguments.debut - 1 + arguments.nombre]
    for rang, texte in enumerate(tranche, start=arguments.debut):
        nom = f"{arguments.prefixe}{rang:02d}"
        apercu = unicodedata.normalize("NFC", texte)[:96].replace(" ", " ")
        print(f"  {nom}  {len(texte):3d} car.  {apercu}…")
        if arguments.sortie:
            arguments.sortie.mkdir(parents=True, exist_ok=True)
            # Saut de ligne final : les dix premiers documents en portent un,
            # et la reproduction doit être exacte à l'octet près.
            (arguments.sortie / f"{nom}.txt").write_text(
                texte + "\n", encoding="utf-8")

    if arguments.sortie:
        (arguments.sortie / "SELECTION.json").write_text(json.dumps({
            "source": URL,
            "sha256_page": empreinte,
            "regle": ("paragraphe numéroté d'article, "
                      f"{MIN_CAR}-{MAX_CAR} car., marqueur conditionnel, "
                      "phrase autonome ; pris dans l'ordre du document"),
            "rang_debut": arguments.debut,
            "nombre": len(tranche),
            "source_html": str(arguments.source_html) if arguments.source_html else None,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
