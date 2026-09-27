"""Harnais de mesure : exécute le pipeline sur un corpus, puis le score.

Deux commandes séparées à dessein. `executer` est cher et non reproductible
(l'inférence varie) ; `scorer` est gratuit et déterministe. On garde donc les
sorties brutes sur disque, ce qui permet de re-scorer un run passé avec un
scoreur corrigé sans repayer l'inférence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
# GORGIAS_SRC permet de mesurer une version FIGÉE du code pendant qu'on
# modifie l'arbre de travail : sans cela, la mesure de départ et celle
# d'arrivée ne pourraient pas coexister dans la même session.
sys.path.insert(0, os.environ.get("GORGIAS_SRC", str(RACINE / "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scorer  # noqa: E402


def documents(corpus: Path) -> list[Path]:
    return sorted(corpus.glob("*.txt"))


def _annoter(chemin: Path, options) -> dict:
    from gorgias import app

    depart = time.monotonic()
    avant = app.COMPTEUR_APPELS["total"]
    try:
        sortie = app.annotate(
            chemin.read_text(encoding="utf-8"),
            model_name=options.modele,
            etage=options.etage,
            output_format="brat",
            votes=options.votes,
        )
        statut = "ok"
    except Exception as erreur:                       # noqa: BLE001
        sortie, statut = "", f"{type(erreur).__name__}: {erreur}"
    return {
        "document": chemin.name,
        "statut": statut,
        "secondes": round(time.monotonic() - depart, 1),
        "appels": app.COMPTEUR_APPELS["total"] - avant,
        "annotation": sortie,
    }


def executer(options) -> int:
    sortie = Path(options.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    for corpus in options.corpus:
        corpus = Path(corpus)
        cible = sortie / corpus.name
        cible.mkdir(parents=True, exist_ok=True)
        fichiers = [c for c in documents(corpus)
                    if options.rejouer or not (cible / f"{c.stem}.ann").exists()]
        if not fichiers:
            print(f"{corpus.name} : déjà complet", flush=True)
            continue
        print(f"{corpus.name} : {len(fichiers)} document(s) à annoter", flush=True)
        depart = time.monotonic()
        with ThreadPoolExecutor(max_workers=options.jobs) as pool:
            for index, resultat in enumerate(
                    pool.map(lambda c: _annoter(c, options), fichiers), start=1):
                nom = Path(resultat["document"]).stem
                (cible / f"{nom}.ann").write_text(
                    resultat["annotation"], encoding="utf-8")
                resultat.pop("annotation")
                (cible / f"{nom}.json").write_text(
                    json.dumps(resultat, ensure_ascii=False, indent=1),
                    encoding="utf-8")
                print(f"  [{index}/{len(fichiers)}] {nom} "
                      f"{resultat['secondes']}s {resultat['appels']} appel(s)"
                      + ("" if resultat["statut"] == "ok"
                         else f"  ({resultat['statut']})"), flush=True)
        print(f"{corpus.name} : {round(time.monotonic() - depart)}s au total",
              flush=True)
    return 0


def _lire_meta(cible: Path, nom: str) -> dict:
    chemin = cible / f"{nom}.json"
    if not chemin.exists():
        return {}
    return json.loads(chemin.read_text(encoding="utf-8"))


def scorer_corpus(corpus: Path, cible: Path, strict: bool = False) -> dict:
    scores, par_document, secondes, appels = [], [], 0.0, 0
    for source in documents(corpus):
        reference = (corpus / f"{source.stem}.ann")
        produit = cible / f"{source.stem}.ann"
        if not produit.exists():
            continue
        score = scorer.scorer_document(
            reference.read_text(encoding="utf-8") if reference.exists() else "",
            produit.read_text(encoding="utf-8"),
            strict=strict,
        )
        meta = _lire_meta(cible, source.stem)
        secondes += meta.get("secondes", 0.0)
        appels += meta.get("appels", 0)
        scores.append(score)
        par_document.append({"document": source.stem, **score,
                             "statut": meta.get("statut", "?")})
    return {"documents": len(scores), "secondes": round(secondes),
            "appels": appels, "total": scorer.agreger(scores) if scores else {},
            "par_document": par_document}


def pollution(corpus: Path, cible: Path) -> dict:
    """Récits négatifs : tout ce qui est produit est un faux positif."""
    pollues, aretes, total = [], 0, 0
    for source in documents(corpus):
        produit = cible / f"{source.stem}.ann"
        if not produit.exists():
            continue
        total += 1
        analyse = scorer.analyser(produit.read_text(encoding="utf-8"))
        if analyse.evenements or analyse.entites:
            pollues.append(source.stem)
            aretes += len(analyse.evenements)
    return {"documents": total, "pollues": len(pollues), "aretes": aretes,
            "noms": pollues}


def _ligne(nom: str, mesure: dict) -> str:
    if not mesure:
        return f"{nom:<24} —"
    return (f"{nom:<24} P {mesure['precision']*100:5.1f}%  "
            f"R {mesure['rappel']*100:5.1f}%  F1 {mesure['f1']*100:5.1f}%   "
            f"{mesure['justes']:>3}/{mesure['produits']:>3} produites, "
            f"{mesure['attendus']:>3} attendues")


def scorer_tout(options) -> int:
    sortie = Path(options.sortie)
    rapport: dict = {"sortie": str(sortie)}
    for corpus in options.corpus:
        corpus = Path(corpus)
        cible = sortie / corpus.name
        if not cible.exists():
            continue
        if corpus.name.startswith("narratif"):
            rapport[corpus.name] = pollution(corpus, cible)
            mesure = rapport[corpus.name]
            print(f"\n=== {corpus.name} (récits négatifs) ===")
            print(f"  {mesure['pollues']}/{mesure['documents']} récits pollués, "
                  f"{mesure['aretes']} arête(s) inventée(s)")
            if mesure["noms"]:
                print("  " + ", ".join(mesure["noms"]))
            continue
        mesure = scorer_corpus(corpus, cible, strict=options.strict)
        rapport[corpus.name] = mesure
        print(f"\n=== {corpus.name} "
              f"({mesure['documents']} documents, {mesure['secondes']}s, "
              f"{mesure['appels']} appels) ===")
        for cle, nom in (("entites", "entités (toutes)"),
                         ("Context", "  Context"),
                         ("Option", "  Option"),
                         ("Marker", "  Marker"),
                         ("rule", "règles"),
                         ("prefer", "prefer"),
                         ("meta_prefer", "meta_prefer"),
                         ("priorite", "priorités (les deux)"),
                         ("priorite_sans_ancrage", "  sans le When")):
            print("  " + _ligne(nom, mesure["total"].get(cle, {})))
    if options.rapport:
        Path(options.rapport).write_text(
            json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nrapport écrit : {options.rapport}")
    return 0


def construire() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(description=__doc__)
    sous = parseur.add_subparsers(dest="commande", required=True)

    run = sous.add_parser("executer", help="annoter un corpus")
    run.add_argument("sortie")
    run.add_argument("corpus", nargs="+")
    run.add_argument("--modele", default=os.environ.get("GORGIAS_MODEL", "qwen3.5:4b"))
    run.add_argument("--etage", default="hybride")
    run.add_argument("--votes", type=int, default=1)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--rejouer", action="store_true",
                     help="réannoter même si la sortie existe")
    run.set_defaults(fonction=executer)

    note = sous.add_parser("scorer", help="scorer un run existant")
    note.add_argument("sortie")
    note.add_argument("corpus", nargs="+")
    note.add_argument("--strict", action="store_true",
                      help="empans identiques au caractère près")
    note.add_argument("--rapport", default=None)
    note.set_defaults(fonction=scorer_tout)
    return parseur


def main(argv=None) -> int:
    options = construire().parse_args(argv)
    return options.fonction(options)


if __name__ == "__main__":
    raise SystemExit(main())
