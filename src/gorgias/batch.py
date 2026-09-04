"""Traitement de corpus avec manifeste d'audit et sorties atomiques."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from . import app


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def process_directory(
    source_directory: Path,
    output_directory: Path,
    *,
    model_name: str = app.DEFAULT_MODEL,
    num_ctx: int = app.DEFAULT_NUM_CTX,
    etage: str = app.DEFAULT_ETAGE,
    votes: int = app.VOTES,
    output_format: str = "lpp",
    base_url: str = app.DEFAULT_BASE_URL,
    timeout: float = app.DEFAULT_TIMEOUT,
    recursive: bool = False,
    fail_fast: bool = False,
    annotator: Callable[[str], str] | None = None,
) -> dict:
    """Annote tous les ``.txt`` d'un dossier et rend un manifeste vérifiable.

    ``annotator`` est un point d'injection pour les tests et les intégrations.
    En production, un seul client Ollama est réutilisé pour tout le lot.
    """
    source_directory = source_directory.resolve()
    output_directory = output_directory.resolve()
    if not source_directory.is_dir():
        raise ValueError(f"dossier source introuvable : {source_directory}")
    if source_directory == output_directory:
        raise ValueError("les dossiers source et sortie doivent être distincts")
    if output_format not in app.OUTPUT_FORMATS:
        raise ValueError(f"format de sortie inconnu : {output_format!r}")

    pattern = "**/*.txt" if recursive else "*.txt"
    sources = sorted(source_directory.glob(pattern))
    if not sources:
        raise ValueError(f"aucun fichier .txt dans {source_directory}")

    model = None
    if annotator is None:
        model = app._create_model(model_name, num_ctx, base_url, timeout)

        def annotator(text: str) -> str:
            return app.annotate(
                text, model_name=model_name, num_ctx=num_ctx, etage=etage,
                votes=votes, model=model, output_format=output_format,
            )

    extension = ".ann" if output_format == "brat" else ".lpp"
    documents: list[dict] = []
    started = time.monotonic()
    for source in sources:
        relative = source.relative_to(source_directory)
        destination = (output_directory / relative).with_suffix(extension)
        before_calls = app.COMPTEUR_APPELS.get("total", 0)
        document_started = time.monotonic()
        try:
            text = source.read_text(encoding="utf-8")
            if not text:
                raise ValueError("fichier vide")
            result = annotator(text)
            _atomic_write(destination, result + ("\n" if result else ""))
        except Exception as error:
            documents.append({
                "source": relative.as_posix(), "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "duration_s": round(time.monotonic() - document_started, 3),
                "llm_calls": app.COMPTEUR_APPELS.get("total", 0) - before_calls,
            })
            if fail_fast:
                break
            continue

        documents.append({
            "source": relative.as_posix(),
            "output": destination.relative_to(output_directory).as_posix(),
            "status": "ok", "empty": not bool(result),
            "source_sha256": _sha256(text), "output_sha256": _sha256(result),
            "duration_s": round(time.monotonic() - document_started, 3),
            "llm_calls": app.COMPTEUR_APPELS.get("total", 0) - before_calls,
        })

    manifest = {
        "schema_version": 1,
        "configuration": {
            "model": model_name, "context": num_ctx, "stage": etage,
            "votes": votes, "format": output_format, "base_url": base_url,
            "language": "fr", "completeness_guaranteed": False,
        },
        "summary": {
            "total": len(documents),
            "succeeded": sum(item["status"] == "ok" for item in documents),
            "failed": sum(item["status"] == "error" for item in documents),
            "empty": sum(item.get("empty", False) for item in documents),
            "duration_s": round(time.monotonic() - started, 3),
        },
        "documents": documents,
    }
    _atomic_write(
        output_directory / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("sortie", type=Path)
    parser.add_argument("--modele", default=app.DEFAULT_MODEL)
    parser.add_argument("--contexte", type=int, default=app.DEFAULT_NUM_CTX)
    parser.add_argument("--etage", choices=sorted(app.ETAGES_RELATION),
                        default=app.DEFAULT_ETAGE)
    parser.add_argument("--votes", type=int, default=app.VOTES)
    parser.add_argument("--format", dest="output_format",
                        choices=sorted(app.OUTPUT_FORMATS), default="lpp")
    parser.add_argument("--base-url", default=app.DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=app.DEFAULT_TIMEOUT)
    parser.add_argument("--recursif", action="store_true")
    parser.add_argument("--arreter-sur-erreur", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        manifest = process_directory(
            arguments.source, arguments.sortie, model_name=arguments.modele,
            num_ctx=arguments.contexte, etage=arguments.etage,
            votes=arguments.votes, output_format=arguments.output_format,
            base_url=arguments.base_url, timeout=arguments.timeout,
            recursive=arguments.recursif, fail_fast=arguments.arreter_sur_erreur,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Erreur : {error}", file=os.sys.stderr)
        return 2
    print(arguments.sortie / "manifest.json")
    return 1 if manifest["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
