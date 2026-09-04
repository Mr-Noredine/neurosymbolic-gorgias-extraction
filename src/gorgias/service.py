"""API HTTP optionnelle pour déployer Gorgias derrière un reverse proxy."""
from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import time
from functools import lru_cache
from collections.abc import Sequence
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__, app as extraction


API_KEY = os.environ.get("GORGIAS_API_KEY")
MAX_TEXT_CHARS = int(os.environ.get("GORGIAS_MAX_TEXT_CHARS", "200000"))
MAX_CONCURRENCY = max(1, int(os.environ.get("GORGIAS_MAX_CONCURRENCY", "1")))
_capacity = asyncio.Semaphore(MAX_CONCURRENCY)


class ExtractionRequest(BaseModel):
    text: str = Field(min_length=1)
    format: str = "lpp"
    language: Literal["fr"] = "fr"


class ExtractionResponse(BaseModel):
    schema_version: int = 1
    result: str
    empty: bool
    source_sha256: str
    duration_s: float
    model: str
    format: str
    language: str = "fr"
    completeness_guaranteed: bool = False


def authenticate(x_api_key: str | None = Header(default=None)) -> None:
    """Authentification simple ; à compléter par l'IAM du reverse proxy."""
    if API_KEY is not None and (
        x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY)
    ):
        raise HTTPException(status_code=401, detail="invalid API key")


@lru_cache(maxsize=1)
def _model():
    return extraction._create_model(
        extraction.DEFAULT_MODEL, extraction.DEFAULT_NUM_CTX,
        extraction.DEFAULT_BASE_URL, extraction.DEFAULT_TIMEOUT,
    )


api = FastAPI(
    title="Neurosymbolic Gorgias Extraction",
    version=__version__,
    docs_url="/docs" if os.environ.get("GORGIAS_ENABLE_DOCS") == "1" else None,
    redoc_url=None,
)


@api.get("/healthz")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@api.get("/readyz")
async def readiness(_: None = Depends(authenticate)) -> dict:
    def check() -> None:
        from ollama import Client
        Client(host=extraction.DEFAULT_BASE_URL,
               timeout=extraction.DEFAULT_TIMEOUT).list()

    try:
        await asyncio.to_thread(check)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Ollama unavailable") from error
    return {"status": "ready", "model": extraction.DEFAULT_MODEL}


@api.post("/v1/extractions", response_model=ExtractionResponse)
async def extract(
    request: ExtractionRequest, _: None = Depends(authenticate),
) -> ExtractionResponse:
    if request.format not in extraction.OUTPUT_FORMATS:
        raise HTTPException(status_code=422, detail="format must be brat or lpp")
    if len(request.text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="text exceeds configured limit")

    started = time.monotonic()
    try:
        async with _capacity:
            result = await asyncio.to_thread(
                extraction.annotate, request.text,
                model_name=extraction.DEFAULT_MODEL,
                num_ctx=extraction.DEFAULT_NUM_CTX,
                etage=extraction.DEFAULT_ETAGE,
                votes=extraction.VOTES,
                model=_model(),
                output_format=request.format,
            )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="extraction backend unavailable") from error

    return ExtractionResponse(
        result=result, empty=not bool(result),
        source_sha256=hashlib.sha256(request.text.encode("utf-8")).hexdigest(),
        duration_s=round(time.monotonic() - started, 3),
        model=extraction.DEFAULT_MODEL, format=request.format,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Lance le serveur avec une configuration sûre par défaut."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run("gorgias.service:api", host=args.host, port=args.port,
                access_log=False)
    return 0
