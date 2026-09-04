FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GORGIAS_OLLAMA_URL=http://ollama:11434 \
    GORGIAS_MODEL=qwen3:8b

RUN useradd --create-home --uid 10001 gorgias
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]" \
    && python -m spacy download fr_core_news_lg \
    && pip cache purge

USER 10001
EXPOSE 8000
CMD ["uvicorn", "gorgias.service:api", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
