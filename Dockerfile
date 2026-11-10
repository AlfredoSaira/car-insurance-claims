FROM python:3.11-slim-bookworm

# libgomp is needed by XGBoost (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# Install only the runtime dependencies (no Jupyter / plotting) from the lock file
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY common.py predict.py ./
COPY models/model.bin models/model.bin

EXPOSE 9696
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
