# ------------------------------- Builder Stage ------------------------------- #
FROM python:3.13-bookworm AS builder

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install UV
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY ./pyproject.toml .

RUN uv venv && uv sync

# ------------------------------- Production Stage ------------------------------- #
FROM python:3.13-slim-bookworm AS production

WORKDIR /app

COPY assets assets
COPY doc2image doc2image
COPY .env .env
COPY --from=builder /app/.venv .venv

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

VOLUME ["/app/data"]

CMD ["python", "-m", "streamlit", "run", "doc2image/ui/app/Home.py", "--server.port=8000"]