# knowbase service image.
#
# The service refuses to bind a non-loopback address without KB_API_TOKEN, and inside a
# container 0.0.0.0 is the only useful bind, so the container will not start without the
# token in its environment. That is deliberate: there is no unauthenticated mode to ship
# by accident.
#
#   docker build -t knowbase .
#   docker run --rm -p 8080:8080 -e KB_API_TOKEN=... -v ./corpus:/data/corpus knowbase

FROM python:3.13-slim AS build

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Builds the project wheel and collects the dependency wheels; nothing here needs a compiler,
# so the runtime stage can stay free of pip caches and build context.
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.13-slim

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/corpus \
    && chown -R app:app /data

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

USER app
WORKDIR /home/app

EXPOSE 8080

# /health is an open route (no bearer token needed), so the probe carries no secret.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).status == 200 else 1)"]

CMD ["python", "-m", "knowbase", "serve", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--store", "sqlite", "--cache", "sqlite", \
     "--sqlite-path", "/data/knowbase.sqlite3", \
     "--corpus-root", "/data/corpus"]
