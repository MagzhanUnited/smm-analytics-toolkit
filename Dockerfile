# syntax=docker/dockerfile:1
# Multi-stage build: the wheel is built in a throw-away stage, the runtime image holds only
# the installed package, the sample data and a non-root user.
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="smm-analytics-toolkit" \
      org.opencontainers.image.description="Engagement metrics and sentiment baseline for SMM research" \
      org.opencontainers.image.source="https://github.com/MagzhanUnited/smm-analytics-toolkit" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 smm
WORKDIR /app
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
COPY data/sample ./data/sample
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"
ENTRYPOINT ["smmkit"]
CMD ["serve", "--posts", "data/sample/posts.csv", "--comments", "data/sample/comments_labelled.csv"]
