# Changelog

## Unreleased

- `smmkit serve`: read-only HTTP service with `/`, `/healthz` and `/api/metrics`.
- Multi-stage `Dockerfile` (non-root runtime, health check, OCI labels).
- `Jenkinsfile`: lint, unit tests with JUnit and coverage, image build, smoke test; push and
  rolling deployment to a kind Kubernetes cluster from `main`.
- Jenkins as code in `ci/jenkins/` and Kubernetes manifests in `deploy/`.

## 0.1.0 — 2026-10-08

- Anonymisation of comments (URLs, e-mails, Kazakhstani phone numbers, @handles) and salted author pseudonyms.
- CSV loaders for posts and comments with schema validation and row-level error messages.
- Engagement rate by channel, format, weekday and hour; recommended publication slots.
- Lexicon sentiment baseline for Russian and Kazakh with pre- and post-negation; evaluation with
  accuracy, per-class precision/recall/F1, macro-F1 and confusion matrix.
- `smmkit` command-line tool: `metrics`, `evaluate`, `report` (markdown, optional PNG plots).
- Synthetic sample dataset and its seeded generator.
- GitHub Actions: lint, test matrix Python 3.9–3.12 with coverage gate, build; release on tag.
