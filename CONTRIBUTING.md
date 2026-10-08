# Contributing

1. Open an issue first and describe the research question the change helps to answer.
2. Work on a branch: `feature/<short-name>` or `fix/<short-name>`.
3. Set up the environment and run the checks before pushing:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev,plot]"
   ruff check . && ruff format --check .
   pytest --cov=smmkit
   ```

4. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `data:`.
5. Open a pull request; CI must be green before merge.

**Data rule.** Never commit real comments, usernames, phone numbers or e-mails, even in tests.
Use `scripts/make_sample_data.py` or write synthetic examples.
