#!/bin/bash
# One-time publication to GitHub.
#   1. Create an EMPTY public repository named smm-analytics-toolkit on github.com
#      (no README, no license, no .gitignore — they are already here).
#   2. Run:  scripts/publish.sh <your-github-username>
set -euo pipefail
cd "$(dirname "$0")/.."
OWNER="${1:?usage: scripts/publish.sh <github-username>}"

if grep -q "OWNER/smm-analytics-toolkit" README.md CITATION.cff; then
  sed -i '' "s#OWNER/smm-analytics-toolkit#${OWNER}/smm-analytics-toolkit#g" README.md CITATION.cff
  git add README.md CITATION.cff
  git commit -q -m "docs: set repository owner in badges and citation"
fi

git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${OWNER}/smm-analytics-toolkit.git"
git push -u origin main
git push origin --tags
echo "Published: https://github.com/${OWNER}/smm-analytics-toolkit"
echo "CI runs:   https://github.com/${OWNER}/smm-analytics-toolkit/actions"
