#!/usr/bin/env bash
# Rebuild everything from the PDF: extraction -> corpus -> index.
set -euo pipefail
cd "$(dirname "$0")/.."

PDF="${1:-vachanamrut_-_english.pdf}"
[ -f "$PDF" ] || { echo "PDF not found: $PDF" >&2; exit 1; }

echo "==> Extracting text"
python3 scripts/extract_pdf.py --pdf "$PDF"
echo "==> Extracting glossary and endnotes"
python3 scripts/extract_reference.py --pdf "$PDF"
echo "==> Building corpus"
python3 scripts/build_corpus.py
echo "==> Building embeddings (if configured)"
python3 scripts/build_embeddings.py
echo "==> Evaluating"
python3 scripts/evaluate.py
