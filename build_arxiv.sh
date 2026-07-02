#!/bin/bash
# Build a plain arXiv-style PDF and TeX source from the JOSS manuscript.
#
# This is a local formatting helper only. It does not submit or upload anything.
#
# Output:
#   - paper/arxiv/ssys_arxiv.md
#   - paper/arxiv/ssys_arxiv.pdf
#   - paper/arxiv/ssys_arxiv.tex

set -euo pipefail

OUTDIR="paper/arxiv"
BODY="$OUTDIR/body.md"
MANUSCRIPT="$OUTDIR/ssys_arxiv.md"
PDF="$OUTDIR/ssys_arxiv.pdf"
TEX="$OUTDIR/ssys_arxiv.tex"

mkdir -p "$OUTDIR"

awk '
  BEGIN { in_yaml = 0; seen_yaml = 0; body = 0 }
  /^---$/ && seen_yaml == 0 { in_yaml = 1; seen_yaml = 1; next }
  /^---$/ && in_yaml == 1 { in_yaml = 0; body = 1; next }
  body == 1 { print }
' paper/paper.md > "$BODY"

cat > "$MANUSCRIPT" <<'EOF'
---
title: 'ssys: Exact algebraic recasting of ODE models into S-system or GMA form'
author: 'William S. Hlavacek'
date: '2026-07-02'
bibliography: '../paper.bib'
link-citations: true
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{hyperref}
---

EOF

cat "$BODY" >> "$MANUSCRIPT"

pandoc "$MANUSCRIPT" \
  --standalone \
  --citeproc \
  --pdf-engine=xelatex \
  --resource-path="paper:paper/arxiv" \
  -o "$PDF"

pandoc "$MANUSCRIPT" \
  --standalone \
  --citeproc \
  --pdf-engine=xelatex \
  --resource-path="paper:paper/arxiv" \
  -o "$TEX"

rm "$BODY"

echo "Built plain arXiv-style manuscript:"
echo "  $PDF"
echo "  $TEX"
