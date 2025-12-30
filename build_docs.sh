#!/bin/bash
# Build PDF documentation from markdown files
# Uses XeLaTeX for Unicode support (Greek letters, math symbols)

OPTS="--pdf-engine=xelatex --toc --toc-depth=3 -V geometry:margin=1in -V colorlinks=true"

pandoc README.md    -o README.pdf    $OPTS
pandoc RECASTING.md -o RECASTING.pdf $OPTS
pandoc TEST_MODELS.md -o TEST_MODELS.pdf $OPTS

echo "Documentation PDFs generated:"
ls -la *.pdf
