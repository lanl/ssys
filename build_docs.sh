#!/bin/bash
# Build PDF documentation from markdown files
# Uses XeLaTeX for Unicode support (Greek letters, math symbols)
# Specifies fonts with good Unicode coverage

set -e  # Exit on error

# Common options
COMMON_OPTS="--pdf-engine=xelatex --toc --toc-depth=3 -V colorlinks=true -V geometry:margin=0.75in"

# Check if DejaVu fonts are available
if fc-list 2>/dev/null | grep -qi "DejaVu"; then
    echo "Using DejaVu fonts (best Unicode support)"
    MAINFONT="DejaVu Serif"
    SANSFONT="DejaVu Sans"
    MONOFONT="DejaVu Sans Mono"
else
    echo "DejaVu fonts not found, using macOS system fonts"
    echo "For better Unicode support, run: brew install --cask font-dejavu"
    MAINFONT="Helvetica Neue"
    SANSFONT="Helvetica Neue"
    MONOFONT="Menlo"
fi

echo "Building README.pdf..."
pandoc README.md -o README.pdf $COMMON_OPTS \
    -V mainfont="$MAINFONT" \
    -V sansfont="$SANSFONT" \
    -V monofont="$MONOFONT"

echo "Building RECASTING.pdf..."
pandoc RECASTING.md -o RECASTING.pdf $COMMON_OPTS \
    -V mainfont="$MAINFONT" \
    -V sansfont="$SANSFONT" \
    -V monofont="$MONOFONT"

echo "Building TEST_MODELS.pdf..."
pandoc TEST_MODELS.md -o TEST_MODELS.pdf $COMMON_OPTS \
    -V mainfont="$MAINFONT" \
    -V sansfont="$SANSFONT" \
    -V monofont="$MONOFONT"

echo ""
echo "Documentation PDFs generated:"
ls -la README.pdf RECASTING.pdf TEST_MODELS.pdf
