#!/bin/bash
# Build PDF documentation from markdown files
# Uses XeLaTeX for Unicode support (Greek letters, math symbols)
# Specifies fonts with good Unicode coverage (DejaVu or fallback to system fonts)

# Check if DejaVu fonts are available, otherwise use macOS system fonts
if fc-list | grep -qi "DejaVu"; then
    FONT_OPTS='-V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono"'
else
    # Fallback to macOS system fonts
    FONT_OPTS='-V mainfont="Helvetica Neue" -V sansfont="Helvetica Neue" -V monofont="Menlo"'
fi

OPTS="--pdf-engine=xelatex --toc --toc-depth=3 -V geometry:margin=1in -V colorlinks=true $FONT_OPTS"

pandoc README.md    -o README.pdf    $OPTS
pandoc RECASTING.md -o RECASTING.pdf $OPTS
pandoc TEST_MODELS.md -o TEST_MODELS.pdf $OPTS

echo "Documentation PDFs generated:"
ls -la *.pdf
