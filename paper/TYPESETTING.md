---
title: Typesetting Instructions
source: TYPESETTING.md
date: 2025-12-30
---

# Typesetting Instructions

This document describes how to convert the paper and documentation files to PDF format.

---

## Part 1: JOSS Paper (paper.md)

The Journal of Open Source Software (JOSS) requires a specific format for submissions.

### 1.1 YAML Metadata Header

JOSS requires a YAML metadata block at the beginning of `paper.md`.

**Key points that routinely trip people up:**
- **Date format** must be like `9 October 2024`
- **Authors** must include affiliations by index (you can use more detailed name parts if needed)
- Provide `bibliography: paper.bib` so citations resolve

**Minimal skeleton** (adapt to your content):

```yaml
---
title: "Your software name: a short descriptive tagline"
tags:
  - keyword1
  - keyword2
authors:
  - name: First Last
    affiliation: "1"
  - name: Second Last
    affiliation: "1,2"
affiliations:
  - index: 1
    name: Your institution
  - index: 2
    name: Second institution
date: 30 December 2025
bibliography: paper.bib
---
```

### 1.2 Paper Structure

JOSS papers are **250–1000 words**; anything "full length" is explicitly discouraged.

At minimum, JOSS expects sections covering:
- Summary
- Statement of need
- Acknowledgements
- References

**Common layout:**

```markdown
# Summary
# Statement of need
# Acknowledgements
# References
```

JOSS encourages headings starting at level 1 (`#`) and discourages deep nesting.

### 1.3 Citations and Figures (JOSS-flavored Pandoc)

- **Citations** are Pandoc-style: `[@Smith2020]` with entries in `paper.bib`
- **Cross-references** for figures/tables/equations use LaTeX-style `\label` / `\ref` / `\autoref`

---

## Part 2: Building the JOSS-style PDF on macOS

JOSS documents three compilation options; on a Mac, **Docker is the least painful** and matches what JOSS uses.

### 2.1 Install Docker Desktop

1. Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) (GUI installer)
2. Once installed, make sure Docker is running (icon in the menu bar)

### 2.2 File Organization

Put your paper files in a `paper/` folder:

```
repo/
  paper/
    paper.md
    paper.bib
    figures/
      fig1.png
```

### 2.3 Run the Official JOSS Docker Build Command

From your repo root:

```bash
docker run --rm \
  --volume "$PWD/paper":/data \
  --user "$(id -u)":"$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara
```

On success, you'll get:
- `paper/paper.pdf` (the formatted paper)
- `paper/paper.jats` (JATS XML, usually in a `jats/` subdirectory)

---

## Part 3: Optional GitHub Automation

If your repo is on GitHub, JOSS recommends using the **Open Journals draft action** to compile the PDF on every push.

A typical workflow uses `openjournals/openjournals-draft-action` with:
- `journal: joss`
- `paper-path: paper.md`

Example `.github/workflows/draft-pdf.yml`:

```yaml
name: Draft PDF
on: [push]
jobs:
  paper:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: openjournals/openjournals-draft-action@master
        with:
          journal: joss
          paper-path: paper/paper.md
      - uses: actions/upload-artifact@v4
        with:
          name: paper
          path: paper/paper.pdf
```

---

## Part 4: If You "Need More Space"

JOSS intentionally keeps the paper short. Extended documentation should go in your software docs, not the paper.

**Standard approach:**
- Keep `paper.md` tight (250–1000 words)
- Link to and cite richer docs, examples, benchmarks, or a preprint

---

## Part 5: Documentation PDFs (Generic)

For the documentation files (`README.md`, `RECASTING.md`, `TEST_MODELS.md`), use plain pandoc without JOSS styling.

### 5.1 Prerequisites

Install pandoc and a LaTeX distribution:

```bash
# macOS with Homebrew
brew install pandoc
brew install --cask mactex-no-gui   # or basictex for smaller install
```

### 5.2 Basic Conversion Commands

From the repository root:

```bash
# README.md → README.pdf
pandoc README.md -o README.pdf --pdf-engine=xelatex

# RECASTING.md → RECASTING.pdf  
pandoc RECASTING.md -o RECASTING.pdf --pdf-engine=xelatex

# TEST_MODELS.md → TEST_MODELS.pdf
pandoc TEST_MODELS.md -o TEST_MODELS.pdf --pdf-engine=xelatex
```

**Note:** The `--pdf-engine=xelatex` flag is required because the documentation files contain Unicode characters (Greek letters like ε, θ, mathematical symbols like ≥, ≤). XeLaTeX handles Unicode natively; the default pdflatex does not.

### 5.3 Enhanced Options

For better formatting with table of contents and margins:

```bash
pandoc README.md \
  -o README.pdf \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=3 \
  -V geometry:margin=1in \
  -V colorlinks=true

pandoc RECASTING.md \
  -o RECASTING.pdf \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=3 \
  -V geometry:margin=1in \
  -V colorlinks=true

pandoc TEST_MODELS.md \
  -o TEST_MODELS.pdf \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=3 \
  -V geometry:margin=1in \
  -V colorlinks=true
```

### 5.4 All-in-One Script

The repository includes a `build_docs.sh` script that automates PDF generation for all documentation files.

**Font requirements:** For best Unicode support (Greek letters, math symbols), install DejaVu fonts:
```bash
brew install --cask font-dejavu
```

If DejaVu is not installed, the script falls back to macOS system fonts (Helvetica Neue, Menlo), which have limited Unicode coverage and may show warnings.

**Run the script:**
```bash
chmod +x build_docs.sh
./build_docs.sh
```

This generates `README.pdf`, `RECASTING.pdf`, and `TEST_MODELS.pdf` in the repository root.

### 5.5 Notes on YAML Frontmatter

The documentation files include YAML frontmatter for PDF metadata:

```yaml
---
title: Test Model Collection
source: TEST_MODELS.md
date: 2025-12-30
---
```

Pandoc will use this metadata automatically. The `source` field helps track which markdown file generated each PDF.
