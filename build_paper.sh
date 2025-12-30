#!/bin/bash
# Build JOSS-formatted PDF from paper.md using the official JOSS Docker container
# 
# Requirements:
#   - Docker Desktop installed and running
#   - Run from repository root directory
#
# Output:
#   - paper/paper.pdf  (JOSS-formatted paper)
#   - paper/jats/      (JATS XML for submission)

set -e

echo "Building JOSS paper..."
echo "Note: Requires Docker Desktop to be running"

docker run --rm \
  --volume "$PWD/paper":/data \
  --user "$(id -u)":"$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara

echo ""
echo "Build complete!"
echo "Output: paper/paper.pdf"
