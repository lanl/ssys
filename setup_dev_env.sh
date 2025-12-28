#!/bin/bash
# setup_dev_env.sh - Create ssys_dev development environment using uv
#
# Usage:
#   ./setup_dev_env.sh           # Create environment (interactive)
#   ./setup_dev_env.sh --force   # Overwrite existing environment without prompting
#
# After running, activate with:
#   source ssys_dev/bin/activate
#
# Or use the activate_dev_env.sh script for convenient activation.

set -e

ENV_NAME="ssys_dev"
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Create the ssys_dev development environment using uv."
            echo ""
            echo "Options:"
            echo "  --force, -f   Overwrite existing environment without prompting"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "After setup, activate the environment with:"
            echo "  source ssys_dev/bin/activate"
            echo ""
            echo "Or use ./activate_dev_env.sh for convenient activation."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check for existing environment
if [ -d "$ENV_NAME" ]; then
    if [ "$FORCE" = true ]; then
        echo "Removing existing $ENV_NAME directory (--force)..."
        rm -rf "$ENV_NAME"
    else
        echo "================================================"
        echo "Warning: $ENV_NAME environment already exists!"
        echo "================================================"
        echo ""
        echo "Options:"
        echo "  1) To ACTIVATE the existing environment:"
        echo "     source $ENV_NAME/bin/activate"
        echo ""
        echo "  2) To RECREATE the environment, run:"
        echo "     $0 --force"
        echo ""
        read -p "Do you want to remove and recreate the environment? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY])
                echo "Removing existing $ENV_NAME directory..."
                rm -rf "$ENV_NAME"
                ;;
            *)
                echo "Aborted. Use 'source $ENV_NAME/bin/activate' to use the existing environment."
                exit 0
                ;;
        esac
    fi
fi

echo "================================================"
echo "Setting up $ENV_NAME environment with uv"
echo "================================================"

# Create virtual environment with Python 3.12
# This ensures we get a native Python with compatible libroadrunner wheels
# (bypasses any conda/Rosetta Python installations)
echo "Creating virtual environment with Python 3.12..."
uv venv --python 3.12 "$ENV_NAME"

# Activate environment
echo "Activating environment..."
source "$ENV_NAME/bin/activate"

# Install package with dev extras and JAX
# Note: libroadrunner, antimony, and python-libsbml are now REQUIRED
# core dependencies (SBML-first architecture), not optional extras.
# JAX is optional for end users but included in dev setup for faster validation.
echo "Installing ssys with dev tools and JAX..."
uv pip install -e ".[dev,jax]"

# Verify installation
echo ""
echo "Verifying installation..."
if python -c "import ssys; print(f'ssys version: {ssys.__version__}')" 2>/dev/null; then
    echo "✓ ssys installed successfully"
else
    echo "✗ ssys import failed"
    exit 1
fi

if python -c "import roadrunner; print(f'libroadrunner version: {roadrunner.__version__}')" 2>/dev/null; then
    echo "✓ libroadrunner available"
else
    echo "✗ libroadrunner NOT available - this is required!"
    exit 1
fi

if python -c "import libsbml; print(f'libsbml version: {libsbml.getLibSBMLDottedVersion()}')" 2>/dev/null; then
    echo "✓ libsbml available"
else
    echo "✗ libsbml NOT available - this is required!"
    exit 1
fi

if python -c "import jax; print(f'jax version: {jax.__version__}')" 2>/dev/null; then
    echo "✓ jax available (fast numerical validation)"
else
    echo "⚠ jax NOT available (will use symbolic Jacobian fallback)"
fi

# Register Jupyter kernel for this environment
echo ""
echo "Registering Jupyter kernel..."
if python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" 2>/dev/null; then
    echo "✓ Jupyter kernel '$ENV_NAME' registered"
else
    echo "⚠ Jupyter kernel registration failed (ipykernel may not be installed)"
    echo "  You can manually register later with:"
    echo "  python -m ipykernel install --user --name $ENV_NAME --display-name \"Python ($ENV_NAME)\""
fi

echo ""
echo "================================================"
echo "Environment setup complete!"
echo "================================================"
echo ""
echo "To activate the environment:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "Or use the convenience script:"
echo "  ./activate_dev_env.sh"
echo ""
echo "Quick start commands:"
echo "  ssys-recast --help              # Show CLI help"
echo "  python recast_models.py test_models3  # Run test_models3 validation"
echo "  pytest tests/                   # Run unit tests"
echo ""
