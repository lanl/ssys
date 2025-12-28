#!/bin/bash
# activate_dev_env.sh - Activate the ssys_dev environment and run commands
#
# Usage:
#   ./activate_dev_env.sh              # Enter interactive shell with environment activated
#   ./activate_dev_env.sh COMMAND      # Run a command in the environment
#
# Examples:
#   ./activate_dev_env.sh                              # Start interactive session
#   ./activate_dev_env.sh ssys-recast --help           # Show CLI help
#   ./activate_dev_env.sh python recast_models.py DIR  # Run model validation
#   ./activate_dev_env.sh pytest tests/                # Run unit tests
#   ./activate_dev_env.sh python -c "import ssys"      # Quick Python test

ENV_NAME="ssys_dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

# Check if environment exists
if [ ! -d "$ENV_PATH" ]; then
    echo "================================================"
    echo "Error: $ENV_NAME environment not found!"
    echo "================================================"
    echo ""
    echo "The development environment has not been created yet."
    echo ""
    echo "To create it, run:"
    echo "  ./setup_dev_env.sh"
    echo ""
    exit 1
fi

# Check if activate script exists
if [ ! -f "$ENV_PATH/bin/activate" ]; then
    echo "Error: $ENV_PATH/bin/activate not found."
    echo "The environment may be corrupted. Try recreating it:"
    echo "  ./setup_dev_env.sh --force"
    exit 1
fi

# Disable venv's automatic PS1 modification
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Activate the environment (without PS1 modification)
source "$ENV_PATH/bin/activate"

# Function to show environment info
show_env_info() {
    echo "================================================"
    echo "ssys development environment activated"
    echo "================================================"
    
    # Python version
    local py_version=$(python --version 2>&1)
    echo "Python: $py_version"
    
    # ssys version
    local ssys_version=$(python -c "import ssys; print(ssys.__version__)" 2>/dev/null || echo "not installed")
    echo "ssys: $ssys_version"
    
    # libroadrunner status
    if python -c "import roadrunner" 2>/dev/null; then
        local rr_version=$(python -c "import roadrunner; print(roadrunner.__version__)" 2>/dev/null)
        echo "libroadrunner: $rr_version"
    else
        echo "libroadrunner: not available (RK4 fallback)"
    fi
    
    # JAX status - intentionally disabled (see DEVELOPMENT_NOTES.md)
    echo "jax: not installed (use of JAX is intentionally disabled)"
    
    echo ""
    echo "Available commands:"
    echo "  ssys-recast --help              # CLI help"
    echo "  python recast_models.py DIR     # Run model validation"
    echo "  pytest tests/                   # Run unit tests"
    echo ""
    echo "Type 'exit' or Ctrl+D to leave the environment."
    echo "================================================"
    echo ""
}

# If no arguments, start an interactive shell
if [ $# -eq 0 ]; then
    show_env_info
    
    # Create a minimal rcfile with a clean, consistent prompt
    # Format: (ssys_dev) hostname:directory user$
    RCFILE=$(mktemp)
    cat > "$RCFILE" << 'RCEOF'
# Set a clean prompt: (ssys_dev) hostname:dir user$
PS1='(ssys_dev) \h:\W \u\$ '
RCEOF

    exec bash --rcfile "$RCFILE"
else
    # Run the provided command
    exec "$@"
fi
