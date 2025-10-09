#!/usr/bin/env bash
# SFT-Play Installation Script
# Supports: pip, uv, curl installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
INSTALL_METHOD="pip"
PACKAGE_EXTRAS="standard"
PYTHON_CMD="python3"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
SFT-Play Installation Script

Usage: ./install.sh [OPTIONS]

Options:
    -m, --method METHOD     Installation method: pip, uv (default: pip)
    -e, --extras EXTRAS     Package extras: minimal, standard, full (default: standard)
    -p, --python PYTHON     Python command (default: python3)
    -h, --help             Show this help message

Examples:
    ./install.sh                              # Standard installation with pip
    ./install.sh -m uv -e full                # Full installation with uv
    ./install.sh -e minimal                   # Minimal installation
    ./install.sh -e "gpu,distributed"         # Custom extras

Extras Available:
    minimal      - Core dependencies only
    standard     - Core + GPU + evaluation + monitoring (recommended)
    full         - All features
    gpu          - BitsAndBytes for QLoRA
    unsloth      - Unsloth for faster training
    distributed  - DeepSpeed for multi-GPU
    dapt         - Document processing for domain adaptation
    evaluation   - Evaluation metrics (ROUGE, BLEU)
    monitoring   - TensorBoard
    wandb        - Weights & Biases tracking
    dev          - Development tools

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--method)
            INSTALL_METHOD="$2"
            shift 2
            ;;
        -e|--extras)
            PACKAGE_EXTRAS="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check Python installation
log_info "Checking Python installation..."
if ! command -v $PYTHON_CMD &> /dev/null; then
    log_error "Python not found. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
log_info "Found Python $PYTHON_VERSION"

# Verify Python version >= 3.8
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    log_error "Python 3.8+ required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Install based on method
log_info "Installing sft-play with method: $INSTALL_METHOD"

case $INSTALL_METHOD in
    pip)
        if [ "$PACKAGE_EXTRAS" == "minimal" ]; then
            log_info "Installing minimal package..."
            $PYTHON_CMD -m pip install --upgrade pip
            $PYTHON_CMD -m pip install sft-play
        else
            log_info "Installing with extras: $PACKAGE_EXTRAS"
            $PYTHON_CMD -m pip install --upgrade pip
            $PYTHON_CMD -m pip install "sft-play[$PACKAGE_EXTRAS]"
        fi
        ;;
    uv)
        if ! command -v uv &> /dev/null; then
            log_warn "uv not found. Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi

        if [ "$PACKAGE_EXTRAS" == "minimal" ]; then
            log_info "Installing minimal package with uv..."
            uv pip install sft-play
        else
            log_info "Installing with extras: $PACKAGE_EXTRAS"
            uv pip install "sft-play[$PACKAGE_EXTRAS]"
        fi
        ;;
    *)
        log_error "Unknown installation method: $INSTALL_METHOD"
        log_error "Supported: pip, uv"
        exit 1
        ;;
esac

# Verify installation
log_info "Verifying installation..."
if $PYTHON_CMD -c "import sft_play" 2>/dev/null; then
    VERSION=$($PYTHON_CMD -c "import sft_play; print(sft_play.__version__)")
    log_info "✓ SFT-Play v$VERSION installed successfully!"
else
    log_error "Installation verification failed"
    exit 1
fi

# Check CLI availability
if command -v sft-play &> /dev/null; then
    log_info "✓ CLI command 'sft-play' is available"
else
    log_warn "CLI command not found in PATH. You may need to restart your shell."
fi

# Show next steps
cat << EOF

${GREEN}Installation complete!${NC}

Next steps:
  1. Create a training config (see examples in repo)
  2. Run: sft-play train --config config.yaml
  3. Evaluate: sft-play eval --model ./outputs/checkpoints/final
  4. Infer: sft-play infer --model ./outputs/checkpoints/final

Documentation:
  https://github.com/xynehq/sft-play

Get help:
  sft-play --help

EOF
