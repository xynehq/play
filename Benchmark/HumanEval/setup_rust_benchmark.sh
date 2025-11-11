#!/bin/bash

# =============================================================================
# Rust Benchmarking Setup Script
# =============================================================================
# This script installs all dependencies required for running Rust benchmarking
# including Rust toolchain, Cargo, Python packages, and other utilities.
# =============================================================================

set -e  # Exit on error

echo "======================================================================"
echo "🚀 Rust Benchmarking Environment Setup"
echo "======================================================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo ""
echo "ℹ️  Step 1: Checking system requirements..."
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "ℹ️  Detected OS: $MACHINE"

# =============================================================================
# Install Rust and Cargo
# =============================================================================

echo ""
echo "ℹ️  Step 2: Installing Rust toolchain and Cargo..."
echo ""

if command_exists rustc && command_exists cargo; then
    RUST_VERSION=$(rustc --version)
    CARGO_VERSION=$(cargo --version)
    echo "✅ Rust is already installed: $RUST_VERSION"
    echo "✅ Cargo is already installed: $CARGO_VERSION"
    
    read -p "Do you want to update Rust? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "ℹ️  Updating Rust..."
        rustup update
        echo "✅ Rust updated successfully"
    fi
else
    echo "ℹ️  Installing Rust and Cargo via rustup..."
    
    # Download and install rustup
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Source the cargo environment
    source "$HOME/.cargo/env"
    
    echo "✅ Rust and Cargo installed successfully"
    
    # Verify installation
    rustc --version
    cargo --version
fi

# Ensure cargo is in PATH for current session
export PATH="$HOME/.cargo/bin:$PATH"

# =============================================================================
# Install Python dependencies
# =============================================================================

echo ""
echo "ℹ️  Step 3: Installing Python dependencies..."
echo ""

# Check if Python 3 is installed
if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ Python is installed: $PYTHON_VERSION"

# Install pip if not available
if ! command_exists pip3; then
    echo "ℹ️  Installing pip..."
    
    if [ "$MACHINE" = "Mac" ]; then
        python3 -m ensurepip --upgrade
    elif [ "$MACHINE" = "Linux" ]; then
        sudo apt-get update
        sudo apt-get install -y python3-pip
    fi
    
    echo "✅ pip installed successfully"
fi

# Upgrade pip
echo "ℹ️  Upgrading pip..."
python3 -m pip install --upgrade pip

# Install the human_eval package
echo "ℹ️  Installing human_eval package..."
python3 -m pip install -e .

# Install additional Python dependencies
echo "ℹ️  Installing additional Python dependencies..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
    echo "✅ Requirements installed from requirements.txt"
else
    # Install essential packages manually
    python3 -m pip install numpy openai tqdm anthropic
    echo "✅ Essential Python packages installed"
fi

# =============================================================================
# Install system utilities
# =============================================================================

echo ""
echo "ℹ️  Step 4: Installing system utilities..."
echo ""

if [ "$MACHINE" = "Mac" ]; then
    # macOS using Homebrew
    if ! command_exists brew; then
        echo "⚠️  Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        echo "✅ Homebrew installed successfully"
    else
        echo "✅ Homebrew is already installed"
    fi
    
    # Install useful utilities
    echo "ℹ️  Installing additional utilities via Homebrew..."
    brew install coreutils jq
    
elif [ "$MACHINE" = "Linux" ]; then
    # Linux using apt
    echo "ℹ️  Installing additional utilities via apt..."
    sudo apt-get update
    sudo apt-get install -y build-essential curl git jq
fi

echo "✅ System utilities installed"

# =============================================================================
# Verify Rust setup
# =============================================================================

echo ""
echo "ℹ️  Step 5: Verifying Rust installation..."
echo ""

# Test cargo by creating a simple test project
TEST_DIR=$(mktemp -d)
cd "$TEST_DIR"

echo "ℹ️  Creating test Rust project..."
cargo new test_project --bin >/dev/null 2>&1
cd test_project

echo "ℹ️  Building test project..."
if cargo build >/dev/null 2>&1; then
    echo "✅ Rust toolchain is working correctly"
else
    echo "❌ Rust build test failed"
    exit 1
fi

# Clean up test project
cd - >/dev/null
rm -rf "$TEST_DIR"

# =============================================================================
# Download Rust dataset (if not exists)
# =============================================================================

echo ""
echo "ℹ️  Step 6: Checking for Rust dataset..."
echo ""

if [ -f "data/humaneval-rust.jsonl.gz" ]; then
    echo "✅ Rust dataset already exists at data/humaneval-rust.jsonl.gz"
else
    echo "⚠️  Rust dataset not found at data/humaneval-rust.jsonl.gz"
    echo "ℹ️  Please ensure you have the Rust dataset file in the data/ directory"
    echo "ℹ️  You can download it from the MultiPL-E repository or use your own dataset"
fi

# =============================================================================
# Set up environment variables
# =============================================================================

echo ""
echo "ℹ️  Step 7: Setting up environment variables..."
echo ""

# Create or update shell profile
SHELL_PROFILE=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
fi

if [ -n "$SHELL_PROFILE" ]; then
    # Add cargo to PATH if not already there
    if ! grep -q 'cargo/env' "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Rust environment" >> "$SHELL_PROFILE"
        echo 'source "$HOME/.cargo/env"' >> "$SHELL_PROFILE"
        echo "✅ Added Rust environment to $SHELL_PROFILE"
    else
        echo "✅ Rust environment already configured in $SHELL_PROFILE"
    fi
fi

# =============================================================================
# Create .env.example file for API keys
# =============================================================================

echo ""
echo "ℹ️  Step 8: Creating .env.example file..."
echo ""

cat > .env.example << 'EOF'
# API Configuration
# Copy this file to .env and fill in your API keys

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI API Key (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_NAME=claude-3-7-sonnet-20250219
BASE_MODEL_NAME=claude-3-7-sonnet-20250219

# Benchmark Settings
NUM_SAMPLES=10
TEMPERATURE=0.8
MAX_TOKENS=2048
TIMEOUT=60
EOF

echo "✅ Created .env.example file"

if [ ! -f ".env" ]; then
    echo "⚠️  Please create a .env file with your API keys:"
    echo "ℹ️    cp .env.example .env"
    echo "ℹ️    # Then edit .env with your actual API keys"
fi

# =============================================================================
# Create quick reference guide
# =============================================================================

echo ""
echo "ℹ️  Creating quick reference guide..."
echo ""

cat > SETUP_GUIDE.md << 'EOF'
# Rust Benchmarking Setup Guide

## Prerequisites Installed ✅

This setup script has installed:
- Rust toolchain (rustc, cargo)
- Python 3 and pip
- human_eval package
- Required Python dependencies
- System utilities (jq, etc.)

## Next Steps

### 1. Configure API Keys

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required API keys:
- `ANTHROPIC_API_KEY`: For Claude models
- `OPENAI_API_KEY`: For OpenAI models (optional)

### 2. Verify Setup

Test Rust compilation:
```bash
cargo --version
rustc --version
```

Test Python environment:
```bash
python3 -c "from human_eval.evaluation import evaluate_rust_correctness; print('✅ Setup successful')"
```

### 3. Run Benchmarking

Run the Rust benchmark:
```bash
python3 run_humaneval_api.py
```

This will:
- Load Rust problems from `data/humaneval-rust.jsonl.gz`
- Generate completions using your configured model
- Evaluate correctness using Rust compilation
- Calculate pass@1 and pass@10 metrics
- Save results to `result/` directory

## File Structure

```
.
├── run_humaneval_api.py       # Main benchmarking script
├── human_eval/                 # Core evaluation package
│   ├── evaluation.py          # Rust evaluation functions
│   └── execution.py           # Execution utilities
├── data/
│   └── humaneval-rust.jsonl.gz # Rust problems dataset
├── result/                     # Benchmark results (created automatically)
├── .env                        # API keys (you create this)
└── .env.example               # Template for .env
```

## Troubleshooting

### Rust compilation errors
- Ensure `cargo` is in your PATH: `source ~/.cargo/env`
- Update Rust: `rustup update`

### Python import errors
- Reinstall package: `pip3 install -e .`
- Check Python version: `python3 --version` (should be 3.7+)

### API errors
- Verify API keys in `.env` file
- Check API rate limits
- Ensure sufficient API credits

## Running Custom Benchmarks

Edit `run_humaneval_api.py` to customize:
- `MODEL_NAME`: Your fine-tuned model
- `BASE_MODEL_NAME`: Base model for comparison
- `NUM_SAMPLES`: Number of samples per problem (default: 10)
- `TEMPERATURE`: Sampling temperature (default: 0.8)

## Results

Results are saved in `result/` directory:
- `api_finetuned_rust.jsonl`: Fine-tuned model completions
- `api_base_rust.jsonl`: Base model completions
- `api_*_results.jsonl`: Evaluation results
- `api_rust_benchmark_results.json`: Summary metrics

## Support

For issues or questions:
- Check the logs in `result/*_progress.log`
- Verify Rust dataset exists: `ls -lh data/humaneval-rust.jsonl.gz`
- Test imports: `python3 -c "from human_eval.evaluation import evaluate_rust_correctness; print('OK')"`
EOF

echo "✅ Created SETUP_GUIDE.md"

# =============================================================================
# Final summary
# =============================================================================

echo ""
echo "======================================================================"
echo "✅ 🎉 Setup Complete!"
echo "======================================================================"
echo ""
echo "ℹ️  Summary of installed components:"
echo "  ✅ Rust toolchain (rustc, cargo)"
echo "  ✅ Python 3 and pip"
echo "  ✅ human_eval package"
echo "  ✅ Required Python dependencies"
echo "  ✅ System utilities"
echo ""
echo "ℹ️  Next steps:"
echo "  1. Create .env file with your API keys:"
echo "     cp .env.example .env"
echo "     # Then edit .env with your actual API keys"
echo ""
echo "  2. Verify setup:"
echo "     cargo --version"
echo "     python3 -c 'from human_eval.evaluation import evaluate_rust_correctness; print(\"✅ OK\")'"
echo ""
echo "  3. Run benchmarking:"
echo "     python3 run_humaneval_api.py"
echo ""
echo "ℹ️  For detailed instructions, see SETUP_GUIDE.md"
echo ""
echo "✅ Happy benchmarking! 🚀"
echo "======================================================================"
