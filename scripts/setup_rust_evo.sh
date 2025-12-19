#!/bin/bash

# RustEvo VM Integration Setup Script
# This script clones the original RustEvo repository and sets it up for VM environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
RUSTEVO_REPO="https://github.com/SYSUSELab/RustEvo.git"
RUSTEVO_DIR="/tmp/RustEvo"
TARGET_DIR="Benchmark/RustEvo"

echo -e "${GREEN}🦀 Setting up RustEvo for VM environment${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Please run this script from the root of the VM-benchmarks repository${NC}"
    exit 1
fi

# Clean up any existing RustEvo installation
echo -e "${YELLOW}🧹 Cleaning up existing installation...${NC}"
if [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"
fi

# Clone the original RustEvo repository
echo -e "${YELLOW}📥 Cloning RustEvo repository...${NC}"
if [ -d "$RUSTEVO_DIR" ]; then
    rm -rf "$RUSTEVO_DIR"
fi

git clone "$RUSTEVO_REPO" "$RUSTEVO_DIR"

# Create target directory structure
echo -e "${YELLOW}📁 Creating directory structure...${NC}"
mkdir -p "$TARGET_DIR"
mkdir -p "$TARGET_DIR/Dataset"
mkdir -p "$TARGET_DIR/Evaluate"
mkdir -p "$TARGET_DIR/Scripts"
mkdir -p "$TARGET_DIR/Imgs"

# Copy essential files
echo -e "${YELLOW}📋 Copying essential files...${NC}"

# Copy README and requirements
if [ -f "$RUSTEVO_DIR/README.md" ]; then
    cp "$RUSTEVO_DIR/README.md" "$TARGET_DIR/"
fi

if [ -f "$RUSTEVO_DIR/requirements.txt" ]; then
    cp "$RUSTEVO_DIR/requirements.txt" "$TARGET_DIR/"
fi

# Copy Dataset files
if [ -d "$RUSTEVO_DIR/Dataset" ]; then
    cp -r "$RUSTEVO_DIR/Dataset/"* "$TARGET_DIR/Dataset/" 2>/dev/null || true
fi

# Copy Evaluation scripts
if [ -d "$RUSTEVO_DIR/Evaluate" ]; then
    cp -r "$RUSTEVO_DIR/Evaluate/"* "$TARGET_DIR/Evaluate/" 2>/dev/null || true
fi

# Copy Scripts
if [ -d "$RUSTEVO_DIR/Scripts" ]; then
    cp -r "$RUSTEVO_DIR/Scripts/"* "$TARGET_DIR/Scripts/" 2>/dev/null || true
fi

# Copy Images
if [ -d "$RUSTEVO_DIR/Imgs" ]; then
    cp -r "$RUSTEVO_DIR/Imgs/"* "$TARGET_DIR/Imgs/" 2>/dev/null || true
fi

# Apply VM-specific modifications
echo -e "${YELLOW}⚙️  Applying VM-specific modifications...${NC}"

# Create VM-specific configuration
cat > "$TARGET_DIR/vm_config.yaml" << EOF
# RustEvo VM Configuration
# VM-specific settings for RustEvo benchmark

# Paths
dataset_path: "Dataset"
results_path: "Results"
scripts_path: "Scripts"

# VM Resource Limits
max_memory_gb: 8
max_cpu_cores: 4
timeout_seconds: 300

# Model Configuration
default_model: "gpt-3.5-turbo"
temperature: 0.7
max_tokens: 2048

# Output Configuration
save_intermediate: true
verbose: true
log_level: "INFO"
EOF

# Clean up temporary files
echo -e "${YELLOW}🧹 Cleaning up temporary files...${NC}"
rm -rf "$RUSTEVO_DIR"

# Remove virtual environment and temporary directories
if [ -d "$TARGET_DIR/RustEvo_env" ]; then
    rm -rf "$TARGET_DIR/RustEvo_env"
fi

if [ -d "$TARGET_DIR/Results" ]; then
    rm -rf "$TARGET_DIR/Results"
fi

# Create .gitignore for RustEvo
cat > "$TARGET_DIR/.gitignore" << EOF
# Virtual environments
RustEvo_env/
venv/
.venv/
env/

# Results and temporary files
Results/
*.log
*.tmp
__pycache__/
*.pyc
*.pyo

# OS files
.DS_Store
Thumbs.db
EOF

# Make scripts executable
find "$TARGET_DIR/Scripts" -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
find "$TARGET_DIR/Evaluate" -name "*.py" -exec chmod +x {} \; 2>/dev/null || true

echo -e "${GREEN}✅ RustEvo setup completed successfully!${NC}"
echo -e "${GREEN}📁 RustEvo is now available at: $TARGET_DIR${NC}"
echo -e "${YELLOW}💡 To run RustEvo benchmarks, use: python scripts/run_rust_evo.py${NC}"
