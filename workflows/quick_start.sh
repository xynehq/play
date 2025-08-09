#!/bin/bash

# SFT-Play Quick Start Script
# This script sets up the project and runs a complete pipeline

set -e  # Exit on any error

echo "🚀 SFT-Play Quick Start"
echo "======================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "scripts" ]; then
    print_error "Please run this script from the sft-play root directory"
    exit 1
fi

# Step 1: Setup directories
print_status "Setting up directories..."
make setup-dirs
print_success "Directories created"

# Step 2: Install dependencies
print_status "Installing dependencies..."
make install
print_success "Dependencies installed"

# Step 3: Check for raw data
if [ ! -f "data/raw/raw.json" ] && [ ! -f "data/raw/raw.jsonl" ] && [ ! -f "data/raw/raw.csv" ]; then
    print_warning "No raw data found in data/raw/"
    print_status "Creating sample data for demonstration..."
    
    cat > data/raw/sample.jsonl << 'EOF'
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."}
{"system": "You are a helpful assistant.", "user": "Explain neural networks.", "assistant": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information and learn patterns from data."}
{"system": "You are a helpful assistant.", "user": "What is deep learning?", "assistant": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."}
{"system": "You are a helpful assistant.", "user": "How does training work?", "assistant": "Training involves feeding data to a model, calculating errors in predictions, and adjusting the model's parameters to minimize these errors through optimization algorithms."}
{"system": "You are a helpful assistant.", "user": "What is fine-tuning?", "assistant": "Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task by training it on task-specific data with a lower learning rate."}
EOF
    
    print_success "Sample data created at data/raw/sample.jsonl"
fi

# Step 4: Process data
print_status "Processing raw data..."
if ! make process; then
    print_error "Data processing failed. Please check your raw data format."
    exit 1
fi
print_success "Data processed successfully"

# Step 5: Apply style (optional)
read -p "Do you want to apply style prompts? (y/N): " apply_style
if [[ $apply_style =~ ^[Yy]$ ]]; then
    print_status "Applying style prompts..."
    make style
    print_success "Style prompts applied"
fi

# Step 6: Render templates (optional)
read -p "Do you want to render chat templates? (y/N): " render_templates
if [[ $render_templates =~ ^[Yy]$ ]]; then
    print_status "Rendering chat templates..."
    make render
    print_success "Templates rendered"
fi

# Step 7: Show next steps
echo ""
print_success "Quick start completed! 🎉"
echo ""
echo "Next steps:"
echo "1. Review your processed data:"
echo "   - data/processed/train.jsonl"
echo "   - data/processed/val.jsonl (if available)"
echo "   - data/processed/test.jsonl (if available)"
echo ""
echo "2. Configure your training in configs/config_run.yaml"
echo ""
echo "3. Start training:"
echo "   make train"
echo ""
echo "4. Evaluate your model:"
echo "   make eval"
echo ""
echo "5. Run inference:"
echo "   make infer"
echo ""
echo "For more commands, run: make help"
echo "Done! Enjoy using SFT-Play! 🚀"