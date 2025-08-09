#!/bin/bash

# SFT-Play Batch Processing Script
# Process multiple datasets with different configurations

set -e

echo "📦 SFT-Play Batch Processing"
echo "============================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Function to process a single dataset
process_dataset() {
    local raw_file="$1"
    local config_file="$2"
    local style_prompt="$3"
    local output_suffix="$4"
    
    print_status "Processing dataset: $raw_file"
    
    # Create output directories
    mkdir -p "data/processed_${output_suffix}"
    mkdir -p "data/processed_with_style_${output_suffix}"
    mkdir -p "data/rendered_${output_suffix}"
    
    # Process raw data
    print_status "Step 1: Processing raw data..."
    python scripts/process_data.py \
        --config "$config_file" \
        --raw_path "$raw_file" \
        --output_dir "data/processed_${output_suffix}"
    
    # Apply style if provided
    if [ -n "$style_prompt" ]; then
        print_status "Step 2: Applying style prompts..."
        for split in train val test; do
            if [ -f "data/processed_${output_suffix}/${split}.jsonl" ]; then
                python scripts/style_prompt.py \
                    --config "$config_file" \
                    --style "$style_prompt" \
                    --in "data/processed_${output_suffix}/${split}.jsonl" \
                    --out "data/processed_with_style_${output_suffix}/${split}.jsonl" \
                    --mode prepend
            fi
        done
    else
        print_warning "No style prompt provided, skipping style application"
    fi
    
    # Render templates
    print_status "Step 3: Rendering templates..."
    for split in train val test; do
        input_dir="data/processed_with_style_${output_suffix}"
        if [ ! -f "${input_dir}/${split}.jsonl" ]; then
            input_dir="data/processed_${output_suffix}"
        fi
        
        if [ -f "${input_dir}/${split}.jsonl" ]; then
            python scripts/render_template.py \
                --config "$config_file" \
                --in "${input_dir}/${split}.jsonl" \
                --out "data/rendered_${output_suffix}/${split}.jsonl"
        fi
    done
    
    print_success "Dataset $raw_file processed successfully!"
}

# Function to run training experiments
run_experiments() {
    local base_config="$1"
    local experiment_name="$2"
    local data_suffix="$3"
    
    print_status "Running experiment: $experiment_name"
    
    # Create experiment-specific config
    local exp_config="configs/config_${experiment_name}.yaml"
    cp "$base_config" "$exp_config"
    
    # Update data paths in config
    sed -i "s|data/processed/|data/processed_${data_suffix}/|g" "$exp_config"
    sed -i "s|data/rendered/|data/rendered_${data_suffix}/|g" "$exp_config"
    
    # Create experiment output directory
    mkdir -p "outputs/${experiment_name}"
    mkdir -p "adapters/${experiment_name}"
    
    # Run training
    print_status "Starting training for $experiment_name..."
    python scripts/train.py --config "$exp_config" --output_dir "outputs/${experiment_name}"
    
    # Run evaluation
    print_status "Running evaluation for $experiment_name..."
    python scripts/eval.py --config "$exp_config" --output_dir "outputs/${experiment_name}"
    
    print_success "Experiment $experiment_name completed!"
}

# Main execution
main() {
    # Check if batch config file exists
    local batch_config="workflows/batch_config.yaml"
    
    if [ ! -f "$batch_config" ]; then
        print_warning "Batch config not found. Creating example batch_config.yaml..."
        
        cat > "$batch_config" << 'EOF'
# Batch Processing Configuration
datasets:
  - name: "dataset1"
    raw_file: "data/raw/dataset1.jsonl"
    config: "configs/config_run.yaml"
    style: "Answer concisely in 2 lines. No markdown."
    
  - name: "dataset2"
    raw_file: "data/raw/dataset2.jsonl"
    config: "configs/config_run.yaml"
    style: "Provide detailed explanations with examples."

experiments:
  - name: "qlora_experiment"
    base_config: "configs/config_run.yaml"
    data_suffix: "dataset1"
    
  - name: "lora_experiment"
    base_config: "configs/config_lora.yaml"
    data_suffix: "dataset2"
EOF
        
        print_status "Example batch_config.yaml created. Please edit it and run again."
        return 0
    fi
    
    # Parse and process datasets (simplified - in real implementation would use yq or python)
    print_status "Processing datasets from batch configuration..."
    
    # Example processing (you would implement proper YAML parsing)
    print_warning "This is a template script. Implement YAML parsing for production use."
    print_status "For now, you can manually run:"
    echo ""
    echo "# Process individual datasets:"
    echo "python scripts/process_data.py --config configs/config_run.yaml --raw_path data/raw/your_data.jsonl"
    echo ""
    echo "# Apply style to all splits:"
    echo "for split in train val test; do"
    echo "  python scripts/style_prompt.py --config configs/config_run.yaml \\"
    echo "    --style 'Your style prompt' \\"
    echo "    --in data/processed/\${split}.jsonl \\"
    echo "    --out data/processed_with_style/\${split}.jsonl"
    echo "done"
    echo ""
    echo "# Render templates:"
    echo "for split in train val test; do"
    echo "  python scripts/render_template.py --config configs/config_run.yaml \\"
    echo "    --in data/processed_with_style/\${split}.jsonl \\"
    echo "    --out data/rendered/\${split}.jsonl"
    echo "done"
}

# Run main function
main "$@"
echo ""
print_success "Batch processing script completed! 🎉"