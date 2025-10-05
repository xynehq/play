# Setup and Installation

Complete setup guide for Xyne-LLM-Play, including installation, configuration, and troubleshooting.

## Quick Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd play

# Install dependencies
make install

# Create necessary directories
make setup-dirs

# Verify installation
make check
```

## Setup Guides

### Core Setup
- **[Installation Guide](SETUP_DOCUMENTATION.md)** - Detailed installation instructions
- **[Accelerate Setup](ACCELERATE_SETUP_GUIDE.md)** - Multi-GPU configuration
- **[Automation Guide](AUTOMATION_GUIDE.md)** - Advanced automation features

### Backend-Specific Setup
- **[Unsloth Setup](UNSLOTH_FIX_DOCUMENTATION.md)** - Unsloth backend configuration and fixes

## Hardware Setup

### Single GPU Setup
```bash
# Check your GPU
make gpu-info

# Start with sample data
./workflows/quick_start.sh

# Begin training
make train-bnb-tb
```

### Multi-GPU Setup
```bash
# Configure Accelerate for multi-GPU
make setup-accelerate

# Verify multi-GPU detection
make gpu-info

# Start distributed training
make train-distributed-tb
```

## Configuration

### Environment Variables
```bash
# Set HuggingFace token (for private models)
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Set CUDA device (if needed)
export CUDA_VISIBLE_DEVICES="0,1"

# Increase NCCL timeout for multi-GPU
export NCCL_TIMEOUT=1800
```

### Project Structure
```
xyne-llm-play/
├── configs/           # Training configurations
├── data/             # Training data
├── scripts/          # Core training scripts
├── adapters/         # Trained model adapters
├── outputs/          # Training outputs and logs
└── examples/         # Complete examples
```

## Verification

### System Check
```bash
# Complete system validation
make check

# GPU information
make gpu-info

# Memory usage
make memory-check
```

### Test Training
```bash
# Quick test with sample data
./workflows/quick_start.sh

# Verify training works
make train-bnb-tb CONFIG=configs/run_bnb.yaml
```

## Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Clean installation
make clean
pip install -r requirements.txt

# With constraints (production)
pip install -r requirements.txt -c constraints.txt
```

#### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Memory Issues
```bash
# Reduce batch size
make train CONFIG=configs/run_bnb.yaml TRAIN_BATCH_SIZE=1

# Enable gradient checkpointing
make train CONFIG=configs/run_bnb.yaml TRAIN_GRADIENT_CHECKPOINTING=true
```

#### Multi-GPU Issues
```bash
# Reconfigure Accelerate
make setup-accelerate

# Check GPU connectivity
nvidia-smi topo -m

# Increase timeout
export NCCL_TIMEOUT=1800
```

### Performance Optimization

#### Speed Up Training
```bash
# Use Unsloth backend
make train-unsloth-tb

# Optimize data loading
make train CONFIG=configs/run_bnb.yaml TRAIN_DATALOADER_NUM_WORKERS=4

# Reduce evaluation frequency
make train CONFIG=configs/run_bnb.yaml TRAIN_EVAL_STEPS=500
```

#### Memory Optimization
```bash
# Use QLoRA
make train CONFIG=configs/run_bnb.yaml

# Reduce sequence length
make train CONFIG=configs/run_bnb.yaml MODEL_MAX_SEQ_LEN=256

# Enable gradient checkpointing
make train CONFIG=configs/run_bnb.yaml TRAIN_GRADIENT_CHECKPOINTING=true
```

## Next Steps

### After Setup
1. **[Basic SFT Example](../../examples/basic-sft/)** - Your first training project
2. **[Supervised Fine-Tuning](../sft/)** - Learn SFT techniques
3. **[Configuration Guide](../configuration/)** - Advanced configuration
4. **[Examples](../../examples/)** - Complete end-to-end examples

### Advanced Setup
- **[Multi-GPU Training](../multi-gpu/)** - Distributed training setup
- **[Domain Adaptation](../dapt/)** - Specialized training setup
- **[Production Deployment](../deployment/)** - Production configuration

## Help and Support

### Built-in Help
```bash
# See all available commands
make help

# Check system status
make check

# Validate configuration
python scripts/validate_config.py --config configs/run_bnb.yaml
```

### Common Commands
```bash
# Installation
make install && make setup-dirs

# Quick start
./workflows/quick_start.sh

# Training
make train-bnb-tb

# Multi-GPU
make setup-accelerate && make train-distributed-tb

# Monitoring
make gpu-info && make memory-check
```

### Getting Help
- Check the [Troubleshooting Guide](SETUP_DOCUMENTATION.md#troubleshooting)
- Review [Configuration Options](../configuration/)
- Try the [Basic Example](../../examples/basic-sft/)
- Use `make help` for command assistance

---

**Ready to start training? Begin with the [Basic SFT Example](../../examples/basic-sft/)! 🚀**
