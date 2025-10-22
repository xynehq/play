# LLM Fine-Tuning with Xyne-Play

## 🎯 What is LLM Fine-Tuning?

LLM Fine-Tuning adapts pre-trained language models to specific tasks or domains by training them on custom datasets. This improves model performance for your specific use case while retaining the model's general capabilities.

**Note**: This guide covers general fine-tuning concepts. For specific instruction following, see [Supervised Fine-Tuning (SFT)](supervised-fine-tuning.md). For domain-specific knowledge training, see [Domain-Adaptive Pretraining (DAPT/CPT)](dapt-cpt.md).

## 📊 Expected Dataset Format

### Format 1: Chat Format (Recommended)
```json
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence..."}
{"system": "You are a helpful assistant.", "user": "Explain neural networks.", "assistant": "Neural networks are computing systems inspired by biological neural networks..."}
```

### Format 2: Messages Format (OpenAI Compatible)
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is..."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain neural networks."}, {"role": "assistant", "content": "Neural networks are..."}]}
```

### Format 3: Simple Q&A
```json
{"input": "What is machine learning?", "target": "Machine learning is a subset of artificial intelligence..."}
{"input": "Explain neural networks.", "target": "Neural networks are computing systems inspired by biological neural networks..."}
```

## 🚀 Training Command on Toy Dataset

### Step 1: Create Sample Data
```bash
# Create toy dataset
mkdir -p data/raw
cat > data/raw/toy_dataset.jsonl << EOF
{"system": "You are a helpful assistant.", "user": "What is Python?", "assistant": "Python is a high-level programming language known for its simplicity and readability."}
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn from data."}
{"system": "You are a helpful assistant.", "user": "What is deep learning?", "assistant": "Deep learning is a subset of machine learning using neural networks with multiple layers."}
{"system": "You are a helpful assistant.", "user": "What is NLP?", "assistant": "Natural Language Processing (NLP) is a field of AI focused on interactions between computers and human language."}
{"system": "You are a helpful assistant.", "user": "What is computer vision?", "assistant": "Computer vision is an AI field that enables computers to interpret and understand visual information from the world."}
EOF
```

### Step 2: Process Data
```bash
# Process raw data to training format
make process

# Add style instructions (optional)
make style STYLE="Answer concisely and clearly"

# Validate setup
make check
```

### Step 3: Start Training
```bash
# Train with monitoring (recommended)
make train-bnb-tb

# Or basic training
make train
```

### Step 4: Test Your Model
```bash
# Interactive chat
make infer

# Batch inference
make infer-batch
```

## ⚙️ Custom Configuration

### Using Custom Config File
```bash
# Create custom config
cp configs/config_base.yaml configs/my_config.yaml

# Edit the config file
nano configs/my_config.yaml

# Train with custom config
make train CONFIG=configs/my_config.yaml
```

### Partial Configuration Override
```bash
# Override specific parameters without editing files
make train CONFIG=configs/run_bnb.yaml model.name=Qwen/Qwen2.5-7B-Instruct train.epochs=5

# Override training parameters
make train CONFIG=configs/run_bnb.yaml train.learning_rate=1e-4 train.batch_size=4

# Override model parameters
make train CONFIG=configs/run_bnb.yaml model.max_seq_len=1024 tuning.lora.r=64
```

### Common Configuration Overrides

#### Model Selection
```bash
# Small models (8GB VRAM)
make train CONFIG=configs/run_bnb.yaml model.name=Qwen/Qwen2.5-1.5B-Instruct

# Medium models (16GB+ VRAM)
make train CONFIG=configs/run_bnb.yaml model.name=Qwen/Qwen2.5-7B-Instruct

# Large models (24GB+ VRAM)
make train CONFIG=configs/run_bnb.yaml model.name=Qwen/Qwen2.5-14B-Instruct
```

#### Training Parameters
```bash
# Faster training (less epochs)
make train CONFIG=configs/run_bnb.yaml train.epochs=1 train.eval_steps=50

# Higher quality training (more epochs)
make train CONFIG=configs/run_bnb.yaml train.epochs=5 train.learning_rate=1e-4

# Memory-efficient training
make train CONFIG=configs/run_bnb.yaml model.max_seq_len=256 train.batch_size=1
```

#### LoRA Parameters
```bash
# Stronger adaptation
make train CONFIG=configs/run_bnb.yaml tuning.lora.r=64 tuning.lora.alpha=128

# Lighter adaptation
make train CONFIG=configs/run_bnb.yaml tuning.lora.r=16 tuning.lora.alpha=32

# Custom target modules
make train CONFIG=configs/run_bnb.yaml tuning.lora.target_modules='["q_proj","v_proj","o_proj"]'
```

## 📈 Monitoring and Evaluation

### TensorBoard Monitoring
```bash
# Start training with TensorBoard
make train-bnb-tb

# View logs at http://localhost:6006
make tensorboard
```

### Evaluation
```bash
# Quick evaluation (200 samples)
make eval-quick

# Full evaluation
make eval

# Test set evaluation
make eval-test
```

## 🔧 Advanced Options

### Multi-GPU Training
```bash
# Setup multi-GPU (one-time)
make setup-accelerate

# Distributed training
make train-distributed-tb CONFIG=configs/run_bnb.yaml
```

### Different Backends
```bash
# BitsAndBytes (stable, recommended)
make train-bnb-tb

# Unsloth (faster, experimental)
make train-unsloth-tb
```

### Model Merging
```bash
# Merge LoRA adapters to full model
make merge

# Test merged model
make merge-test
```

## 📝 Best Practices

### Data Quality
- Use consistent formatting across all examples
- Ensure high-quality, accurate responses
- Include diverse examples covering your use case
- Start with 100-1000 examples for initial testing

### Training Parameters
- Start with default parameters, then tune based on results
- Use smaller sequence lengths (256-512) for memory efficiency
- Monitor training loss to ensure convergence
- Use validation set to prevent overfitting

### Hardware Optimization
- Use QLoRA for memory-constrained training
- Enable gradient checkpointing for large models
- Monitor GPU memory usage during training
- Use appropriate batch sizes for your hardware

## 🚨 Troubleshooting

### Common Issues
```bash
# CUDA out of memory
make train CONFIG=configs/run_bnb.yaml model.max_seq_len=256 train.batch_size=1

# Slow training
make train CONFIG=configs/run_unsloth.yaml

# Poor results
make train CONFIG=configs/run_bnb.yaml train.epochs=5 train.learning_rate=1e-4
```

### Validation
```bash
# Check setup
make check

# GPU information
make gpu-info

# Memory usage
make memory-check
```

## 📚 Related Documentation

- [Configuration Reference](../configs/) - All configuration options
- [Multi-GPU Training](../advanced/multi-gpu.md) - Distributed training guide
- [Advanced Configuration](../advanced/custom-configs.md) - Custom configuration patterns
