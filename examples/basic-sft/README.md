# Basic SFT Example

Your first supervised fine-tuning project with Xyne-LLM-Play. This example teaches a small language model to be a helpful assistant.

## What You'll Learn

- How to prepare training data in the correct format
- How to configure Xyne-LLM-Play for basic fine-tuning
- How to train and evaluate your model
- How to chat with your fine-tuned model

## Hardware Requirements

| GPU | VRAM | Model | Training Time |
|-----|------|-------|---------------|
| RTX 4060 | 8GB | Qwen2.5-1.5B | ~15 minutes |
| RTX 4070 | 12GB | Qwen2.5-3B | ~20 minutes |
| RTX 4090 | 24GB | Qwen2.5-7B | ~30 minutes |

## Step 1: Setup

```bash
# Navigate to the example directory
cd examples/basic-sft

# Copy sample data to the main data directory
cp data/sample_data.jsonl ../../data/raw/

# Go back to project root
cd ../..

# Process the data
make process

# Verify setup
make check
```

## Step 2: Configuration

The `config.yaml` file is optimized for this example:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"  # Small model for quick training
  max_seq_len: 512

tuning:
  mode: "qlora"                       # Memory-efficient training
  backend: "bnb"
  lora:
    r: 16
    alpha: 32
    dropout: 0.1

train:
  epochs: 3                           # Quick training
  learning_rate: 2e-4
  batch_size: 4
  grad_accum: 4
  eval_strategy: "steps"
  eval_steps: 50
  bf16: true
  gradient_checkpointing: true
```

## Step 3: Training

```bash
# Start training with monitoring
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml

# Monitor progress at: http://localhost:6006
# You should see:
# - Training loss decreasing
# - Evaluation loss stable
# - Memory usage within limits
```

Expected training output:
```
{'train_loss': 1.2345, 'learning_rate': 2e-4, 'epoch': 0.5}
{'eval_loss': 1.1567, 'eval_runtime': 2.3, 'epoch': 1.0}
{'train_loss': 0.9876, 'learning_rate': 1.5e-4, 'epoch': 1.5}
{'eval_loss': 1.0234, 'eval_runtime': 2.1, 'epoch': 2.0}
{'train_loss': 0.8765, 'learning_rate': 1e-4, 'epoch': 2.5}
{'eval_loss': 0.9876, 'eval_runtime': 2.0, 'epoch': 3.0}
```

## Step 4: Evaluation

```bash
# Evaluate the model
make eval CONFIG=examples/basic-sft/config.yaml

# Quick evaluation (200 samples)
make eval-quick CONFIG=examples/basic-sft/config.yaml
```

Expected evaluation results:
```
{'eval_loss': 0.9876, 'eval_runtime': 2.0, 'eval_samples_per_second': 45.2}
```

## Step 5: Interactive Testing

```bash
# Chat with your model
make infer CONFIG=examples/basic-sft/config.yaml

# Try these questions:
# - "What is machine learning?"
# - "Explain neural networks simply."
# - "How do I learn Python?"
```

Expected conversation:
```
You: What is machine learning?
Model: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

You: Explain neural networks simply.
Model: Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes that process information and learn patterns from data.
```

## Step 6: Customization

### Try Different Models
```bash
# Use a larger model (if you have enough VRAM)
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

# Use a different model family
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  MODEL_NAME="microsoft/DialoGPT-medium"
```

### Adjust Training Parameters
```bash
# Train longer for better quality
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  TRAIN_EPOCHS=5

# Use larger batch size (if memory allows)
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  TRAIN_BATCH_SIZE=8 TRAIN_GRAD_ACCUM=2
```

### Add Your Own Data
```bash
# Create your own data file
echo '{"system": "You are a helpful assistant.", "user": "What is your favorite color?", "assistant": "I don't have personal preferences, but blue is often considered calming and trustworthy."}' > ../../data/raw/my_data.jsonl

# Re-process with your data
make process

# Train with your data
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml
```

## Troubleshooting

### Common Issues

#### "CUDA out of memory"
```bash
# Reduce batch size
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  TRAIN_BATCH_SIZE=2 TRAIN_GRAD_ACCUM=8

# Reduce sequence length
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  MODEL_MAX_SEQ_LEN=256
```

#### "Training is very slow"
```bash
# Use Unsloth backend (faster)
make train-unsloth-tb CONFIG=examples/basic-sft/config.yaml

# Reduce evaluation frequency
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  TRAIN_EVAL_STEPS=100
```

#### "Model gives poor responses"
```bash
# Train longer
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  TRAIN_EPOCHS=5

# Use lower learning rate
make train-bnb-tb CONFIG=examples/basic-sft/config.yaml \
  TRAIN_LEARNING_RATE=1e-4
```

### Monitoring Commands
```bash
# Check GPU usage
make memory-check

# Check training progress
tail -f outputs/*/training.log

# Validate configuration
python scripts/validate_config.py --config examples/basic-sft/config.yaml
```

## Expected Results

After successful training, you should have:

1. **Trained Model**: `adapters/last/` directory with LoRA adapters
2. **Training Logs**: `outputs/*/training.log` with detailed metrics
3. **TensorBoard Logs**: `outputs/tb/` with visualization
4. **Evaluation Results**: Loss around 0.9-1.1 for this small dataset

### Performance Benchmarks

| Model | Dataset Size | Training Time | Final Loss | Quality |
|-------|--------------|---------------|------------|---------|
| Qwen2.5-1.5B | 100 samples | 15 min | ~0.99 | Good |
| Qwen2.5-3B | 100 samples | 20 min | ~0.95 | Better |
| Qwen2.5-7B | 100 samples | 30 min | ~0.90 | Best |

## Next Steps

### Continue Learning
- **[Customer Support Bot](../customer-support/)** - Domain-specific training
- **[Multi-GPU Training](../multi-gpu/)** - Scale to larger models
- **[Configuration Guide](../../docs/configuration/)** - Advanced options

### Experiment Ideas
1. **Different Personas**: Train with different system prompts
2. **Larger Datasets**: Add more training examples
3. **Multi-turn Conversations**: Try conversation format
4. **Different Models**: Compare model families

### Production Tips
1. **More Data**: Use 1000+ examples for production models
2. **Longer Training**: 5-10 epochs for better quality
3. **Evaluation**: Use separate test set for quality assessment
4. **Merging**: Use `make merge` for deployment-ready models

## Success Criteria

You've successfully completed this example if:
- ✅ Training completed without errors
- ✅ Evaluation loss is < 1.2
- ✅ Model responds coherently to test questions
- ✅ TensorBoard shows decreasing loss
- ✅ You can chat with the model interactively

**Congratulations! You've fine-tuned your first language model! 🎉**

## Help

If you encounter issues:
```bash
# Check your setup
make check

# Get help with commands
make help

# Validate everything is working
./workflows/quick_start.sh
