# Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning teaches language models to follow instructions and engage in helpful conversations through example-based training.

## What SFT Does

SFT transforms base language models into helpful assistants by training them on high-quality instruction-response pairs. This process teaches models to:

- Follow user instructions accurately
- Engage in natural conversations
- Provide helpful and relevant responses
- Adopt specific personas or communication styles

## Expected Dataset Format

### JSONL Format
Create a `.jsonl` file where each line contains a training example:

```json
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
{"system": "You are a helpful assistant.", "user": "Explain neural networks simply.", "assistant": "Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes that process information and learn patterns from data."}
```

### Field Descriptions
- **`system`** (optional): Sets the model's persona and behavior
- **`user`**: The user's instruction or question
- **`assistant`**: The desired response the model should learn

### Data Splits
Organize your data into three files for optimal training:
```
data/raw/
├── train.jsonl    # Training examples (majority of data)
├── val.jsonl      # Validation examples (10-20% of data)
└── test.jsonl     # Test examples (5-10% of data)
```

## Training Command on Toy Dataset

### Quick Start with Sample Data
```bash
# Generate sample dataset and start training
./workflows/quick_start.sh

# Manual training with sample data
make process
make train-bnb-tb
```

### Step-by-Step Training
```bash
# 1. Process raw data into training format
make process

# 2. (Optional) Add style instructions
make style STYLE="Answer in a friendly, helpful tone"

# 3. Start training with monitoring
make train-bnb-tb

# 4. Monitor progress at http://localhost:6006
```

### Training Commands
```bash
# Single GPU training
make train-bnb-tb              # BitsAndBytes backend (stable)
make train-unsloth-tb          # Unsloth backend (faster)

# Multi-GPU training
make train-distributed-tb      # Auto-detect GPUs
make train-deepspeed-tb        # Maximum memory efficiency
```

## Custom Configuration and Overrides

### Using Custom Config Files
Create your own configuration file:

```yaml
# configs/my_sft_config.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  max_seq_len: 512

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 16
    alpha: 32
    dropout: 0.1

train:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 4
  eval_strategy: "steps"
  eval_steps: 100
```

Use your custom config:
```bash
make train CONFIG=configs/my_sft_config.yaml
```

### Partial Parameter Overrides
Override specific parameters without creating a full config file:

```bash
# Override learning rate and epochs
make train CONFIG=configs/run_bnb.yaml TRAIN_LEARNING_RATE=1e-4 TRAIN_EPOCHS=5

# Override model and batch size
make train CONFIG=configs/run_bnb.yaml MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" TRAIN_BATCH_SIZE=2
```

### Common Override Patterns
```bash
# For smaller GPUs (reduce memory)
make train CONFIG=configs/run_bnb.yaml MODEL_MAX_SEQ_LEN=256 TRAIN_BATCH_SIZE=2

# For faster training (reduce quality)
make train CONFIG=configs/run_bnb.yaml TRAIN_EPOCHS=1 TRAIN_EVAL_STEPS=200

# For better quality (increase training)
make train CONFIG=configs/run_bnb.yaml TRAIN_EPOCHS=5 TRAIN_LEARNING_RATE=1e-4
```

## Advanced SFT Techniques

### Style and Persona Training
```bash
# Train with specific persona
make style STYLE="You are an expert in machine learning. Be concise and technical."
make train-bnb-tb

# Train with communication style
make style STYLE="Answer in a friendly, conversational tone. Use simple language."
make train-bnb-tb
```

### Multi-turn Conversations
For multi-turn conversations, structure your data like this:
```json
{"system": "You are a helpful assistant.", "conversations": [
  {"role": "user", "content": "What is Python?"},
  {"role": "assistant", "content": "Python is a high-level programming language..."},
  {"role": "user", "content": "What are its main features?"},
  {"role": "assistant", "content": "Python's main features include..."}
]}
```

### Quality Tips
- **High-quality examples**: Ensure responses are accurate and helpful
- **Consistent formatting**: Maintain consistent style across examples
- **Diverse instructions**: Include various types of tasks and questions
- **Appropriate length**: Keep responses concise but comprehensive

## Evaluation and Testing

### Evaluate Your Model
```bash
# Evaluate on validation set
make eval

# Quick evaluation (200 samples)
make eval-quick

# Interactive testing
make infer
```

### Test with Custom Inputs
```bash
# Create test inputs
echo "What is machine learning?" > test_inputs.txt
echo "Explain neural networks." >> test_inputs.txt

# Batch inference
make infer-batch
```

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce `batch_size` or `max_seq_len`
- **Poor responses**: Increase training epochs or improve data quality
- **Slow training**: Use Unsloth backend or multiple GPUs
- **Overfitting**: Add more data or use regularization

### Monitoring
```bash
# Check GPU usage
make memory-check

# Monitor training progress
# Visit: http://localhost:6006
```

## Next Steps

- **[Domain Adaptation](../dapt/)** - Add specialized knowledge
- **[Multi-GPU Training](../multi-gpu/)** - Scale to larger models
- **[Configuration Guide](../configuration/)** - Advanced configuration options
- **[Examples](../../examples/)** - Complete end-to-end examples
