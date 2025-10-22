# Supervised Fine-Tuning (SFT) with Xyne-Play

## 🎯 What is Supervised Fine-Tuning?

Supervised Fine-Tuning (SFT) trains language models to follow instructions and respond appropriately to user inputs. It's the most common approach for creating chat assistants and task-specific models.

### When to Use SFT
- **Chat Assistants**: Create conversational AI for customer service, personal assistants
- **Task-Specific Models**: Train models for specific tasks like summarization, translation
- **Instruction Following**: Teach models to follow specific formats or styles
- **Domain Chat**: Create chatbots with domain expertise combined with conversational ability

## 📊 Expected Dataset Format

### Format 1: Chat Format (Recommended)
```json
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."}
{"system": "You are a helpful assistant.", "user": "Explain neural networks.", "assistant": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."}
```

### Format 2: Messages Format (OpenAI Compatible)
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain neural networks."}, {"role": "assistant", "content": "Neural networks are..."}]}
```

### Format 3: Simple Q&A
```json
{"input": "What is machine learning?", "target": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."}
{"input": "Explain neural networks.", "target": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."}
```

## 🚀 Training Command on Toy Dataset

### Step 1: Create Sample Data
```bash
# Create toy SFT dataset
mkdir -p data/raw
cat > data/raw/sft_toy_dataset.jsonl << EOF
{"system": "You are a helpful assistant.", "user": "What is Python?", "assistant": "Python is a high-level programming language known for its simplicity and readability. It's widely used for web development, data science, and artificial intelligence."}
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It involves training algorithms to recognize patterns."}
{"system": "You are a helpful assistant.", "user": "What is deep learning?", "assistant": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input."}
{"system": "You are a helpful assistant.", "user": "What is natural language processing?", "assistant": "Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language."}
{"system": "You are a helpful assistant.", "user": "What is computer vision?", "assistant": "Computer vision is an AI field that enables computers to interpret and understand visual information from the world, including images and videos."}
EOF
```

### Step 2: Configure SFT Training
```bash
# Create SFT configuration
cat > configs/my_sft.yaml << EOF
include: configs/config_base.yaml

# SFT mode configuration
mode: sft

# Model configuration
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  type: causal
  max_seq_len: 512

# Training configuration
train:
  epochs: 3
  learning_rate: 2e-4
  output_dir: outputs/sft-run

# Data configuration
data:
  format: chat
  template_path: chat_templates/default.jinja
  train_path: data/processed/train.jsonl
  val_path: data/processed/val.jsonl
  test_path: data/processed/test.jsonl
EOF
```

### Step 3: Process and Train
```bash
# Process data for SFT
make process

# Add style instructions (optional)
make style STYLE="Be helpful and concise"

# Start SFT training
make train CONFIG=configs/my_sft.yaml

# Or with monitoring
make train-bnb-tb CONFIG=configs/my_sft.yaml
```

### Step 4: Test Your Model
```bash
# Interactive chat with your SFT model
make infer CONFIG=configs/my_sft.yaml

# Test with specific questions
echo "What is Python?" | make infer-batch CONFIG=configs/my_sft.yaml
```

## ⚙️ Custom Configuration

### Model Selection
```yaml
# Small models (8GB VRAM)
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  max_seq_len: 512

# Medium models (16GB+ VRAM)
model:
  name: Qwen/Qwen2.5-7B-Instruct
  max_seq_len: 1024

# Large models (24GB+ VRAM)
model:
  name: Qwen/Qwen2.5-14B-Instruct
  max_seq_len: 2048
```

### Training Parameters
```yaml
# Faster training
train:
  epochs: 2
  learning_rate: 3e-4
  batch_size: auto
  grad_accum: auto

# Higher quality training
train:
  epochs: 5
  learning_rate: 1e-4
  warmup_ratio: 0.1

# Memory-efficient training
train:
  max_seq_len: 256
  batch_size: 1
  grad_accum: 16
```

### LoRA Configuration
```yaml
# Stronger adaptation
tuning:
  mode: qlora
  lora:
    r: 64
    alpha: 128
    dropout: 0.1
    target_modules: auto

# Lighter adaptation
tuning:
  mode: qlora
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: auto
```

### Configuration Override Examples
```bash
# Override model selection
make train CONFIG=configs/my_sft.yaml model.name=Qwen/Qwen2.5-7B-Instruct

# Override training parameters
make train CONFIG=configs/my_sft.yaml train.epochs=5 train.learning_rate=1e-4

# Override LoRA parameters
make train CONFIG=configs/my_sft.yaml tuning.lora.r=64 tuning.lora.alpha=128
```

## 📈 Advanced SFT Features

### Template Customization
```yaml
# Custom chat template
data:
  template_path: chat_templates/custom.jinja
  format: chat
```

Create custom template `chat_templates/custom.jinja`:
```jinja
{% for message in messages %}
{% if message['role'] == 'system' %}
System: {{ message['content'] }}
{% elif message['role'] == 'user' %}
Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
```

### Multi-Turn Conversations
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is Python?"},
  {"role": "assistant", "content": "Python is a programming language."},
  {"role": "user", "content": "What are its main features?"},
  {"role": "assistant", "content": "Python is known for simplicity, readability, and extensive libraries."}
]}
```

### Style and Format Control
```bash
# Apply specific style
make style STYLE="Answer in bullet points. Be concise."

# Apply format constraints
make style STYLE="Always respond in JSON format with 'answer' and 'confidence' fields."

# Apply persona
make style STYLE="You are an expert software engineer. Provide technical, detailed answers."
```

## 🔧 Evaluation and Testing

### Automatic Evaluation
```bash
# Evaluate on validation set
make eval CONFIG=configs/my_sft.yaml

# Quick evaluation (200 samples)
make eval-quick CONFIG=configs/my_sft.yaml

# Full evaluation
make eval-full CONFIG=configs/my_sft.yaml
```

### Manual Testing
```bash
# Interactive testing
make infer CONFIG=configs/my_sft.yaml

# Batch testing
echo -e "What is Python?\nExplain machine learning\nWhat is NLP?" > test_questions.txt
make infer-batch CONFIG=configs/my_sft.yaml INPUT_FILE=test_questions.txt
```

### Quality Metrics
- **ROUGE Scores**: Automatic evaluation using ROUGE metrics
- **Human Evaluation**: Manual assessment of response quality
- **Task-Specific Metrics**: Domain-specific evaluation criteria

## 📝 Best Practices

### Data Quality
- **Consistent Formatting**: Use the same format throughout your dataset
- **High-Quality Responses**: Ensure assistant responses are accurate and helpful
- **Diverse Examples**: Include various types of questions and scenarios
- **Appropriate Length**: Keep responses concise but comprehensive

### Training Strategy
- **Start Small**: Begin with smaller models and datasets
- **Monitor Training**: Use TensorBoard to track loss and metrics
- **Validate Regularly**: Check model performance on validation set
- **Experiment with Hyperparameters**: Tune learning rate, batch size, etc.

### Model Selection
- **Match Hardware**: Choose model size based on available GPU memory
- **Consider Task Complexity**: Larger models for complex tasks
- **Evaluate Base Model**: Start with a strong pre-trained model
- **Test Multiple Models**: Compare different base models

## 🚨 Troubleshooting

### Common Issues
```bash
# CUDA out of memory
make train CONFIG=configs/my_sft.yaml model.max_seq_len=256 train.batch_size=1

# Slow training
make train CONFIG=configs/my_sft.yaml tuning.backend=unsloth

# Poor responses
make train CONFIG=configs/my_sft.yaml train.epochs=5 train.learning_rate=1e-4

# Overfitting
make train CONFIG=configs/my_sft.yaml train.epochs=2 train.warmup_ratio=0.1
```

### Data Issues
```bash
# Check data format
head -n 5 data/processed/train.jsonl

# Validate data processing
make check

# Re-process data
make clean && make process
```

### Model Issues
```bash
# Test base model
make infer CONFIG=configs/config_base.yaml

# Check model loading
python -c "from transformers import AutoModelForCausalLM; print('Model loads successfully')"

# Verify LoRA application
python -c "from peft import get_peft_model; print('LoRA applied successfully')"
```

## 📚 Related Documentation

- [DAPT/CPT](dapt-cpt.md) - Domain adaptation before SFT
- [Mixed Training](mixed-training.md) - Combine domain knowledge with SFT
- [Configuration Reference](../configs/) - All configuration options
- [Advanced Configuration](../advanced/custom-configs.md) - Custom configuration patterns

## 🔗 External Resources

- [Supervised Fine-Tuning Guide](https://huggingface.co/docs/transformers/training) - HuggingFace documentation
- [Chat Template Guide](https://huggingface.co/docs/transformers/chat_templating) - Template formatting
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - LoRA research paper
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - QLoRA research paper
