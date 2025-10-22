# Mixed Training: Domain Knowledge + Instruction Following with Xyne-Play

## 🎯 What is Mixed Training?

Mixed training combines **Domain-Adaptive Pretraining (DAPT)** with **Supervised Fine-Tuning (SFT)** in a single training run. This approach teaches models both domain-specific knowledge and instruction-following capabilities simultaneously.

### When to Use Mixed Training
- **Domain Expert Chatbots**: Create medical assistants that can chat naturally
- **Technical Support**: Build help systems with domain knowledge and conversational ability
- **Educational Tutors**: Combine subject matter expertise with teaching skills
- **Professional Assistants**: Legal, financial, or scientific assistants that can explain concepts

## 📊 Expected Dataset Format

### Mixed Dataset Configuration
```yaml
# In your config file
mode: cpt_mixed

datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.7
  - type: chat
    path: data/processed/instruction_data.jsonl
    weight: 0.3
```

### Domain Documents (CPT Component)
```json
{"text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."}
{"text": "Deep learning neural networks consist of multiple layers that transform input data into increasingly abstract representations."}
{"text": "Natural language processing enables computers to understand, interpret, and generate human language."}
```

### Instruction Data (SFT Component)
```json
{"system": "You are a helpful ML tutor.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."}
{"system": "You are a helpful ML tutor.", "user": "Explain neural networks.", "assistant": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."}
```

## 🚀 Training Command on Toy Dataset

### Step 1: Prepare Mixed Dataset
```bash
# Create domain documents
mkdir -p data/raw
cat > data/raw/domain_docs.txt << EOF
Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.
Deep learning is a subset of machine learning using artificial neural networks with multiple layers.
Natural language processing is a field of AI focused on interactions between computers and human language.
Computer vision is an AI field that enables computers to interpret and understand visual information.
Python is a high-level programming language widely used in machine learning and data science.
EOF

# Create instruction data
cat > data/raw/instruction_data.jsonl << EOF
{"system": "You are a helpful ML tutor.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."}
{"system": "You are a helpful ML tutor.", "user": "What is deep learning?", "assistant": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to progressively extract higher-level features."}
{"system": "You are a helpful ML tutor.", "user": "What is NLP?", "assistant": "Natural language processing is a field of AI focused on enabling computers to understand, interpret, and generate human language."}
{"system": "You are a helpful ML tutor.", "user": "What is computer vision?", "assistant": "Computer vision is an AI field that enables computers to interpret and understand visual information from the world, including images and videos."}
{"system": "You are a helpful ML tutor.", "user": "Why use Python for ML?", "assistant": "Python is preferred for machine learning due to its simplicity, extensive libraries like TensorFlow and PyTorch, and strong community support."}
EOF
```

### Step 2: Process Data for Mixed Training
```bash
# Process domain documents for CPT
make dapt-docx

# Process instruction data for SFT
make process

# This creates:
# - data/processed/domain_documents.jsonl (from domain_docs.txt)
# - data/processed/train.jsonl (from instruction_data.jsonl)
```

### Step 3: Configure Mixed Training
```bash
# Create mixed training configuration
cat > configs/mixed_training.yaml << EOF
include: configs/config_base.yaml

# Mixed mode configuration
mode: cpt_mixed

# Dataset configuration
datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.7           # 70% domain pretraining
  - type: chat
    path: data/processed/train.jsonl
    weight: 0.3           # 30% instruction following

# Model configuration
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  type: causal
  max_seq_len: 512

# Training configuration
train:
  epochs: 4
  learning_rate: 2e-4
  output_dir: outputs/mixed-run

# CPT-specific settings
block_size: 512
pack_factor: 4
EOF
```

### Step 4: Start Mixed Training
```bash
# Start mixed training
make train CONFIG=configs/mixed_training.yaml

# Or with monitoring
make train-bnb-tb CONFIG=configs/mixed_training.yaml
```

### Step 5: Test Your Model
```bash
# Test domain knowledge
make infer CONFIG=configs/mixed_training.yaml

# Example questions:
# "What is machine learning?" (should show domain knowledge)
# "Explain ML like I'm 10" (should show instruction following)
# "What are the applications of deep learning?" (should combine both)
```

## ⚙️ Custom Configuration

### Dataset Weight Balancing
```yaml
# Domain-heavy training
datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.8
  - type: chat
    path: data/processed/train.jsonl
    weight: 0.2

# Instruction-heavy training
datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.4
  - type: chat
    path: data/processed/train.jsonl
    weight: 0.6

# Balanced training
datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.5
  - type: chat
    path: data/processed/train.jsonl
    weight: 0.5
```

### Advanced Configuration
```yaml
# Mixed training with custom parameters
mode: cpt_mixed

datasets:
  - type: cpt
    path: data/processed/medical_papers.jsonl
    weight: 0.6
  - type: chat
    path: data/processed/medical_qa.jsonl
    weight: 0.4

# CPT parameters
block_size: 1024          # Larger context for domain documents
pack_factor: 2            # Less packing for longer documents

# Training parameters
train:
  epochs: 5
  learning_rate: 1e-4     # Lower learning rate for mixed training
  warmup_ratio: 0.1
```

### Configuration Override Examples
```bash
# Override dataset weights
make train CONFIG=configs/mixed_training.yaml datasets[0].weight=0.8 datasets[1].weight=0.2

# Override block size for longer context
make train CONFIG=configs/mixed_training.yaml block_size=1024 pack_factor=2

# Override learning rate for stability
make train CONFIG=configs/mixed_training.yaml train.learning_rate=1e-4 train.epochs=5
```

## 📈 Advanced Mixed Training Features

### Three-Way Mixing
```yaml
# Mix domain, instruction, and general data
datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.5
  - type: chat
    path: data/processed/instruction_data.jsonl
    weight: 0.3
  - type: chat
    path: data/processed/general_chat.jsonl
    weight: 0.2
```

### Curriculum Learning
```yaml
# Start with domain-heavy, gradually shift to instruction-heavy
# Phase 1 (epochs 1-2): 80% domain, 20% instruction
# Phase 2 (epochs 3-4): 50% domain, 50% instruction
# Phase 3 (epochs 5-6): 20% domain, 80% instruction
```

### Domain-Specific Templates
```yaml
# Custom system prompts for different domains
data:
  system_prompts:
    domain: "You are an expert in {domain}. Provide accurate, detailed information."
    instruction: "You are a helpful tutor specializing in {domain}. Explain concepts clearly."
```

## 🔧 Evaluation and Testing

### Dual Evaluation
```bash
# Test domain knowledge
echo "What are the key concepts in machine learning?" | make infer-batch CONFIG=configs/mixed_training.yaml

# Test instruction following
echo "Explain machine learning to a beginner" | make infer-batch CONFIG=configs/mixed_training.yaml

# Test mixed capabilities
echo "Create a lesson plan for teaching neural networks" | make infer-batch CONFIG=configs/mixed_training.yaml
```

### Quality Assessment
```bash
# Create comprehensive test file
cat > mixed_test_questions.txt << EOF
# Domain knowledge questions
What is the difference between supervised and unsupervised learning?
How do neural networks work?
What are the main applications of NLP?

# Instruction following questions
Explain machine learning in simple terms
Create a tutorial for deep learning beginners
Summarize the key concepts in computer vision

# Mixed capability questions
Design a machine learning course for beginners
Explain how to implement a neural network for image classification
What advice would you give someone starting in AI?
EOF

# Run comprehensive test
make infer-batch CONFIG=configs/mixed_training.yaml INPUT_FILE=mixed_test_questions.txt
```

## 📝 Best Practices

### Dataset Preparation
- **Quality Domain Documents**: Use authoritative, accurate domain content
- **Diverse Instruction Data**: Include various question types and formats
- **Balanced Examples**: Ensure both domain and instruction aspects are well-represented
- **Consistent Formatting**: Maintain consistent data formats across both datasets

### Training Strategy
- **Start with Balanced Weights**: Begin with 50/50 or 60/40 splits
- **Monitor Both Aspects**: Evaluate both domain knowledge and instruction following
- **Adjust Weights Based on Results**: Fine-tune based on evaluation outcomes
- **Use Appropriate Learning Rates**: Mixed training may require lower learning rates

### Model Selection
- **Strong Base Models**: Start with models that have good domain knowledge
- **Appropriate Size**: Choose model size based on task complexity and hardware
- **Consider Multi-Modal**: For domains requiring visual understanding
- **Test Different Architectures**: Compare transformer variants for your domain

## 🚨 Troubleshooting

### Common Issues
```bash
# Poor domain knowledge retention
make train CONFIG=configs/mixed_training.yaml datasets[0].weight=0.8 datasets[1].weight=0.2

# Weak instruction following
make train CONFIG=configs/mixed_training.yaml datasets[0].weight=0.3 datasets[1].weight=0.7

# Catastrophic forgetting of general capabilities
make train CONFIG=configs/mixed_training.yaml datasets[0].weight=0.4 datasets[1].weight=0.4
# Add general chat data as third dataset with weight 0.2
```

### Training Instability
```bash
# Reduce learning rate for mixed training
make train CONFIG=configs/mixed_training.yaml train.learning_rate=1e-4

# Increase warmup ratio
make train CONFIG=configs/mixed_training.yaml train.warmup_ratio=0.2

# Reduce block size for memory efficiency
make train CONFIG=configs/mixed_training.yaml block_size=256 pack_factor=4
```

### Quality Issues
```bash
# Improve domain document quality
python scripts/filter_documents.py \
  --input data/processed/domain_documents.jsonl \
  --output data/processed/filtered_domain.jsonl \
  --min_length 100 \
  --max_length 1500

# Enhance instruction data diversity
python scripts/augment_instructions.py \
  --input data/processed/train.jsonl \
  --output data/processed/augmented_train.jsonl
```

## 📚 Related Documentation

- [Supervised Fine-Tuning](supervised-fine-tuning.md) - Pure instruction following
- [DAPT/CPT](dapt-cpt.md) - Pure domain adaptation
- [Configuration Reference](../configs/) - All configuration options
- [Advanced Configuration](../advanced/custom-configs.md) - Custom configuration patterns

## 🔗 External Resources

- [Mixed Training Research](https://arxiv.org/abs/2309.09153) - Recent papers on mixed training
- [Curriculum Learning](https://arxiv.org/abs/1909.09157) - Curriculum learning strategies
- [Multi-Task Learning](https://arxiv.org/abs/1706.05098) - Multi-task learning principles
