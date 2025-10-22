# Domain-Adaptive Pretraining (DAPT) & Continual Pretraining (CPT)

## 🎯 What is DAPT/CPT?

**Domain-Adaptive Pretraining (DAPT)** and **Continual Pretraining (CPT)** are techniques to teach language models domain-specific knowledge by training them on specialized documents before or during instruction fine-tuning.

### When to Use DAPT/CPT
- **Domain Expert Models**: Medical, legal, financial, scientific models
- **Specialized Knowledge**: Technical documentation, research papers
- **Knowledge Base**: Train on company internal documents
- **Foundation for SFT**: Pre-train on domain before instruction fine-tuning

## 📊 Expected Dataset Format

### Format 1: Raw Documents (Recommended)
Place your documents in `data/raw/`:
```
data/raw/
├── medical_papers/
│   ├── paper1.pdf
│   ├── paper2.docx
│   └── paper3.txt
├── legal_docs/
│   ├── contract1.pdf
│   └── regulation2.docx
└── technical_docs/
    ├── api_spec.pdf
    └── user_guide.docx
```

### Format 2: Preprocessed Text
```json
{"text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."}
{"text": "Deep learning neural networks consist of multiple layers that transform input data into increasingly abstract representations."}
{"text": "Natural language processing enables computers to understand, interpret, and generate human language."}
```

### Format 3: Mixed DAPT + SFT
```yaml
# In your config file
datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.7
  - type: chat
    path: data/processed/instruction_data.jsonl
    weight: 0.3
```

## 🚀 Training Command on Toy Dataset

### Step 1: Prepare Domain Documents
```bash
# Create sample domain documents
mkdir -p data/raw/domain_docs

# Create sample medical text
cat > data/raw/domain_docs/medical_notes.txt << EOF
Hypertension, also known as high blood pressure, is a chronic medical condition in which the blood pressure in the arteries is persistently elevated.
Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels over a prolonged period.
Cardiovascular disease refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain, or stroke.
EOF

# Create sample technical documentation
cat > data/raw/domain_docs/tech_docs.txt << EOF
Python is an interpreted, high-level programming language with dynamic semantics.
Machine learning algorithms build a mathematical model based on sample data, known as training data.
Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
EOF
```

### Step 2: Process Documents for DAPT
```bash
# Convert documents to training format
make dapt-docx

# This will:
# 1. Extract text from PDF/DOCX files
# 2. Clean and preprocess the text
# 3. Create CPT-compatible datasets
# 4. Save to data/processed/ for training
```

### Step 3: Configure DAPT Training
```bash
# Create DAPT configuration
cp configs/config_base.yaml configs/my_dapt.yaml

# Edit the config for DAPT mode
cat > configs/my_dapt.yaml << EOF
include: configs/config_base.yaml

# Enable DAPT mode
mode: cpt

# Model configuration
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  type: causal
  max_seq_len: 512

# Training configuration
train:
  epochs: 2
  learning_rate: 1e-4
  output_dir: outputs/dapt-run

# CPT-specific settings
block_size: 512
pack_factor: 4
EOF
```

### Step 4: Start DAPT Training
```bash
# Train domain-adaptive model
make dapt-train

# Or with specific config
make train CONFIG=configs/my_dapt.yaml
```

### Step 5: Add Instruction Following (Optional)
```bash
# Create instruction data for the domain
cat > data/raw/domain_instructions.jsonl << EOF
{"system": "You are a medical assistant.", "user": "What is hypertension?", "assistant": "Hypertension is a chronic condition characterized by persistently elevated blood pressure in the arteries."}
{"system": "You are a medical assistant.", "user": "What are the symptoms of diabetes?", "assistant": "Diabetes symptoms include increased thirst, frequent urination, unexplained weight loss, and fatigue."}
{"system": "You are a medical assistant.", "user": "What causes cardiovascular disease?", "assistant": "Cardiovascular disease is caused by narrowed or blocked blood vessels due to plaque buildup."}
EOF

# Process instruction data
make process

# Mixed training (DAPT + SFT)
cat > configs/mixed_training.yaml << EOF
include: configs/config_base.yaml

# Mixed mode: DAPT + SFT
mode: cpt_mixed

# Dataset configuration
datasets:
  - type: cpt
    path: data/processed/domain_documents.jsonl
    weight: 0.7
  - type: chat
    path: data/processed/train.jsonl
    weight: 0.3

# Training configuration
train:
  epochs: 3
  learning_rate: 2e-4
  output_dir: outputs/mixed-run
EOF

# Start mixed training
make train CONFIG=configs/mixed_training.yaml
```

## ⚙️ Custom Configuration

### DAPT-Specific Configuration
```yaml
# configs/dapt_config.yaml
include: configs/config_base.yaml

# Training mode
mode: cpt

# CPT parameters
block_size: 1024          # Context window for pretraining
pack_factor: 4            # How many documents to pack per sequence

# Model configuration
model:
  name: Qwen/Qwen2.5-7B-Instruct
  max_seq_len: 1024

# Training parameters
train:
  epochs: 2               # Fewer epochs for pretraining
  learning_rate: 1e-4     # Lower learning rate for stability
  batch_size: auto
  grad_accum: auto
```

### Mixed Training Configuration
```yaml
# configs/mixed_dapt_sft.yaml
include: configs/config_base.yaml

# Mixed mode configuration
mode: cpt_mixed

# Dataset weights
datasets:
  - type: cpt
    path: data/processed/domain_docs.jsonl
    weight: 0.6           # 60% domain pretraining
  - type: chat
    path: data/processed/instructions.jsonl
    weight: 0.4           # 40% instruction following

# Training parameters
train:
  epochs: 3
  learning_rate: 2e-4
  warmup_ratio: 0.1
```

### Configuration Override Examples
```bash
# Override block size for longer context
make train CONFIG=configs/my_dapt.yaml block_size=1024 pack_factor=2

# Override dataset weights
make train CONFIG=configs/mixed_training.yaml datasets[0].weight=0.8 datasets[1].weight=0.2

# Override learning rate for domain pretraining
make train CONFIG=configs/my_dapt.yaml train.learning_rate=5e-5 train.epochs=3
```

## 📈 Advanced DAPT Features

### Document Processing Pipeline
```bash
# Process different document types
make dapt-docx                    # Process DOCX files
make dapt-pdf                     # Process PDF files (if implemented)
make dapt-txt                     # Process TXT files

# Custom document processing
python scripts/ingest_documents.py \
  --input_dir data/raw/my_docs \
  --output data/processed/my_domain.jsonl \
  --chunk_size 1000 \
  --overlap 100
```

### Quality Filtering
```yaml
# Add quality filters to your config
data:
  min_chunk_length: 100          # Minimum characters per chunk
  max_chunk_length: 2000         # Maximum characters per chunk
  quality_threshold: 0.8         # Minimum quality score
  remove_duplicates: true        # Remove duplicate content
```

### Domain-Specific Tokenization
```yaml
# Add domain-specific tokens
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  special_tokens:
    - "[MED_START]"
    - "[MED_END]"
    - "[TECH_START]"
    - "[TECH_END]"
```

## 🔧 Evaluation and Testing

### Domain Knowledge Evaluation
```bash
# Test domain knowledge
make infer

# Example questions for medical domain:
# "What are the symptoms of hypertension?"
# "How is diabetes diagnosed?"
# "What are the risk factors for cardiovascular disease?"
```

### Mixed Capability Testing
```bash
# Test both domain knowledge and chat capabilities
make infer-batch

# Create test file with mixed questions
cat > domain_test_questions.txt << EOF
What is hypertension?
Explain machine learning in simple terms.
How does deep learning work?
What causes diabetes?
EOF
```

## 📝 Best Practices

### Document Quality
- Use high-quality, accurate domain documents
- Remove irrelevant or low-quality content
- Ensure consistent terminology and formatting
- Include diverse examples within your domain

### Training Strategy
- Start with pure DAPT, then add SFT if needed
- Use lower learning rates for domain pretraining
- Monitor for catastrophic forgetting
- Validate on domain-specific test sets

### Data Preparation
- Chunk documents appropriately (500-2000 characters)
- Remove duplicate content
- Filter out low-quality text
- Maintain document context when possible

## 🚨 Troubleshooting

### Common Issues
```bash
# Poor domain knowledge retention
make train CONFIG=configs/my_dapt.yaml train.epochs=3 datasets[0].weight=0.8

# Catastrophic forgetting of general capabilities
make train CONFIG=configs/mixed_training.yaml datasets[1].weight=0.5

# Memory issues with large documents
make train CONFIG=configs/my_dapt.yaml block_size=256 pack_factor=2
```

### Quality Issues
```bash
# Improve document quality
python scripts/filter_documents.py \
  --input data/processed/domain_docs.jsonl \
  --output data/processed/filtered_docs.jsonl \
  --min_length 200 \
  --max_length 1500
```

## 📚 Related Documentation

- [Supervised Fine-Tuning](supervised-fine-tuning.md) - Pure instruction following
- [Mixed Training](mixed-training.md) - Combine domain knowledge with SFT
- [Embedding Fine-Tuning](embedding-fine-tuning.md) - Semantic embedding training
- [Configuration Reference](../configs/) - All configuration options
- [Advanced Configuration](../advanced/custom-configs.md) - Custom configuration patterns
