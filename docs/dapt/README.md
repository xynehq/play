# Domain-Adaptive Pretraining (DAPT)

Domain-Adaptive Pretraining teaches language models specialized knowledge from specific domains like medicine, law, finance, or any specialized field through document-based training.

## What DAPT Does

DAPT adapts general-purpose language models to become domain experts by training them on domain-specific documents and knowledge sources. This process enables models to:

- Understand specialized terminology and concepts
- Provide accurate domain-specific information
- Adopt professional communication styles
- Handle domain-specific tasks and queries

## Expected Dataset Format

### Document-Based Training
DAPT works with various document formats placed in `data/raw/`:

```
data/raw/
├── medical_papers/
│   ├── paper1.pdf
│   ├── paper2.docx
│   └── research_notes.txt
├── legal_documents/
│   ├── contracts.pdf
│   ├── case_studies.docx
│   └── regulations.txt
└── custom_domain/
    ├── domain_docs.pdf
    └── notes.docx
```

### Supported Document Formats
- **PDF**: Research papers, articles, documentation
- **DOCX**: Structured documents, reports, notes
- **TXT**: Plain text files, documentation
- **JSONL**: Structured domain data

### Mixed Training Data (DAPT + SFT)
For combined domain knowledge and instruction following:

```json
{"system": "You are a medical expert.", "user": "What are the symptoms of diabetes?", "assistant": "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, and fatigue."}
{"system": "You are a medical expert.", "user": "Explain Type 1 vs Type 2 diabetes.", "assistant": "Type 1 diabetes is an autoimmune condition where the body doesn't produce insulin, while Type 2 diabetes involves insulin resistance where cells don't respond effectively to insulin."}
```

## Training Command on Toy Dataset

### Quick Start with Documents
```bash
# 1. Add your documents to data/raw/
mkdir -p data/raw/my_domain
# Copy your PDF/DOCX files here

# 2. Process documents for training
make dapt-docx

# 3. Start domain adaptation training
make dapt-train

# 4. Test domain knowledge
make infer
```

### Step-by-Step DAPT Training
```bash
# 1. Prepare domain documents
mkdir -p data/raw/medical
# Copy medical documents to this folder

# 2. Process documents into training format
make dapt-docx

# 3. (Optional) Add instruction data
echo '{"system": "You are a medical expert.", "user": "What is hypertension?", "assistant": "Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is consistently too high."}' > data/raw/medical_instructions.jsonl

# 4. Start DAPT training
make dapt-train

# 5. Monitor progress at http://localhost:6006
```

### Training Commands
```bash
# Document processing
make dapt-docx                    # Process all documents in data/raw/

# DAPT training
make dapt-train                   # Domain-adaptive pretraining
make dapt-train-tb               # With TensorBoard monitoring

# Mixed training (DAPT + SFT)
make dapt-mixed                   # Combined domain + instruction training
```

## Custom Configuration and Overrides

### DAPT-Specific Configuration
Create a DAPT-optimized configuration:

```yaml
# configs/my_dapt_config.yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_seq_len: 2048              # Longer for document context

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 32                        # Higher rank for domain knowledge
    alpha: 64
    dropout: 0.05                # Lower dropout for knowledge retention

train:
  epochs: 5                      # More epochs for domain adaptation
  learning_rate: 1e-4            # Lower learning rate for stability
  batch_size: 2                  # Smaller batches for long documents
  eval_strategy: "steps"
  eval_steps: 200
  
  # DAPT-specific settings
  task_mode: "cpt_mixed"         # Mixed CPT + instruction training
  cpt_weight: 0.7                # Weight for document training
  sft_weight: 0.3                # Weight for instruction training
```

### Override DAPT Parameters
```bash
# Override domain weights
make dapt-train CONFIG=configs/run_dapt.yaml TRAIN_CPT_WEIGHT=0.8 TRAIN_SFT_WEIGHT=0.2

# Override learning rate for domain training
make dapt-train CONFIG=configs/run_dapt.yaml TRAIN_LEARNING_RATE=5e-5

# Override model for domain adaptation
make dapt-train CONFIG=configs/run_dapt.yaml MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
```

### Document Processing Configuration
```yaml
# Document processing settings
data:
  chunk_size: 512                # Document chunk size
  chunk_overlap: 50              # Overlap between chunks
  min_chunk_length: 100          # Minimum chunk length
  
  # File processing
  supported_formats: ["pdf", "docx", "txt", "jsonl"]
  max_file_size_mb: 50           # Maximum file size
  
  # Quality filters
  min_text_length: 50            # Minimum text length per chunk
  remove_duplicates: true        # Remove duplicate chunks
```

## Advanced DAPT Techniques

### Progressive Domain Training
```bash
# Stage 1: Basic domain knowledge
make dapt-train CONFIG=configs/dapt_stage1.yaml TRAIN_EPOCHS=3

# Stage 2: Specialized knowledge
make dapt-train CONFIG=configs/dapt_stage2.yaml TRAIN_EPOCHS=2

# Stage 3: Instruction following
make dapt-train CONFIG=configs/dapt_stage3.yaml TRAIN_TASK_MODE="sft_only"
```

### Multi-Domain Training
```bash
# Train on multiple domains simultaneously
# Organize documents by domain:
data/raw/
├── medical/
├── legal/
└── financial/

# Process all domains
make dapt-docx

# Train with domain balancing
make dapt-train CONFIG=configs/multi_domain.yaml
```

### Curriculum Learning
```bash
# Start with general domain, then specialize
# Week 1: General medical knowledge
make dapt-train CONFIG=configs/medical_general.yaml

# Week 2: Specialized medical field
make dapt-train CONFIG=configs/medical_specialized.yaml

# Week 3: Clinical applications
make dapt-train CONFIG=configs/medical_clinical.yaml
```

## Domain-Specific Best Practices

### Medical Domain
```yaml
# Medical-specific configuration
train:
  learning_rate: 5e-5            # Very low for medical accuracy
  epochs: 8                      # More epochs for medical knowledge
  batch_size: 1                  # Small batches for precision
  
data:
  chunk_size: 1024               # Larger chunks for medical context
  min_text_length: 200           # Longer minimum for medical texts
```

### Legal Domain
```yaml
# Legal-specific configuration
train:
  learning_rate: 1e-4
  epochs: 6
  eval_steps: 100                # More frequent evaluation
  
data:
  chunk_size: 768                # Medium chunks for legal documents
  remove_duplicates: true        # Critical for legal texts
```

### Financial Domain
```yaml
# Financial-specific configuration
train:
  learning_rate: 2e-4
  epochs: 5
  weight_decay: 0.1              # Higher regularization for financial data
  
data:
  chunk_size: 512                # Smaller chunks for financial data
  chunk_overlap: 100             # More overlap for context
```

## Evaluation and Testing

### Domain Knowledge Evaluation
```bash
# Evaluate domain knowledge
make eval

# Test with domain-specific questions
make infer

# Create domain test suite
echo "What are the key principles of medical ethics?" > medical_test.txt
echo "Explain the difference between civil and criminal law." >> medical_test.txt
make infer-batch
```

### Quality Assessment
```bash
# Check domain accuracy
python scripts/evaluate_domain.py --domain medical --test_file medical_test.txt

# Compare with baseline model
python scripts/compare_models.py --baseline_model base --adapted_model adapters/last
```

## Troubleshooting

### Common DAPT Issues
- **Poor domain knowledge**: Increase training epochs or use more domain data
- **Overfitting to domain**: Reduce domain weight or add regularization
- **Memory issues with documents**: Reduce chunk size or batch size
- **Slow training**: Use fewer documents or smaller chunks

### Document Processing Issues
```bash
# Check document processing
python scripts/check_documents.py --data_dir data/raw/

# Re-process documents with different settings
make dapt-docx DATA_CHUNK_SIZE=256 DATA_MIN_TEXT_LENGTH=50
```

### Monitoring DAPT Training
```bash
# Monitor domain knowledge acquisition
make tensorboard
# Look for: decreasing loss, improving domain accuracy

# Check memory usage
make memory-check

# Validate document chunks
python scripts/validate_chunks.py --data_dir data/processed/
```

## Integration with SFT

### Combined DAPT + SFT Pipeline
```bash
# Stage 1: Domain adaptation
make dapt-train

# Stage 2: Instruction following on domain
make style STYLE="You are a medical expert. Provide accurate, evidence-based answers."
make train-bnb-tb

# Stage 3: Mixed training
make dapt-mixed
```

### Progressive Specialization
```bash
# Start with general SFT
make train-bnb-tb CONFIG=configs/general_sft.yaml

# Then specialize with DAPT
make dapt-train CONFIG=configs/domain_specialization.yaml

# Finally, fine-tune with domain instructions
make train-bnb-tb CONFIG=configs/domain_sft.yaml
```

## Next Steps

- **[Supervised Fine-Tuning](../sft/)** - Add instruction following capabilities
- **[Multi-GPU Training](../multi-gpu/)** - Scale to larger domain models
- **[Configuration Guide](../configuration/)** - Advanced configuration options
- **[Examples](../../examples/)** - Complete domain adaptation examples
