# Embedding Fine-Tuning with Xyne-Play

## 🎯 What is Embedding Fine-Tuning?

Embedding Fine-Tuning improves semantic search and retrieval systems by training embedding models to better understand domain-specific similarities and relationships. Xyne-Play uses **contrastive learning** with **hard negative mining** to create superior semantic representations.

### When to Use Embedding Fine-Tuning
- **Semantic Search**: Improve search relevance for domain-specific content
- **Retrieval-Augmented Generation (RAG)**: Better document retrieval for Q&A systems
- **Document Similarity**: Find similar documents within your domain
- **Recommendation Systems**: Content-based recommendations
- **Clustering & Classification**: Group similar documents automatically

## 📊 Expected Dataset Format

### Format 1: Query-Document Pairs (Recommended)
```json
{"query": "What are the symptoms of diabetes?", "document": "Diabetes symptoms include increased thirst, frequent urination, and unexplained weight loss."}
{"query": "How does machine learning work?", "document": "Machine learning algorithms build mathematical models from training data to make predictions."}
{"query": "What causes hypertension?", "document": "Hypertension is caused by factors including genetics, diet, stress, and lack of physical activity."}
```

### Format 2: Triplets (Query, Positive, Negative)
```json
{"query": "What is diabetes?", "positive": "Diabetes is a metabolic disorder with high blood sugar levels.", "negative": "Hypertension is high blood pressure."}
{"query": "Machine learning basics", "positive": "ML algorithms learn patterns from data.", "negative": "Deep learning uses neural networks."}
```

### Format 3: Document-Document Similarity
```json
{"doc1": "Python is a high-level programming language.", "doc2": "Python is used for web development and data science.", "similarity": 0.8}
{"doc1": "Machine learning enables computers to learn.", "doc2": "ML algorithms build models from training data.", "similarity": 0.9}
```

## 🚀 Training Command on Toy Dataset

### Step 1: Create Sample Dataset
```bash
# Create toy embedding dataset
mkdir -p data/raw
cat > data/raw/embedding_dataset.jsonl << EOF
{"query": "What is machine learning?", "document": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."}
{"query": "How do neural networks work?", "document": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."}
{"query": "What is deep learning?", "document": "Deep learning is a subset of machine learning using artificial neural networks with multiple layers."}
{"query": "Explain natural language processing", "document": "Natural Language Processing (NLP) is a field of AI focused on interactions between computers and human language."}
{"query": "What is computer vision?", "document": "Computer vision is an AI field that enables computers to interpret and understand visual information from the world."}
{"query": "How does Python work?", "document": "Python is an interpreted, high-level programming language with dynamic semantics and garbage collection."}
{"query": "What are algorithms?", "document": "An algorithm is a finite sequence of well-defined instructions to solve a specific problem."}
{"query": "Explain data science", "document": "Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data."}
EOF
```

### Step 2: Setup Embedding Directories
```bash
# Create embedding-specific directories
make setup-embed-dirs

# This creates:
# embeddingFT/{configs,scripts,data,models,checkpoints,outputs}
```

### Step 3: Configure Embedding Training
```bash
# Create embedding configuration
cat > embeddingFT/configs/my_embedding.yaml << EOF
# Dataset configuration
dataset_name: "toy_embedding_dataset"
sample_size: 500
test_size: 100
seed: 42

# Model configuration
model_name: "sentence-transformers/all-MiniLM-L6-v2"
mining_model_name: "sentence-transformers/all-MiniLM-L6-v2"

# Training parameters
num_epochs: 4
batch_size: 32
learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.1

# Cache directory
cache_dir: "/tmp/embedding_cache"
EOF
```

### Step 4: Start Embedding Training
```bash
# Train embedding model
make train-embed

# Or with specific config
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml

# With TensorBoard monitoring
make train-embed-tb
```

### Step 5: Test Your Embeddings
```bash
# Test embedding similarity (when implemented)
make infer-embed

# Or test with Python
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('outputs/embedding_model')
embeddings = model.encode(['What is ML?', 'Machine learning is...'])
print('Similarity:', model.similarity(embeddings[0], embeddings[1]))
"
```

## ⚙️ Custom Configuration

### Model Selection
```yaml
# Fast training (smaller model)
model_name: "sentence-transformers/all-MiniLM-L6-v2"
mining_model_name: "sentence-transformers/all-MiniLM-L6-v2"

# Balanced quality/speed
model_name: "sentence-transformers/static-retrieval-mrl-en-v1"
mining_model_name: "sentence-transformers/static-retrieval-mrl-en-v1"

# High quality (larger model)
model_name: "sentence-transformers/stsb-roberta-large"
mining_model_name: "sentence-transformers/stsb-roberta-large"
```

### Training Parameters
```yaml
# Faster training
num_epochs: 2
batch_size: 64
learning_rate: 3e-5

# Higher quality
num_epochs: 8
batch_size: 16
learning_rate: 1e-5

# Memory efficient
batch_size: 8
gradient_checkpointing: true
```

### Domain-Specific Models
```yaml
# Biomedical domain
model_name: "sentence-transformers/allenai-specter"
mining_model_name: "sentence-transformers/allenai-specter"

# Legal domain
model_name: "sentence-transformers/legal-roberta-base"
mining_model_name: "sentence-transformers/legal-roberta-base"

# Multilingual
model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
mining_model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Configuration Override Examples
```bash
# Override model selection
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml model_name=sentence-transformers/all-MiniLM-L6-v2

# Override training parameters
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml num_epochs=6 batch_size=16

# Override dataset size
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml sample_size=1000 test_size=200
```

## 📈 Advanced Features

### Hard Negative Mining
```yaml
# Mining configuration
mining:
  num_negatives: 3          # Number of hard negatives per query
  batch_size: 256          # Mining batch size
  margin: 0.1              # Similarity margin threshold
  range_min: 0             # Minimum similarity range
  range_max: 100           # Maximum similarity range
  sampling_strategy: "top" # How to select negatives
```

### Multiple Negatives Ranking Loss (MNRL)
```python
# The training process:
# 1. Encode all queries and documents
# 2. For each query, find hard negatives using mining model
# 3. Create (query, positive, negative) triplets
# 4. Train with MNRL loss
# 5. In-batch negatives provide additional training signals
```

### Evaluation Metrics
```yaml
# Evaluation configuration
evaluation:
  metrics: ["accuracy", "map", "mrr", "ndcg"]
  eval_steps: 100
  save_best_model: true
  early_stopping_patience: 3
```

## 🔧 Integration with RAG Systems

### Using Fine-Tuned Embeddings in RAG
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load fine-tuned model
model = SentenceTransformer('outputs/embedding_model')

# Encode documents
documents = ["Doc 1 content", "Doc 2 content", ...]
doc_embeddings = model.encode(documents)

# Create FAISS index
index = faiss.IndexFlatIP(len(doc_embeddings[0]))
index.add(np.array(doc_embeddings))

# Search function
def search(query, k=5):
    query_embedding = model.encode([query])
    scores, indices = index.search(query_embedding, k)
    return [(documents[i], scores[0][j]) for j, i in enumerate(indices[0])]
```

### Integration with Xyne-Play LLM Training
```bash
# 1. Fine-tune embeddings first
make train-embed

# 2. Use embeddings to create better training data for LLM
python scripts/create_rag_dataset.py \
  --embedding_model outputs/embedding_model \
  --documents data/raw/domain_docs \
  --output data/processed/rag_enhanced_data.jsonl

# 3. Train LLM with enhanced data
make process && make train-bnb-tb
```

## 📝 Best Practices

### Dataset Quality
- Use high-quality, relevant query-document pairs
- Ensure diverse coverage of your domain
- Include difficult negative examples
- Balance positive and negative examples

### Model Selection
- Start with smaller models for faster iteration
- Use domain-specific models when available
- Consider multilingual models for international use
- Balance model size vs. inference speed

### Training Strategy
- Monitor training loss and validation metrics
- Use early stopping to prevent overfitting
- Experiment with different learning rates
- Validate on held-out test set

## 🚨 Troubleshooting

### Common Issues
```bash
# CUDA out of memory
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml batch_size=16

# Slow training
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml model_name=sentence-transformers/all-MiniLM-L6-v2

# Poor results
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml num_epochs=8 learning_rate=1e-5
```

### Quality Issues
```bash
# Improve mining quality
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml mining_model_name=sentence-transformers/stsb-roberta-large

# Add more training data
make train-embed EMBED_CONFIG=embeddingFT/configs/my_embedding.yaml sample_size=2000
```

## 📚 Related Documentation

- [LLM Fine-Tuning](llm-fine-tuning.md) - Instruction fine-tuning
- [DAPT/CPT](dapt-cpt.md) - Domain adaptation
- [Configuration Reference](../configs/) - All configuration options
- [Advanced Configuration](../advanced/custom-configs.md) - Custom configuration patterns

## 🔗 External Resources

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Multiple Negatives Ranking Loss Paper](https://arxiv.org/abs/2004.09813)
- [Hard Negative Mining Techniques](https://arxiv.org/abs/2008.02240)
