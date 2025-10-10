# Embedding Fine-Tuning

Advanced embedding fine-tuning using MultipleNegativesRankingLoss with hard negative mining for superior semantic representations.

## How It Works

### Core Technology

The system uses **contrastive learning** to improve embedding quality by teaching the model to distinguish between similar and dissimilar text pairs. Here's the technical breakdown:

#### 1. Hard Negative Mining
```python
# Uses a separate mining model to find difficult negatives
embed_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
hard_negatives = mine_hard_negatives(dataset, embed_model, ...)
```

**Process:**
- **Mining Model**: A pre-trained sentence transformer encodes all documents
- **Similarity Search**: For each anchor, finds the most similar non-matching documents
- **Hard Negatives**: These are "tricky" examples that look similar but are incorrect matches
- **Triplet Generation**: Creates (anchor, positive, negative) triplets for training

#### 2. MultipleNegativesRankingLoss (MNRL)
```python
loss = MultipleNegativesRankingLoss(model)
```

**Technical Details:**
- **In-Batch Negatives**: Each positive pair is compared against all other pairs in the batch
- **Efficiency**: One forward pass provides N-1 negative examples per positive pair
- **No Duplicates**: `BatchSamplers.NO_DUPLICATES` prevents anchor appearing as its own negative
- **Temperature Scaling**: Controls the sharpness of similarity distributions

#### 3. Training Pipeline
```python
# 1. Baseline Evaluation
baseline_score = evaluate_baseline(model, dataset)

# 2. Hard Negative Mining
hard_triplets = mine_hard_negatives(dataset, mining_model)

# 3. Contrastive Training
trainer.train(hard_triplets, MNRL_loss)

# 4. Final Evaluation
final_score = evaluate_model(model, dataset)
```

### Architecture Components

#### Data Flow
```
Raw Dataset → Train/Eval Split → Baseline Eval → Hard Negative Mining → Triplet Generation → MNRL Training → Final Model
```

#### Model Roles
- **Main Model**: The model being fine-tuned (e.g., `google/embeddinggemma-300m`)
- **Mining Model**: Static model for finding hard negatives (runs on CPU)
- **Loss Function**: MNRL for efficient contrastive learning
- **Evaluator**: TripletEvaluator for performance measurement

#### Memory Management
- **CPU Mining**: Mining model runs on CPU to save GPU memory
- **Chunked Processing**: Large datasets processed in 200K sample chunks
- **Batch Sampling**: No duplicate samples prevent memory waste

## Technical Implementation

### Hard Negative Mining Algorithm

```python
def mine_hard_negatives(dataset, mining_model, **kwargs):
    # 1. Encode all documents with mining model
    embeddings = mining_model.encode(dataset['answer'])
    
    # 2. For each query, find most similar non-matching documents
    similarities = cosine_similarity(query_embeddings, document_embeddings)
    
    # 3. Select top-k hardest negatives
    hard_negatives = top_k_similar(similarities, exclude_matches=True)
    
    return triplets  # (anchor, positive, negative)
```

### MNRL Loss Computation

```python
# For each batch:
# anchors: [a1, a2, a3, ..., an]
# positives: [p1, p2, p3, ..., pn]

# Compute similarities
similarities = model.compute_similarity(anchors, all_candidates)

# Loss: maximize positive similarity, minimize negative similarity
loss = -log(exp(sim(a1,p1)) / sum(exp(sim(a1,pi)) for all i))
```

### Evaluation Metrics

```python
# TripletEvaluator computes:
# - Accuracy: anchor closer to positive than negative
# - Average similarity scores
# - Ranking metrics
```

## Quick Start

```bash
# Setup directories
make setup-embed-dirs

# Start training
make train-embed

# With TensorBoard
make train-embed-tb
```

## Configuration

Edit `configs/run_embedding_mnrl.yaml`:

```yaml
# Dataset configuration
dataset_name: "ayushexel/xyneft"
sample_size: 1080
test_size: 250
seed: 12

# Model configuration
model_name: "google/embeddinggemma-300m"
mining_model_name: "sentence-transformers/static-retrieval-mrl-en-v1"

# Training parameters
num_epochs: 8
batch_size: 128
learning_rate: 3e-5
weight_decay: 0.01
warmup_ratio: 0.1

# Cache directory
cache_dir: "/models"
```

## Mining Model Selection

### Performance vs Speed Trade-offs

```yaml
# Fast mining (smaller model, less accurate)
mining_model_name: "sentence-transformers/all-MiniLM-L6-v2"
# Speed: ~2x faster, Quality: Good

# Balanced mining (default)
mining_model_name: "sentence-transformers/static-retrieval-mrl-en-v1"
# Speed: Baseline, Quality: High

# High-quality mining (larger model, slower)
mining_model_name: "sentence-transformers/stsb-roberta-large"
# Speed: ~2x slower, Quality: Very High
```

### Domain-Specific Mining

```yaml
# Biomedical domain
mining_model_name: "sentence-transformers/allenai-specter"

# Legal domain
mining_model_name: "sentence-transformers/legal-roberta-base"

# Multilingual
mining_model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## Performance Optimization

### Memory Optimization
```yaml
# Reduce batch size for GPU memory constraints
batch_size: 64  # instead of 128

# Use smaller mining model
mining_model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

### Speed Optimization
```yaml
# Smaller dataset for testing
sample_size: 500
test_size: 100

# Fewer epochs
num_epochs: 3
```

### Quality Optimization
```yaml
# More data
sample_size: 5000
test_size: 500

# Longer training
num_epochs: 10
learning_rate: 1e-5

# Better mining model
mining_model_name: "sentence-transformers/static-retrieval-mrl-en-v1"
```

## Advanced Configuration

### Custom Mining Parameters
```yaml
# These are used in the mining process
mining:
  num_negatives: 1          # Number of negatives per anchor
  batch_size: 512          # Mining batch size
  margin: 0                # Similarity margin threshold
  range_min: 0             # Minimum similarity range
  range_max: 100           # Maximum similarity range
  sampling_strategy: "top" # How to select negatives
```

### Training Parameters
```yaml
# Advanced training settings
training:
  gradient_checkpointing: false
  fp16: false
  bf16: true
  eval_steps: 10
  save_steps: 10
  logging_steps: 10
  warmup_ratio: 0.1
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```yaml
batch_size: 64  # Reduce from 128
```

**Slow Mining**
```yaml
mining_model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

**Poor Results**
```yaml
# Increase data or epochs
sample_size: 2000
num_epochs: 10

# Use better mining model
mining_model_name: "sentence-transformers/static-retrieval-mrl-en-v1"
```

## Commands

```bash
make train-embed              # Basic training
make train-embed-tb          # With TensorBoard
make eval-embed               # Evaluate model
make infer-embed              # Run inference
```

---

**Embedding Fine-Tuning**: Production-ready contrastive learning with hard negative mining for superior semantic representations.
