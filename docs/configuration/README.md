# Configuration Guide

Complete guide to configuring Xyne-LLM-Play for different models, hardware, and training scenarios.

## What Configuration Does

Configuration files tell Xyne-LLM-Play how to train your model. Think of them as recipes that specify:
- Which model to use and how to load it
- How to tune the model (LoRA, QLoRA, etc.)
- Training parameters (learning rate, batch size, etc.)
- Data processing and evaluation settings

## Configuration Structure

### Basic Configuration File
```yaml
# configs/example_config.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  max_seq_len: 512
  trust_remote_code: true

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: "auto"

train:
  epochs: 3
  learning_rate: 2e-4
  warmup_ratio: 0.06
  weight_decay: 0.01
  batch_size: "auto"
  grad_accum: "auto"
  bf16: true
  gradient_checkpointing: true

data:
  train_path: "data/processed/train.jsonl"
  val_path: "data/processed/val.jsonl"
```

## Configuration Sections

### Model Configuration
```yaml
model:
  name: "model_name"              # HuggingFace model ID or local path
  max_seq_len: 2048               # Maximum sequence length
  trust_remote_code: true         # Trust remote model code
  torch_dtype: "auto"            # Auto-detect data type
  device_map: "auto"             # Auto-detect device placement
```

### Tuning Configuration
```yaml
tuning:
  mode: "qlora"                   # "qlora", "lora", or "full"
  backend: "bnb"                  # "bnb" or "unsloth"
  
  lora:
    r: 32                        # LoRA attention dimension
    alpha: 64                     # LoRA scaling parameter
    dropout: 0.1                  # LoRA dropout
    target_modules: "auto"        # LoRA target modules
    bias: "none"                  # Bias training
    task_type: "CAUSAL_LM"        # Task type
```

### Training Configuration
```yaml
train:
  # Core training parameters
  epochs: 3                      # Number of training epochs
  learning_rate: 2e-4            # Learning rate
  warmup_ratio: 0.06             # Warmup ratio
  weight_decay: 0.01             # Weight decay
  
  # Batch and memory optimization
  batch_size: "auto"             # Per-device batch size
  grad_accum: "auto"             # Gradient accumulation steps
  target_tokens_per_device_step: 16384  # Target tokens per step
  
  # Precision and optimization
  bf16: true                     # Use bfloat16 precision
  fp16: false                    # Use float16 precision
  gradient_checkpointing: true    # Enable gradient checkpointing
  
  # Evaluation parameters
  eval_strategy: "steps"         # "no", "steps", or "epoch"
  eval_steps: 200                # Steps between evaluations
  predict_with_generate: true    # Enable generation-based evaluation
  generation_num_beams: 1        # Number of beams for generation
  generation_max_new_tokens: 64  # Maximum new tokens to generate
  
  # Saving and loading
  save_strategy: "epoch"         # "no", "steps", or "epoch"
  save_steps: 200                # Steps between saves
  save_total_limit: 5            # Maximum number of checkpoints
  load_best_model_at_end: false  # Load best model at end
  
  # Distributed training
  ddp_find_unused_parameters: false
  ddp_broadcast_buffers: false
  ddp_bucket_cap_mb: 200
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  
  # Logging and monitoring
  logging_steps: 1               # Steps between logging
  report_to: ["tensorboard"]     # Reporting destinations
```

### Data Configuration
```yaml
data:
  train_path: "data/train.jsonl"  # Training data path
  val_path: "data/val.jsonl"      # Validation data path
  test_path: "data/test.jsonl"    # Test data path (optional)
  
  # Data processing
  max_train_samples: null         # Limit training samples
  max_eval_samples: null          # Limit evaluation samples
  shuffle_train: true             # Shuffle training data
  seed: 42                        # Random seed
```

## Pre-made Configurations

### Single GPU Configurations

#### Small Models (1-7B)
```yaml
# configs/run_bnb.yaml
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
  grad_accum: 4
  bf16: true
  gradient_checkpointing: true
```

#### Fast Training (Unsloth)
```yaml
# configs/run_unsloth.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  max_seq_len: 512

tuning:
  mode: "qlora"
  backend: "unsloth"
  lora:
    r: 16
    alpha: 32
    dropout: 0.1

train:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 8
  grad_accum: 2
  bf16: true
  gradient_checkpointing: true
```

### Multi-GPU Configurations

#### Large Models (27B+)
```yaml
# configs/run_distributed_v1.yaml
model:
  name: "google/gemma-3-27b-it"
  max_seq_len: 2048

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 32
    alpha: 64
    dropout: 0.1

train:
  epochs: 3
  learning_rate: 1e-4
  batch_size: "auto"
  grad_accum: "auto"
  bf16: true
  gradient_checkpointing: true
  
  # Multi-GPU specific
  ddp_find_unused_parameters: false
  ddp_broadcast_buffers: false
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  
  # Evaluation optimization
  predict_with_generate: true
  generation_num_beams: 1
  generation_max_new_tokens: 64
  eval_strategy: "steps"
  eval_steps: 200
  eval_do_concat_batches: false
```

### Domain Adaptation Configurations

#### DAPT Training
```yaml
# configs/run_dapt.yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_seq_len: 2048

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 32
    alpha: 64
    dropout: 0.05

train:
  epochs: 5
  learning_rate: 1e-4
  batch_size: 2
  grad_accum: 8
  bf16: true
  gradient_checkpointing: true
  
  # DAPT-specific
  task_mode: "cpt_mixed"         # Mixed CPT + instruction training
  cpt_weight: 0.7                # Weight for document training
  sft_weight: 0.3                # Weight for instruction training
```

## Custom Configuration Creation

### Step 1: Choose Base Configuration
Start with the closest pre-made configuration:
- `configs/run_bnb.yaml` for single GPU
- `configs/run_distributed_v1.yaml` for multi-GPU
- `configs/run_dapt.yaml` for domain adaptation

### Step 2: Copy and Customize
```bash
# Copy base configuration
cp configs/run_bnb.yaml configs/my_config.yaml

# Edit your configuration
nano configs/my_config.yaml
```

### Step 3: Key Customizations

#### Model Selection
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"  # Change model
  max_seq_len: 1024                 # Adjust for memory
```

#### Memory Optimization
```yaml
train:
  batch_size: 2                    # Reduce for memory
  grad_accum: 8                    # Increase for effective batch
  gradient_checkpointing: true     # Enable for memory savings
```

#### Quality vs Speed Trade-offs
```yaml
# For better quality
train:
  epochs: 5                        # More training
  learning_rate: 1e-4              # Lower learning rate
  batch_size: 4                    # Larger batch

# For faster training
train:
  epochs: 1                        # Less training
  learning_rate: 5e-4              # Higher learning rate
  eval_steps: 500                  # Less frequent evaluation
```

## Parameter Overrides

### Command Line Overrides
Override specific parameters without editing config files:

```bash
# Override learning rate and epochs
make train CONFIG=configs/run_bnb.yaml TRAIN_LEARNING_RATE=1e-4 TRAIN_EPOCHS=5

# Override model and batch size
make train CONFIG=configs/run_bnb.yaml MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" TRAIN_BATCH_SIZE=2

# Override multiple parameters
make train CONFIG=configs/run_bnb.yaml \
  MODEL_MAX_SEQ_LEN=1024 \
  TRAIN_BATCH_SIZE=2 \
  TRAIN_GRAD_ACCUM=8 \
  TRAIN_LEARNING_RATE=1e-4
```

### Override Patterns
```bash
# Memory-constrained overrides
make train CONFIG=configs/run_bnb.yaml \
  MODEL_MAX_SEQ_LEN=256 \
  TRAIN_BATCH_SIZE=1 \
  TRAIN_GRAD_ACCUM=16

# Quality-focused overrides
make train CONFIG=configs/run_bnb.yaml \
  TRAIN_EPOCHS=5 \
  TRAIN_LEARNING_RATE=1e-4 \
  TRAIN_WARMUP_RATIO=0.1

# Speed-focused overrides
make train CONFIG=configs/run_bnb.yaml \
  TRAIN_EPOCHS=1 \
  TRAIN_EVAL_STEPS=500 \
  TRAIN_SAVE_STEPS=1000
```

## Hardware-Specific Configurations

### RTX 4060 (8GB)
```yaml
# configs/rtx4060.yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  max_seq_len: 256

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 8
    alpha: 16
    dropout: 0.1

train:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 1
  grad_accum: 16
  bf16: true
  gradient_checkpointing: true
```

### RTX 4090 (24GB)
```yaml
# configs/rtx4090.yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
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
  grad_accum: 4
  bf16: true
  gradient_checkpointing: true
```

### 2x H100 (160GB)
```yaml
# configs/h100_pair.yaml
model:
  name: "google/gemma-3-27b-it"
  max_seq_len: 2048

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 32
    alpha: 64
    dropout: 0.1

train:
  epochs: 3
  learning_rate: 1e-4
  batch_size: "auto"
  grad_accum: "auto"
  bf16: true
  gradient_checkpointing: true
  
  # Multi-GPU settings
  ddp_find_unused_parameters: false
  dataloader_num_workers: 8
  dataloader_pin_memory: true
```

## Advanced Configuration

### LoRA Target Modules
```yaml
tuning:
  lora:
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]  # Specific modules
    # or
    target_modules: "auto"  # Auto-detect best modules
```

### Learning Rate Scheduling
```yaml
train:
  learning_rate: 2e-4
  warmup_ratio: 0.06              # Warmup for 6% of training
  lr_scheduler_type: "cosine"     # Cosine annealing
  # Options: "linear", "cosine", "polynomial", "constant"
```

### Evaluation Configuration
```yaml
train:
  # Generation-based evaluation
  predict_with_generate: true
  generation_num_beams: 1
  generation_max_new_tokens: 64
  generation_min_new_tokens: 1
  generation_early_stopping: true
  
  # Evaluation strategy
  eval_strategy: "steps"
  eval_steps: 200
  eval_accumulation_steps: 8
  eval_do_concat_batches: false
  
  # Metrics
  compute_metrics: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
```

### DeepSpeed Configuration
```yaml
train:
  deepspeed: "configs/deepspeed_z2.json"  # DeepSpeed config file
```

```json
// configs/deepspeed_z2.json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": "auto"
    }
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0
}
```

## Configuration Validation

### Check Configuration
```bash
# Validate configuration file
python scripts/validate_config.py --config configs/my_config.yaml

# Check hardware compatibility
make check

# Test configuration with dry run
make train CONFIG=configs/my_config.yaml DRY_RUN=true
```

### Common Configuration Issues

#### Memory Issues
```yaml
# Reduce memory usage
train:
  batch_size: 1
  grad_accum: 32
  max_seq_len: 256
  gradient_checkpointing: true
```

#### Quality Issues
```yaml
# Improve quality
train:
  epochs: 5
  learning_rate: 1e-4
  warmup_ratio: 0.1
  weight_decay: 0.1
```

#### Speed Issues
```yaml
# Improve speed
train:
  eval_steps: 500
  save_steps: 1000
  logging_steps: 10
  dataloader_num_workers: 8
```

## Next Steps

- **[Supervised Fine-Tuning](../sft/)** - SFT configuration examples
- **[Domain Adaptation](../dapt/)** - DAPT configuration examples
- **[Multi-GPU Training](../multi-gpu/)** - Multi-GPU configuration
- **[Examples](../../examples/)** - Complete configuration examples
