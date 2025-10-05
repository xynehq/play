# Multi-GPU Training

Scale SFT-Play training across multiple GPUs for large models like Gemma 27B, Qwen 32B, and Llama 70B with intelligent memory management and optimized evaluation.

## What Multi-GPU Training Does

Multi-GPU training enables you to:
- **Train larger models**: Scale beyond single GPU memory limits
- **Speed up training**: Near-linear speedup with multiple GPUs
- **Handle bigger datasets**: Process more data in parallel
- **Achieve better performance**: Larger batch sizes and more stable training

## Expected Setup

### Hardware Requirements
- **Minimum**: 2x GPUs with 16GB+ VRAM each
- **Recommended**: 2x H100/H200 for 27B+ models
- **Network**: High-bandwidth interconnect (NVLink preferred)

### Software Setup
```bash
# Install dependencies with multi-GPU support
make install

# Configure Accelerate for multi-GPU (one-time setup)
make setup-accelerate

# Verify GPU detection
make gpu-info
```

## Training Command on Toy Dataset

### Quick Start with Multi-GPU
```bash
# 1. Setup multi-GPU environment
make setup-accelerate

# 2. Start distributed training with monitoring
make train-distributed-tb CONFIG=configs/run_bnb.yaml

# 3. Monitor progress at http://localhost:6006
```

### Step-by-Step Multi-GPU Training
```bash
# 1. Check your GPU setup
make gpu-info
# Should show: "Number of GPUs: 2+"

# 2. Configure Accelerate (one-time)
make setup-accelerate
# Select: Multi-GPU, your GPU count, bf16 precision

# 3. Start distributed training
make train-distributed-tb CONFIG=configs/run_bnb.yaml

# 4. Monitor both GPUs
make memory-check
# Visit: http://localhost:6006
```

### Training Commands
```bash
# Standard distributed training
make train-distributed CONFIG=configs/your_config.yaml

# With TensorBoard monitoring
make train-distributed-tb CONFIG=configs/your_config.yaml

# DeepSpeed for maximum memory efficiency
make train-deepspeed CONFIG=configs/your_config.yaml
make train-deepspeed-tb CONFIG=configs/your_config.yaml
```

## Custom Configuration and Overrides

### Multi-GPU Optimized Configuration
Create a multi-GPU specific configuration:

```yaml
# configs/multi_gpu_config.yaml
model:
  name: "google/gemma-3-27b-it"    # Large model for multi-GPU
  max_seq_len: 2048                # Adjust based on memory
  
tuning:
  mode: "qlora"                    # Recommended for large models
  backend: "bnb"                   # Stable for distributed training
  lora:
    r: 32                          # Higher rank for large models
    alpha: 64
    dropout: 0.1
  
train:
  # Multi-GPU specific settings
  batch_size: "auto"               # Auto-calculated per GPU
  grad_accum: "auto"               # Auto-calculated for efficiency
  bf16: true                       # Optimal for modern GPUs
  gradient_checkpointing: true     # Essential for memory efficiency
  
  # Distributed training settings
  ddp_find_unused_parameters: false
  ddp_broadcast_buffers: false
  ddp_bucket_cap_mb: 200
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  
  # Evaluation optimization
  predict_with_generate: true      # Enable generation-based eval
  generation_num_beams: 1          # Fast beam search
  generation_max_new_tokens: 64    # Reasonable length
  eval_strategy: "steps"           # Regular evaluation
  eval_steps: 200                 # Evaluation frequency
  eval_do_concat_batches: false    # Reduce memory
```

### Override Multi-GPU Parameters
```bash
# Override batch size for memory constraints
make train-distributed CONFIG=configs/run_bnb.yaml TRAIN_BATCH_SIZE=1 TRAIN_GRAD_ACCUM=32

# Override learning rate for multi-GPU
make train-distributed CONFIG=configs/run_bnb.yaml TRAIN_LEARNING_RATE=1e-4

# Force specific GPU count
make train-distributed CONFIG=configs/run_bnb.yaml NUM_GPUS=2
```

### DeepSpeed Configuration
Create custom DeepSpeed configurations:

```json
// configs/deepspeed_custom.json
{
  "zero_optimization": {
    "stage": 3,                    // More aggressive memory saving
    "cpu_offload": true           // Offload to CPU memory
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto"
}
```

Use with:
```bash
make train-deepspeed CONFIG=your_config.yaml DEEPSPEED_CONFIG=configs/deepspeed_custom.json
```

## Advanced Multi-GPU Techniques

### Memory Optimization Strategies
```bash
# For memory-constrained training
make train-deepspeed CONFIG=configs/run_bnb.yaml \
  TRAIN_BATCH_SIZE=1 \
  TRAIN_GRAD_ACCUM=64 \
  MODEL_MAX_SEQ_LEN=1024

# For speed-optimized training
make train-distributed CONFIG=configs/run_bnb.yaml \
  TRAIN_BATCH_SIZE=8 \
  TRAIN_GRAD_ACCUM=4 \
  MODEL_MAX_SEQ_LEN=2048
```

### Model Size Scaling
```yaml
# For 2x RTX 4090 (48GB total)
model:
  name: "Qwen/Qwen2.5-14B-Instruct"
train:
  batch_size: 4
  grad_accum: 4

# For 2x H100 (160GB total)
model:
  name: "google/gemma-3-27b-it"
train:
  batch_size: 8
  grad_accum: 4

# For 4x H200 (564GB total)
model:
  name: "meta-llama/Llama-3.1-70B-Instruct"
train:
  batch_size: 16
  grad_accum: 2
```

### Evaluation Optimization
```yaml
# Fast evaluation for multi-GPU
train:
  predict_with_generate: true
  generation_num_beams: 1          # Single beam for speed
  generation_max_new_tokens: 64    # Reasonable length
  eval_strategy: "steps"
  eval_steps: 200
  eval_do_concat_batches: false    # Reduce memory
  eval_accumulation_steps: 8       # Accumulation for memory
```

## Hardware-Specific Configurations

### RTX 4090 SLI (2x 24GB)
```yaml
# configs/rtx4090_sli.yaml
model:
  name: "Qwen/Qwen2.5-14B-Instruct"
  max_seq_len: 1536
  
train:
  batch_size: 6                   # Per GPU
  grad_accum: 4
  bf16: true
  gradient_checkpointing: true
```

### A100 Pair (2x 80GB)
```yaml
# configs/a100_pair.yaml
model:
  name: "google/gemma-3-27b-it"
  max_seq_len: 2048
  
train:
  batch_size: 8                   # Per GPU
  grad_accum: 4
  bf16: true
  gradient_checkpointing: true
```

### H200 Quad (4x 141GB)
```yaml
# configs/h200_quad.yaml
model:
  name: "meta-llama/Llama-3.1-70B-Instruct"
  max_seq_len: 4096
  
train:
  batch_size: 16                  # Per GPU
  grad_accum: 2
  bf16: true
  gradient_checkpointing: true
```

## Performance Monitoring

### GPU Monitoring
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Memory usage check
make memory-check

# Detailed GPU information
make gpu-info
```

### Training Metrics
```bash
# Start TensorBoard with multi-GPU training
make train-distributed-tb CONFIG=configs/your_config.yaml

# Monitor at: http://localhost:6006
# Look for:
# - Loss per GPU and aggregated
# - GPU utilization across all devices
# - Memory usage per GPU
# - Training throughput
```

### Performance Benchmarks
| Model | GPUs | Memory/GPU | Batch Size | Speed (tokens/sec) |
|-------|------|------------|------------|-------------------|
| Qwen2.5-7B | 2x RTX 4090 | ~18GB | 6 | ~1200 |
| Gemma-27B | 2x H100 | ~65GB | 8 | ~800 |
| Llama-70B | 4x H200 | ~120GB | 16 | ~600 |

## Troubleshooting

### Common Multi-GPU Issues

#### "NCCL timeout" errors
```bash
# Increase NCCL timeout
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

# Retry training
make train-distributed CONFIG=configs/your_config.yaml
```

#### "CUDA out of memory" on multi-GPU
```bash
# Reduce batch size
make train-distributed CONFIG=configs/your_config.yaml \
  TRAIN_BATCH_SIZE=1 TRAIN_GRAD_ACCUM=32

# Use DeepSpeed for maximum efficiency
make train-deepspeed CONFIG=configs/your_config.yaml
```

#### "Process group not initialized"
```bash
# Reconfigure Accelerate
make setup-accelerate

# Check GPU detection
make gpu-info
```

#### Uneven GPU utilization
```bash
# Check GPU connectivity
nvidia-smi topo -m

# Optimize data loading
make train-distributed CONFIG=configs/your_config.yaml \
  TRAIN_DATALOADER_NUM_WORKERS=8
```

### Performance Optimization

#### Slow Training Speed
```bash
# Increase batch size if memory allows
make train-distributed CONFIG=configs/your_config.yaml \
  TRAIN_BATCH_SIZE=8 TRAIN_GRAD_ACCUM=2

# Use Unsloth backend for faster training
make train-distributed CONFIG=configs/run_unsloth.yaml

# Optimize data loading
make train-distributed CONFIG=configs/your_config.yaml \
  TRAIN_DATALOADER_NUM_WORKERS=8 TRAIN_DATALOADER_PIN_MEMORY=true
```

#### Memory Optimization
```bash
# Enable gradient checkpointing
make train-distributed CONFIG=configs/your_config.yaml \
  TRAIN_GRADIENT_CHECKPOINTING=true

# Reduce sequence length
make train-distributed CONFIG=configs/your_config.yaml \
  MODEL_MAX_SEQ_LEN=1024

# Use DeepSpeed ZeRO
make train-deepspeed CONFIG=configs/your_config.yaml
```

## Advanced Features

### Automatic Batch Size Calculation
The system automatically calculates optimal batch sizes:

```python
# For Gemma 27B on 2x H200 (282GB total):
# - Model: ~13.5GB per GPU (QLoRA)
# - Available: ~127GB per GPU for batches
# - Auto-calculated: batch_size=4, grad_accum=4
# - Effective batch size: 32 (4 * 2 GPUs * 4 accum)
```

### Smart Evaluation System
- **Automatic Compatibility Detection**: Detects Transformers version capabilities
- **Generation-Based Evaluation**: ROUGE-L scoring when supported
- **Loss-Only Evaluation**: 300x faster fallback when generation unsupported
- **EvalSpeedCallback**: Intelligent caching and optimization

### Backend Stamping
Prevents cross-backend resume conflicts:
```bash
# Each training run gets a unique backend stamp
# Prevents loading BitsAndBytes checkpoints with Unsloth
# Ensures clean training state management
```

## Multi-Node Training

For training across multiple machines:

```bash
# On each node, configure accelerate
accelerate config
# Select: Multi-node distributed training
# Specify: main node IP, number of nodes, etc.

# Then run on all nodes
make train-distributed CONFIG=your_config.yaml
```

## Next Steps

- **[Supervised Fine-Tuning](../sft/)** - Single-GPU training basics
- **[Domain Adaptation](../dapt/)** - Multi-GPU domain training
- **[Configuration Guide](../configuration/)** - Advanced configuration options
- **[Examples](../../examples/)** - Complete multi-GPU examples
