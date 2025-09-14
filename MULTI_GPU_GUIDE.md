# Multi-GPU Training Guide for SFT-Play

This guide explains how to scale SFT-Play training across multiple GPUs for large models like Gemma 27B, Qwen 32B, and Llama 70B.

## 🎯 **Quick Start for Your 2x H200 Setup**

### **Problem**: Gemma3 27B hitting CUDA OOM on single GPU
### **Solution**: Distributed training across your 2 H200s

```bash
# 1. Setup (one-time)
make setup-accelerate          # Configure Accelerate for multi-GPU
make gpu-info                  # Verify your 2x H200s are detected

# 2. Train Gemma 27B across 2 GPUs
make train-distributed-tb CONFIG=configs/run_gemma27b_distributed.yaml

# 3. Alternative: Use DeepSpeed for even better memory efficiency
make train-deepspeed-tb CONFIG=configs/run_gemma27b_distributed.yaml
```

## 🏗️ **Architecture Overview**

SFT-Play now supports three scaling approaches:

1. **Standard Distributed Training**: Uses Accelerate + PyTorch DDP
2. **DeepSpeed Integration**: ZeRO optimizer states for maximum memory efficiency  
3. **Automatic Optimization**: Smart batch sizing and memory management

## 📋 **Prerequisites**

### **Required Dependencies**
```bash
pip install accelerate deepspeed
# or if using uv:
uv pip install accelerate deepspeed
```

### **Hardware Requirements**
- **Minimum**: 2x GPUs with 16GB+ VRAM each
- **Recommended**: 2x H100/H200 for 27B+ models
- **Network**: High-bandwidth interconnect (NVLink preferred)

## ⚙️ **Configuration**

### **1. Accelerate Setup (One-time)**
```bash
make setup-accelerate
```

This will prompt you to configure:
- **Compute environment**: Multi-GPU
- **Machine type**: No distributed training (for single-node)
- **Number of processes**: 2 (for your 2 H200s)
- **GPU IDs**: 0,1
- **Mixed precision**: bf16 (optimal for H200)

### **2. Model Configuration**

Any existing config works with distributed training. Key parameters:

```yaml
# configs/your_model_config.yaml
model:
  name: google/gemma-3-27b-it    # Any model
  max_seq_len: 2048              # Adjust based on memory
  
tuning:
  mode: qlora                    # Recommended for large models
  backend: bnb                   # Stable for distributed training
  
train:
  batch_size: auto               # Auto-calculated per GPU
  grad_accum: auto               # Auto-calculated for efficiency
  bf16: true                     # Optimal for H200
  gradient_checkpointing: true   # Essential for memory efficiency
```

## 🚀 **Training Commands**

### **Standard Distributed Training**
```bash
# Basic distributed training
make train-distributed CONFIG=configs/your_config.yaml

# With TensorBoard monitoring
make train-distributed-tb CONFIG=configs/your_config.yaml

# Examples for different models:
make train-distributed CONFIG=configs/run_bnb.yaml              # Small models
make train-distributed CONFIG=configs/run_gemma27b_distributed.yaml  # Large models
make train-distributed CONFIG=configs/run_dapt.yaml            # DAPT training
```

### **DeepSpeed Training (Maximum Memory Efficiency)**
```bash
# DeepSpeed with ZeRO-2
make train-deepspeed CONFIG=configs/your_config.yaml

# DeepSpeed with TensorBoard
make train-deepspeed-tb CONFIG=configs/your_config.yaml
```

### **All Training Modes Supported**
- **SFT**: `make train-distributed CONFIG=configs/run_bnb.yaml`
- **DAPT**: `make train-distributed CONFIG=configs/run_dapt.yaml`
- **Mixed CPT**: Any config with `task_mode: cpt_mixed`
- **Any Backend**: BitsAndBytes (recommended) or Unsloth

## 🧠 **Memory Management**

### **Automatic Batch Size Calculation**

The system automatically calculates optimal batch sizes:

```python
# For Gemma 27B on 2x H200 (282GB total):
# - Model: ~13.5GB per GPU (QLoRA)
# - Available: ~127GB per GPU for batches
# - Auto-calculated: batch_size=4, grad_accum=4
# - Effective batch size: 32 (4 * 2 GPUs * 4 accum)
```

### **Memory Optimization Features**

1. **QLoRA**: 4-bit quantization reduces model memory by ~75%
2. **Gradient Checkpointing**: Trades compute for memory
3. **Smart Device Mapping**: Optimal model placement across GPUs
4. **DeepSpeed ZeRO**: Shards optimizer states across GPUs

### **Memory Troubleshooting**

If you still hit OOM:

```bash
# 1. Check current memory usage
make memory-check

# 2. Reduce sequence length
# Edit your config: max_seq_len: 1024 (instead of 2048)

# 3. Force smaller batch size
# Edit your config: batch_size: 1, grad_accum: 32

# 4. Use DeepSpeed for maximum efficiency
make train-deepspeed CONFIG=your_config.yaml
```

## 📊 **Performance Expectations**

### **Gemma 27B on 2x H200**
- **Memory Usage**: ~130GB per GPU (with QLoRA)
- **Training Speed**: ~2x faster than single GPU
- **Effective Batch Size**: 32-64 (auto-optimized)
- **Context Length**: Up to 4096 tokens

### **Scaling Efficiency**
- **2 GPUs**: ~1.8x speedup (communication overhead)
- **Memory**: Linear scaling (2x total VRAM)
- **Throughput**: Near-linear for large models

## 🔧 **Advanced Configuration**

### **Custom DeepSpeed Config**

Create custom DeepSpeed configurations:

```json
// configs/deepspeed_custom.json
{
  "zero_optimization": {
    "stage": 3,                    // More aggressive memory saving
    "cpu_offload": true           // Offload to CPU memory
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": "auto"
}
```

Use with:
```bash
make train-deepspeed CONFIG=your_config.yaml DEEPSPEED_CONFIG=configs/deepspeed_custom.json
```

### **Multi-Node Training**

For training across multiple machines:

```bash
# On each node, configure accelerate with:
accelerate config
# Select: Multi-node distributed training
# Specify: main node IP, number of nodes, etc.

# Then run on all nodes:
make train-distributed CONFIG=your_config.yaml
```

## 🛠️ **Monitoring & Debugging**

### **GPU Monitoring**
```bash
# Check GPU status
make gpu-info

# Monitor memory usage during training
make memory-check

# Watch GPU utilization
watch -n 1 nvidia-smi
```

### **TensorBoard Monitoring**
```bash
# All distributed training commands support TensorBoard
make train-distributed-tb CONFIG=your_config.yaml

# View at: http://localhost:6006
# Metrics include:
# - Loss per GPU and aggregated
# - Memory usage
# - Training throughput
```

### **Common Issues & Solutions**

**1. "NCCL timeout" errors:**
```bash
export NCCL_TIMEOUT=1800  # Increase timeout
export NCCL_DEBUG=INFO    # Enable debug logging
```

**2. "CUDA out of memory" on multi-GPU:**
```bash
# Reduce batch size in config
batch_size: 1
grad_accum: 64
```

**3. "Process group not initialized":**
```bash
# Reconfigure accelerate
make setup-accelerate
```

## 📈 **Scaling Examples**

### **Small Models (7B-13B)**
```bash
# Single GPU sufficient, but distributed for speed
make train-distributed CONFIG=configs/run_bnb.yaml
# Expected: 2x speedup, minimal memory benefit
```

### **Medium Models (20B-30B)**
```bash
# Distributed recommended for memory
make train-distributed CONFIG=configs/run_gemma27b_distributed.yaml
# Expected: Fits in memory, 1.8x speedup
```

### **Large Models (70B+)**
```bash
# DeepSpeed required for memory
make train-deepspeed CONFIG=configs/run_llama70b.yaml
# Expected: Only way to fit in 2x H200
```

## 🎯 **Best Practices**

### **For Your 2x H200 Setup**

1. **Use bf16**: Optimal for H200 architecture
2. **Enable gradient checkpointing**: Essential for large models
3. **Start with QLoRA**: Best memory/quality tradeoff
4. **Monitor memory**: Use `make memory-check` during training
5. **Use TensorBoard**: Monitor both GPUs with `-tb` commands

### **Configuration Recommendations**

```yaml
# Optimal config for 2x H200 + large models
train:
  bf16: true                     # H200 optimized
  gradient_checkpointing: true   # Memory efficiency
  batch_size: auto               # Let system optimize
  grad_accum: auto               # Let system optimize
  dataloader_num_workers: 4      # Parallel data loading
  dataloader_pin_memory: false   # Avoid multi-GPU issues
```

## 🔄 **Migration from Single GPU**

### **Existing Configs Work As-Is**

No changes needed to existing configurations:

```bash
# This works for both single and multi-GPU:
make train CONFIG=configs/run_bnb.yaml           # Single GPU
make train-distributed CONFIG=configs/run_bnb.yaml  # Multi-GPU
```

### **Gradual Migration**

1. **Test with small model**: Verify multi-GPU setup works
2. **Scale up gradually**: Try medium models before large ones
3. **Monitor memory**: Ensure you're not hitting limits
4. **Compare performance**: Verify speedup vs single GPU

## 🎉 **Success Validation**

### **Verify Multi-GPU Training Works**

```bash
# 1. Check GPU detection
make gpu-info
# Should show: "Number of GPUs: 2"

# 2. Start training with monitoring
make train-distributed-tb CONFIG=configs/run_bnb.yaml

# 3. Check TensorBoard
# Visit: http://localhost:6006
# Should show: metrics from both GPUs

# 4. Monitor memory usage
make memory-check
# Should show: both GPUs being utilized
```

### **Expected Output**
```
[train] Detected 2 GPUs:
  GPU 0: NVIDIA H200 (141.0 GB)
  GPU 1: NVIDIA H200 (141.0 GB)
[train] Total VRAM: 282.0 GB
[train] Auto-calculated batch size: 4 per GPU, gradient accumulation: 4
[train] Effective batch size: 32
[train] Starting new bnb distributed run with bf16 precision on 2 GPUs
```

## 🚀 **Ready to Scale!**

Your 2x H200 setup is now ready to train large models efficiently. Start with:

```bash
make train-distributed-tb CONFIG=configs/run_gemma27b_distributed.yaml
```

This will train Gemma 27B across both H200s with automatic memory optimization and live monitoring.
