# Accelerate Setup Guide for Multi-GPU Training

## Overview

This guide provides step-by-step instructions for configuring Accelerate to enable multi-GPU distributed training in the SFT-Play repository. Accelerate is a PyTorch library that simplifies distributed training across multiple GPUs and nodes.

## Prerequisites

Before setting up Accelerate, ensure you have:
- Multiple GPUs available on your system
- SFT-Play repository installed and configured
- Virtual environment activated 
- Dependencies installed (`make install`)

## Quick Setup Command

```bash
make setup-accelerate
```

This command launches the interactive Accelerate configuration wizard.

## Detailed Configuration Process

When you run `make setup-accelerate`, you'll be prompted with several configuration options. Below is the complete configuration flow with explanations for each choice:

### Step 1: Compute Environment
```
In which compute environment are you running?
This machine
```

**Explanation**: 
- **This machine**: Select this when running on a single machine with multiple GPUs
- **Other options**: For cloud environments or cluster setups

### Step 2: Machine Type
```
Which type of machine are you using?
multi-GPU
```

**Explanation**:
- **multi-GPU**: Choose this when you have multiple GPUs on a single machine
- **Other options**: For CPU-only, TPU, or single GPU setups

### Step 3: Number of Machines
```
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
```

**Explanation**:
- **1**: For single-node multi-GPU training (most common setup)
- **>1**: For multi-node training across multiple machines

### Step 4: Distributed Operations Error Checking
```
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: NO
```

**Explanation**:
- **NO**: Recommended for better performance during training
- **yes**: Enables additional error checking at the cost of speed

### Step 5: Torch Dynamo Optimization
```
Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
```

**Explanation**:
- **NO**: Torch Dynamo is experimental and may cause compatibility issues
- **yes**: For potential performance improvements (advanced users)

### Step 6: DeepSpeed Integration
```
Do you want to use DeepSpeed? [yes/NO]: no
```

**Explanation**:
- **no**: For standard distributed training without DeepSpeed
- **yes**: If you want to use DeepSpeed optimizations (requires additional setup)

### Step 7: FullyShardedDataParallel
```
Do you want to use FullyShardedDataParallel? [yes/NO]: NO
```

**Explanation**:
- **NO**: Standard data parallelism is sufficient for most use cases
- **yes**: For advanced memory optimization with very large models

### Step 8: Megatron-LM
```
Do you want to use Megatron-LM ? [yes/NO]: NO
```

**Explanation**:
- **NO**: Megatron-LM is for specialized large-scale training
- **yes**: For enterprise-level massive model training

### Step 9: Number of GPUs
```
How many GPU(s) should be used for distributed training? [1]: 2
```

**Explanation**:
- **2**: Use 2 GPUs (adjust based on your available GPUs)
- **Common values**: 2, 4, 8 (depending on your hardware)

### Step 10: GPU Selection
```
What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]: all
```

**Explanation**:
- **all**: Use all available GPUs
- **Specific IDs**: e.g., "0,1" for specific GPUs (use `nvidia-smi` to see GPU IDs)

### Step 11: NUMA Efficiency
```
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
```

**Explanation**:
- **NO**: NUMA optimization is advanced and may not be necessary
- **yes**: For potential performance improvements on NUMA architectures

### Step 12: Mixed Precision
```
Do you wish to use mixed precision?
bf16
```

**Explanation**:
- **bf16**: Brain Floating Point 16 - recommended for modern GPUs (RTX 40xx series, A100, H100)
- **fp16**: Floating Point 16 - for older GPUs
- **no**: For full precision training (uses more memory)

## Configuration File Location

After completing the setup, Accelerate saves your configuration to:
```
/root/.cache/huggingface/accelerate/default_config.yaml
```

## Verifying Your Configuration

### Check Configuration File
```bash
cat /root/.cache/huggingface/accelerate/default_config.yaml
```

### Test Multi-GPU Detection
```bash
make gpu-info
```

### Validate Accelerate Setup
```bash
accelerate env
```

## Usage Examples

### Basic Multi-GPU Training
```bash
make train-distributed
```

### Multi-GPU Training with Monitoring
```bash
make train-distributed-tb
```

### DeepSpeed Multi-GPU Training
```bash
make train-deepspeed-tb
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use QLoRA instead of full fine-tuning
   - Enable gradient checkpointing

2. **GPU Not Detected**
   - Verify GPU visibility: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`
   - Ensure PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Distributed Training Fails**
   - Re-run `make setup-accelerate`
   - Check configuration file permissions
   - Verify all GPUs are accessible

4. **Performance Issues**
   - Ensure proper GPU affinity
   - Check for NUMA alignment
   - Monitor GPU utilization with `nvidia-smi`

### Reset Configuration

If you need to reconfigure Accelerate:

```bash
# Remove existing configuration
rm /root/.cache/huggingface/accelerate/default_config.yaml

# Re-run setup
make setup-accelerate
```

## Advanced Configuration

### Custom Configuration File

For project-specific configurations:

```bash
# Create custom config
accelerate config --config_file accelerate_custom.yaml

# Use custom config
accelerate launch --config_file accelerate_custom.yaml scripts/train_distributed.py --config configs/run_distributed.yaml
```

### Environment Variables

Key environment variables for distributed training:

```bash
# Set number of GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Set master address and port (for multi-node)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500

# Set world size
export WORLD_SIZE=2
```

## Performance Optimization Tips

### 1. Batch Size Optimization
- Start with small batch sizes and increase gradually
- Use `auto` batch size in configuration for automatic optimization
- Monitor GPU memory usage during training
- The system now automatically calculates optimal batch sizes based on:
  - Model size and parameter count
  - Available VRAM per GPU
  - Sequence length requirements
  - Memory safety margins

### 2. Memory Management
- Enable gradient checkpointing for large models
- Use mixed precision (bf16/fp16) to reduce memory usage
- Consider QLoRA for very large models
- The system automatically handles device placement with Accelerate
- `low_cpu_mem_usage=True` is enabled for better CPU memory management

### 3. Data Loading
- Ensure fast data storage (SSD recommended)
- Use multiple data workers if available
- Pre-process data when possible

### 4. Monitoring
- Use TensorBoard for real-time monitoring
- Monitor GPU utilization and memory usage
- Track training metrics and loss curves

## Resource-Aware Auto-Scaling

The distributed training system now includes intelligent resource awareness:

### Automatic Resource Detection
- GPU count and model detection
- VRAM per GPU calculation
- Model size estimation based on parameter count
- Safety margin calculations to prevent OOM errors

### Smart Batch Size Calculation
- Dynamic batch size adjustment based on available resources
- Gradient accumulation scaling for optimal effective batch sizes
- Model-specific optimizations (small vs. large models)
- Hardware-aware clamping to prevent excessive values

### Memory-Efficient Loading
- `low_cpu_mem_usage=True` for reduced CPU memory footprint
- Accelerate handles device placement automatically
- Optimized memory per sample calculations
- Safety margins to prevent memory overflow

## Configuration Reference

### Recommended Configurations

#### **2 GPU Setup (RTX 4090)**
```yaml
compute_environment: local_machine
distributed_type: multi_gpu
mixed_precision: bf16
num_processes: 2
```

#### **4 GPU Setup (A100)**
```yaml
compute_environment: local_machine
distributed_type: multi_gpu
mixed_precision: bf16
num_processes: 4
```

#### **8 GPU Setup (H100)**
```yaml
compute_environment: local_machine
distributed_type: multi_gpu
mixed_precision: bf16
num_processes: 8
```

## Integration with SFT-Play

### Makefile Integration

The SFT-Play Makefile includes several multi-GPU commands:

- `make train-distributed`: Basic multi-GPU training
- `make train-distributed-tb`: Multi-GPU with TensorBoard
- `make train-deepspeed`: DeepSpeed multi-GPU training
- `make train-deepspeed-tb`: DeepSpeed with TensorBoard

### Configuration Files

SFT-Play provides pre-configured files for different scenarios:

- `configs/run_distributed.yaml`: Basic multi-GPU configuration
- `configs/run_gemma27b_distributed.yaml`: Large model multi-GPU setup
- `configs/deepspeed_z2.json`: DeepSpeed ZeRO configuration

## Best Practices

1. **Before Training**
   - Always run `make check` to validate setup
   - Verify GPU availability with `make gpu-info`
   - Test with a small dataset first

2. **During Training**
   - Monitor GPU utilization and memory usage
   - Watch for training stability
   - Use TensorBoard for real-time metrics

3. **After Training**
   - Evaluate model performance
   - Test merged model loading
   - Backup trained adapters and configurations

## Support and Resources

### Getting Help
```bash
make help                    # Show all commands
make check                   # Validate setup
accelerate test              # Test Accelerate installation
```

### Documentation
- [Accelerate Official Documentation](https://huggingface.co/docs/accelerate)
- [SFT-Play README](README.md)
- [Multi-GPU Training Guide](MULTI_GPU_GUIDE.md)

### Community
- Report issues on GitHub
- Check existing discussions and solutions
- Share your multi-GPU training results

---

**Note**: This guide is based on the configuration process for SFT-Play v2.0. Configuration options may vary with different versions of Accelerate or PyTorch.
