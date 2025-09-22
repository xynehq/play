# Energy Logging and Performance Measurement Guide

## Overview

This guide explains the comprehensive energy logging and performance measurement system implemented in SFT-Play. The system provides detailed metrics for comparing the efficiency of different training backends (BitsAndBytes vs Unsloth) and optimizing training performance.

## Features

### 1. **Energy Monitoring**
- **GPU Power Consumption**: Real-time power usage monitoring via NVML
- **Energy Integration**: Calculates total energy consumption in Joules
- **Warmup Period**: Excludes initial warmup period from energy calculations
- **Peak/Average Power**: Tracks both peak and average power consumption

### 2. **Performance Metrics**
- **Token Throughput**: Accurate tokens/second calculation
- **VRAM Usage**: Peak and average VRAM consumption tracking
- **Wall Clock Time**: Total training time measurement
- **Energy Efficiency**: Joules per token and energy per 10K tokens

### 3. **Automated Logging**
- **CSV Export**: All metrics automatically saved to CSV for analysis
- **Run Metadata**: Captures model, backend, optimizer, and hyperparameter details
- **Comparative Analysis**: Easy comparison between different configurations

## Configuration

### Enable Energy Logging

Energy logging is controlled via the `metrics` section in your config files:

```yaml
metrics:
  enable_energy_logging: true
  sample_interval_s: 0.05       # 50ms sampling rate
  warmup_s: 60.0               # 60s warmup period
  log_vram_every_s: 1.0        # VRAM sampling every 1s
  results_csv: "outputs/energy_results.csv"
```

### Backend Configurations

The system includes optimized configurations for fair comparison:

**BitsAndBytes (configs/run_bnb.yaml):**
```yaml
tuning:
  backend: bnb
  lora:
    r: 32
    alpha: 32          # Matched with Unsloth
    dropout: 0.05

train:
  bf16: false
  fp16: true
  optim: paged_adamw_8bit  # BnB optimized optimizer
```

**Unsloth (configs/run_unsloth.yaml):**
```yaml
tuning:
  backend: unsloth
  lora:
    r: 32              # Matched with BnB
    alpha: 32
    dropout: 0.05

train:
  bf16: true           # Unsloth prefers bfloat16
  fp16: false
  optim: adamw_torch   # Standard optimizer for Unsloth
```

## Usage

### Quick Start

1. **Run BitsAndBytes measurement:**
   ```bash
   make measure-bnb
   ```

2. **Run Unsloth measurement:**
   ```bash
   make measure-unsloth
   ```

3. **View results:**
   ```bash
   cat outputs/energy_results.csv
   ```

### Manual Training with Metrics

```bash
# BitsAndBytes with energy logging
python scripts/train.py --config configs/run_bnb.yaml

# Unsloth with energy logging
python scripts/train.py --config configs/run_unsloth.yaml
```

## System Architecture

### Components

1. **NVML Utilities** (`metrics/nvml_utils.py`)
   - GPU power and memory monitoring
   - Multi-threaded sampling for accurate measurements
   - Robust error handling for NVML operations

2. **Token Counter** (`metrics/token_counter.py`)
   - Accurate token counting in data collator
   - Handles packed sequences correctly
   - Thread-safe implementation

3. **Energy Logger** (`scripts/train.py`)
   - Integrates power measurements over time
   - Calculates efficiency metrics
   - Exports results to CSV

### Data Flow

```
Training Data → Collator (counts tokens) → Model Training
                    ↓
GPU Power Monitor → Energy Integration → Metrics Calculation
                    ↓
CSV Export ← Run Metadata ← Performance Stats
```

## Metrics Explained

### Core Metrics

| Metric | Description | Units |
|--------|-------------|-------|
| `tokens_processed` | Total tokens processed during training | count |
| `wall_s` | Total wall clock time | seconds |
| `warmup_s` | Warmup period excluded from calculations | seconds |
| `energy_J` | Total energy consumed (post-warmup) | Joules |
| `avg_power_W` | Average power consumption | Watts |
| `peak_power_W` | Peak power consumption | Watts |
| `tok_per_s` | Token throughput | tokens/second |
| `J_per_token` | Energy efficiency | Joules/token |
| `E10k_J` | Energy per 10K tokens | Joules |
| `vram_peak_MB` | Peak VRAM usage | MB |
| `vram_avg_MB` | Average VRAM usage | MB |

### Run Metadata

| Field | Description |
|-------|-------------|
| `model` | Model name/path |
| `dataset` | Dataset type (sft/cpt/cpt_mixed) |
| `backend` | Training backend (bnb/unsloth) |
| `optimizer` | Optimizer used |
| `batch_size` | Per-device batch size |
| `grad_accum` | Gradient accumulation steps |
| `seq_len` | Maximum sequence length |
| `precision` | Training precision (fp16/bf16) |
| `seed` | Random seed |
| `notes` | LoRA configuration details |

## Analysis Examples

### Comparing Backends

```python
import pandas as pd

# Load results
df = pd.read_csv('outputs/energy_results.csv')

# Compare efficiency
bnb_runs = df[df['backend'] == 'bnb']
unsloth_runs = df[df['backend'] == 'unsloth']

print(f"BnB avg efficiency: {bnb_runs['J_per_token'].mean():.3f} J/token")
print(f"Unsloth avg efficiency: {unsloth_runs['J_per_token'].mean():.3f} J/token")
print(f"Speedup: {bnb_runs['tok_per_s'].mean() / unsloth_runs['tok_per_s'].mean():.2f}x")
```

### Energy Efficiency Analysis

```python
# Calculate energy savings
energy_savings = (bnb_runs['E10k_J'].mean() - unsloth_runs['E10k_J'].mean()) / bnb_runs['E10k_J'].mean() * 100
print(f"Energy savings with Unsloth: {energy_savings:.1f}%")

# VRAM efficiency
vram_savings = (bnb_runs['vram_peak_MB'].mean() - unsloth_runs['vram_peak_MB'].mean()) / bnb_runs['vram_peak_MB'].mean() * 100
print(f"VRAM savings with Unsloth: {vram_savings:.1f}%")
```

## Requirements

### Dependencies

The energy logging system requires:

```bash
# Core dependencies (already in requirements.txt)
pip install pynvml torch transformers

# Optional for analysis
pip install pandas matplotlib seaborn
```

### Hardware Requirements

- **NVIDIA GPU**: Required for power monitoring via NVML
- **CUDA Support**: GPU must support CUDA for power measurements
- **Driver Version**: Recent NVIDIA drivers (450.80.02+)

### System Compatibility

- **Linux**: Fully supported
- **Windows**: Supported with NVIDIA drivers
- **macOS**: Limited support (no NVML on Apple Silicon)

## Troubleshooting

### Common Issues

1. **NVML Not Available**
   ```
   [train] Warning: Could not import metrics: No module named 'pynvml'
   ```
   **Solution**: Install pynvml: `pip install pynvml`

2. **GPU Not Detected**
   ```
   [train] Warning: Could not import metrics: NVML initialization failed
   ```
   **Solution**: Check NVIDIA drivers and CUDA installation

3. **Permission Denied**
   ```
   [train] Warning: Could not import metrics: Insufficient permissions
   ```
   **Solution**: Run with appropriate permissions or check GPU access

### Fallback Behavior

When energy logging fails:
- Training continues normally without metrics
- Only basic timing and token counting is performed
- CSV export is skipped
- No impact on training performance or results

## Best Practices

### Measurement Guidelines

1. **Consistent Environment**
   - Use same GPU and system configuration
   - Ensure consistent thermal conditions
   - Close unnecessary applications

2. **Multiple Runs**
   - Run each configuration multiple times
   - Average results for statistical significance
   - Account for variance in measurements

3. **Fair Comparison**
   - Use identical datasets and hyperparameters
   - Ensure same sequence lengths and batch sizes
   - Match LoRA configurations between backends

### Optimization Tips

1. **Sampling Rate**
   - 50ms (0.05s) provides good balance of accuracy vs overhead
   - Increase for longer training runs
   - Decrease for very short experiments

2. **Warmup Period**
   - 60s warmup excludes model loading and compilation
   - Adjust based on your model size and complexity
   - Longer warmup for larger models

3. **VRAM Monitoring**
   - 1s interval sufficient for most use cases
   - Increase frequency for memory-sensitive experiments
   - Monitor for memory leaks during long runs

## Integration with Existing Workflows

### TensorBoard Integration

Energy metrics complement TensorBoard logging:

```bash
# Start training with both energy logging and TensorBoard
make train-bnb-tb    # BitsAndBytes with TensorBoard
make train-unsloth-tb # Unsloth with TensorBoard
```

### Automated Pipelines

Integrate energy measurement into CI/CD:

```bash
# Example workflow
make process          # Prepare data
make measure-bnb      # Measure BnB performance
make measure-unsloth  # Measure Unsloth performance
python analyze_results.py  # Custom analysis script
```

### Batch Experiments

For systematic comparison:

```bash
# Run multiple configurations
for config in configs/run_*.yaml; do
    echo "Testing $config"
    python scripts/train.py --config $config
done

# Analyze all results
python scripts/analyze_energy.py outputs/energy_results.csv
```

## Future Enhancements

### Planned Features

1. **Real-time Monitoring Dashboard**
2. **Automated Report Generation**
3. **Integration with MLflow/Weights & Biases**
4. **Multi-GPU Support**
5. **Carbon Footprint Calculation**

### Extensibility

The system is designed for easy extension:

- Add new metrics in `train_with_metrics()`
- Extend CSV schema for additional data
- Create custom analysis scripts
- Integrate with external monitoring tools

## Conclusion

The energy logging system provides comprehensive insights into training efficiency, enabling data-driven decisions about backend selection and optimization. By measuring both performance and energy consumption, you can optimize for the best balance of speed, efficiency, and resource utilization.

For questions or issues, refer to the troubleshooting section or check the implementation in `metrics/` and `scripts/train.py`.
