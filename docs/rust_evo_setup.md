# RustEvo VM Integration Guide

This guide explains how to set up and run the RustEvo benchmark in the VM environment.

## Overview

RustEvo is a benchmark for evaluating code generation models on Rust code evolution tasks. This integration provides a clean, VM-optimized setup that:

- Clones the original RustEvo repository
- Applies VM-specific configurations
- Provides convenient execution scripts
- Manages resources and dependencies

## Quick Start

### 1. Setup RustEvo

```bash
# Run the setup script
./scripts/setup_rust_evo.sh
```

This will:
- Clone the original RustEvo repository
- Copy essential files to `Benchmark/RustEvo/`
- Apply VM-specific configurations
- Clean up temporary files

### 2. Verify Installation

```bash
# Check if RustEvo is properly installed
python scripts/run_rust_evo.py --check
```

### 3. List Available Scripts

```bash
# See all available evaluation and analysis scripts
python scripts/run_rust_evo.py --list
```

### 4. Run Benchmarks

```bash
# Run evaluation script
python scripts/run_rust_evo.py --eval eval_models.py

# Run analysis script
python scripts/run_rust_evo.py --analysis generate_code.py
```

## Directory Structure

After setup, the RustEvo integration creates the following structure:

```
Benchmark/RustEvo/
├── README.md              # Original RustEvo documentation
├── requirements.txt       # Python dependencies
├── vm_config.yaml        # VM-specific configuration
├── .gitignore            # Git ignore rules
├── Dataset/              # Benchmark datasets
│   ├── APIDocs.json
│   └── RustEvo^2.json
├── Evaluate/             # Evaluation scripts
│   ├── eval_models.py
│   ├── eval_models_rq1.py
│   ├── eval_RAG_models.py
│   └── ...
├── Scripts/              # Analysis and utility scripts
│   ├── generate_code.py
│   ├── crate_analyzer.py
│   └── ...
└── Imgs/                 # Documentation images
    ├── overview.png
    └── ...
```

## Configuration

### VM Configuration

The VM configuration is stored in `configs/rust_evo_config.yaml` and includes:

- **Resource Limits**: Memory, CPU, timeout settings
- **Model Configuration**: Default models and parameters
- **Paths**: Directory locations for datasets and results
- **Security**: Sandbox and access restrictions

### Runtime Configuration

Individual benchmark runs can be configured using environment variables:

```bash
export RUSTEVO_MAX_MEMORY_GB=16
export RUSTEVO_MAX_CPU_CORES=8
export RUSTEVO_TIMEOUT_SECONDS=600
export RUSTEVO_LOG_LEVEL=DEBUG
```

## Available Benchmarks

### Evaluation Scripts

Located in `Benchmark/RustEvo/Evaluate/`:

- `eval_models.py` - Main model evaluation
- `eval_models_rq1.py` - Research Question 1 evaluation
- `eval_models_rq3.py` - Research Question 3 evaluation
- `eval_RAG_models.py` - RAG-enhanced model evaluation
- `eval_RAG_rq4.py` - RAG Research Question 4 evaluation

### Analysis Scripts

Located in `Benchmark/RustEvo/Scripts/`:

- `generate_code.py` - Code generation analysis
- `generate_test.py` - Test generation analysis
- `crate_analyzer.py` - Crate analysis tools
- `rust_api_analyzer.py` - Rust API analysis
- `compare_stable_api.py` - Stable API comparison

## Usage Examples

### Basic Evaluation

```bash
# Run main evaluation with default settings
python scripts/run_rust_evo.py --eval eval_models.py

# Run with custom arguments
python scripts/run_rust_evo.py --eval eval_models.py --args --model gpt-4 --verbose
```

### Analysis Tasks

```bash
# Generate code analysis
python scripts/run_rust_evo.py --analysis generate_code.py

# Analyze Rust crates
python scripts/run_rust_evo.py --analysis crate_analyzer.py --args --crate serde
```

### Batch Processing

```bash
# Run multiple evaluations
for script in eval_models.py eval_models_rq1.py eval_RAG_models.py; do
    python scripts/run_rust_evo.py --eval "$script"
done
```

## Troubleshooting

### Common Issues

1. **Setup Fails**
   ```bash
   # Clean up and retry
   rm -rf Benchmark/RustEvo
   ./scripts/setup_rust_evo.sh
   ```

2. **Script Not Found**
   ```bash
   # Verify installation
   python scripts/run_rust_evo.py --check
   python scripts/run_rust_evo.py --list
   ```

3. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x scripts/setup_rust_evo.sh
   chmod +x Benchmark/RustEvo/Scripts/*.py
   chmod +x Benchmark/RustEvo/Evaluate/*.py
   ```

4. **Dependency Issues**
   ```bash
   # Install requirements
   pip install -r Benchmark/RustEvo/requirements.txt
   ```

### Logs and Debugging

Enable verbose logging:

```bash
export RUSTEVO_LOG_LEVEL=DEBUG
python scripts/run_rust_evo.py --eval eval_models.py
```

Check log files in `Benchmark/RustEvo/Results/` for detailed output.

## Integration with VM Environment

### Resource Management

The integration automatically manages VM resources:

- **Memory**: Limited to configured maximum
- **CPU**: Restricts concurrent processes
- **Timeout**: Prevents infinite loops
- **Disk**: Monitors temporary file usage

### Security

- **Sandbox**: Scripts run in restricted environment
- **Network**: Disabled by default
- **File Access**: Limited to RustEvo directory
- **Process Isolation**: Separate process for each benchmark

## Contributing

To modify the integration:

1. **Setup Script**: Edit `scripts/setup_rust_evo.sh`
2. **Configuration**: Modify `configs/rust_evo_config.yaml`
3. **Wrapper**: Update `scripts/run_rust_evo.py`
4. **Documentation**: Update this file

## License

This integration maintains the original RustEvo MIT license. The original repository is at: https://github.com/rust-ml/RustEvo

## Support

For issues with:
- **RustEvo Benchmark**: Check the original repository
- **VM Integration**: Create an issue in this repository
- **Setup Problems**: Check the troubleshooting section above
