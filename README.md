# LLM-Workbench
A universal training suite for SFT, DAPT, multi-GPU, embeddings finetuning, and RL. Batteries included (configs, eval, TensorBoard).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Universal LLM training suite: SFT, DAPT, multi-GPU, embeddings finetuning, and RL—ready for research & production.**  
Scale from RTX 4060 to H200 clusters. Train any model, any size, anywhere.

## What is Xyne-LLM-Play?

Xyne-LLM-Play is a universal command-line tool that makes LLM fine-tuning accessible to everyone:

- **🏠 Hobbyists**: Fine-tune 7B models on RTX 4060 (8GB)
- **🎓 Researchers**: Experiment with 13B-20B models on RTX 4090 (24GB)  
- **🏢 Enterprises**: Train 70B+ models on H100/H200 clusters
- **🚀 Everyone**: One tool, any hardware, any model size

## Installation

```bash
# Clone and install
git clone <repository-url>
cd play
make install && make setup-dirs

# Quick start with sample data
./workflows/quick_start.sh
```

## Usage Patterns

### Command Line Interface
```bash
# Single GPU training
make train-bnb-tb

# Multi-GPU training
make train-distributed-tb

# Domain adaptation
make dapt-docx && make dapt-train
```

### Python API
```python
from scripts.train import main
from scripts.infer import main as infer_main

# Train model
main(config="configs/run_bnb.yaml")

# Run inference
infer_main(config="configs/run_bnb.yaml", mode="interactive")
```

## Example: Fine-Tuning LLM

```bash
# 1. Prepare your data
echo '{"user": "What is machine learning?", "assistant": "Machine learning is..."}' > data/raw/my_data.jsonl

# 2. Process and train
make process
make train-bnb-tb

# 3. Chat with your model
make infer
```

## Find More Examples

- **[Supervised Fine-Tuning](docs/sft/)** - Instruction following and chat capabilities
- **[Domain Adaptation](docs/dapt/)** - Specialized domain knowledge training  
- **[Multi-GPU Training](docs/multi-gpu/)** - Distributed training across multiple GPUs
- **[Configuration Guide](docs/configuration/)** - Custom configs and parameter overrides
- **[Examples](examples/)** - Complete end-to-end examples

## Hardware Support

| Your Hardware | VRAM | Recommended Models | Training Mode |
|---------------|------|-------------------|---------------|
| **RTX 4060** | 8GB | Qwen2.5-1.5B, Gemma-2B | QLoRA |
| **RTX 4090** | 24GB | Qwen2.5-14B, Llama-13B | QLoRA/LoRA |
| **2x H200** | 282GB | Any model | DeepSpeed |

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

**From 8GB to 800GB. From hobbyist to enterprise. One tool, infinite possibilities. 🚀**
