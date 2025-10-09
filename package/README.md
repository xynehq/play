# SFT-Play: Universal LLM Fine-Tuning CLI

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Scale from RTX 4060 to H200 clusters. Train any model, any size, anywhere.**

## 🚀 Quick Install

### Minimal (Core only - cross-platform)
```bash
pip install sft-play
```

### Standard (with GPU support & monitoring)
```bash
pip install sft-play[standard]
```

### Full (all features)
```bash
pip install sft-play[full]
```

## 📦 Installation Options

Choose the installation that fits your needs:

| Install Command | Includes | Best For |
|----------------|----------|----------|
| `pip install sft-play` | Core training deps | Mac/Linux, learning |
| `pip install sft-play[standard]` | + GPU, eval, monitoring | Single GPU training |
| `pip install sft-play[gpu]` | + BitsAndBytes (QLoRA) | Memory-efficient training |
| `pip install sft-play[distributed]` | + DeepSpeed | Multi-GPU clusters |
| `pip install sft-play[unsloth]` | + Unsloth backend | Fastest training |
| `pip install sft-play[dapt]` | + Document processing | Domain adaptation |
| `pip install sft-play[full]` | Everything | Production use |

## 🎯 What You Get

### Core Package (Minimal)
- ✅ Cross-platform (Mac & Linux)
- ✅ Single-GPU training with LoRA/QLoRA
- ✅ Hugging Face Transformers integration
- ✅ CLI tools: `sft-play`, `sft-train`, `sft-eval`, `sft-infer`
- ✅ Configuration-driven workflows

### Optional Features
- **[gpu]**: BitsAndBytes for 4-bit/8-bit quantization (Linux)
- **[unsloth]**: 2x faster training with Unsloth backend
- **[distributed]**: Multi-GPU with DeepSpeed ZeRO
- **[dapt]**: Domain-Adaptive Pretraining from documents
- **[evaluation]**: ROUGE, BLEU, and custom metrics
- **[monitoring]**: TensorBoard integration
- **[wandb]**: Weights & Biases experiment tracking

## 💻 Quick Start

### 1. Install
```bash
pip install sft-play[standard]
```

### 2. Create Config
```yaml
# config.yaml
model_name: "Qwen/Qwen2.5-1.5B"
dataset: "tatsu-lab/alpaca"
output_dir: "./outputs"
lora_r: 16
lora_alpha: 32
num_train_epochs: 3
```

### 3. Train
```bash
sft-play train --config config.yaml
```

### 4. Chat with Your Model
```bash
sft-play infer --model ./outputs/checkpoints/final
```

## 🎛️ CLI Commands

### Main Interface
```bash
sft-play --help                    # Show all commands
sft-play --version                 # Check version
```

### Training
```bash
sft-play train --config config.yaml
sft-train --config config.yaml     # Direct alias
```

### Evaluation
```bash
sft-play eval --model ./checkpoints/final
sft-eval --model ./checkpoints/final
```

### Inference
```bash
sft-play infer --model ./checkpoints/final
sft-play infer --model ./checkpoints/final --prompt "Hello, how are you?"
sft-infer --model ./checkpoints/final
```

### Data Processing
```bash
sft-play process --input data/raw --output data/processed
sft-process --input data/raw --output data/processed
```

## 🏗️ Hardware Support

| Hardware | Memory | Install | Training Mode |
|----------|--------|---------|---------------|
| **RTX 4060** | 8GB | `[standard]` | QLoRA |
| **RTX 4070** | 12GB | `[standard]` | QLoRA |
| **RTX 4080** | 16GB | `[standard,unsloth]` | QLoRA/LoRA |
| **RTX 4090** | 24GB | `[standard,unsloth]` | QLoRA/LoRA |
| **A100** | 40/80GB | `[distributed]` | QLoRA/LoRA/Full |
| **H100/H200** | 80/141GB | `[full]` | Any |

## 📚 Documentation

- **Full Documentation**: [GitHub Repository](https://github.com/xynehq/sft-play)
- **Getting Started Guide**: [GETTING_STARTED.md](https://github.com/xynehq/sft-play/blob/main/GETTING_STARTED.md)
- **Multi-GPU Guide**: [MULTI_GPU_GUIDE.md](https://github.com/xynehq/sft-play/blob/main/MULTI_GPU_GUIDE.md)
- **Changelog**: [CHANGELOG.md](https://github.com/xynehq/sft-play/blob/main/CHANGELOG.md)

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/xynehq/sft-play/issues)
- **Discussions**: [GitHub Discussions](https://github.com/xynehq/sft-play/discussions)

## 📄 License

Apache License 2.0 - See [LICENSE](../LICENSE) for details.

## 🙏 Credits

Built by the Xyne community. Powered by:
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [Unsloth](https://github.com/unslothai/unsloth)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

---

**From 8GB to 800GB. From hobbyist to enterprise. One tool, infinite possibilities.** 🚀
