# Xyne-Play: Universal LLM Fine-Tuning CLI

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.42+-orange.svg)](https://huggingface.co/transformers/)

**The Universal Command-Line Tool for LLM Fine-Tuning**  
Scale from RTX 4060 to H200 clusters. Train any model, any size, anywhere.

---

## 🎯 What is Xyne-Play?

Xyne-Play is a production-ready CLI tool that makes LLM fine-tuning accessible to everyone. Whether you're a hobbyist with a consumer GPU or an enterprise with GPU clusters, Xyne-Play provides one unified interface for:

- **🏠 Hobbyists**: Fine-tune 7B models on RTX 4060 (8GB)
- **🎓 Researchers**: Experiment with 13B-20B models on RTX 4090 (24GB)  
- **🏢 Enterprises**: Train 70B+ models on H100/H200 clusters
- **🚀 Everyone**: One tool, any hardware, any model size

---

## 🚀 Installation

### Quick Install
```bash
git clone <repository-url>
cd xyne-play
make install && make setup-dirs
```

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **CUDA**: 11.8+ for GPU training
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ for models and data

---

## 💡 Usage Patterns

### Command Line Interface (Recommended)
```bash
# Interactive setup with sample data
./workflows/quick_start.sh

# Start training with monitoring
make train-bnb-tb

# Chat with your model
make infer
```

### Python API
```python
from xyne_play import XyneTrainer

trainer = XyneTrainer(config="configs/run_bnb.yaml")
trainer.train()
```

---

## 🎯 Example 1: Fine-Tuning LLM

### Dataset Format
Create `data/raw/my_data.jsonl`:
```json
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is..."}
{"system": "You are a helpful assistant.", "user": "Explain neural networks.", "assistant": "Neural networks are..."}
```

### Training Command
```bash
# Process data and start training
make process && make train-bnb-tb

# Interactive chat with trained model
make infer
```

### Custom Configuration
```bash
# Use custom config
make train CONFIG=configs/my_config.yaml

# Override specific parameters
make train CONFIG=configs/run_bnb.yaml model.name=Qwen/Qwen2.5-7B-Instruct
```

---

## 📚 Find More Examples

- **[📋 Documentation](docs/)** - Complete guides and tutorials
- **[🎯 Task-Specific Examples](docs/tasks/)** - SFT, DAPT, Embedding FT, and more
- **[🚀 Advanced Guides](docs/advanced/)** - Multi-GPU, DeepSpeed, production deployment
- **[🔧 Configuration Reference](docs/configs/)** - All configuration options

---

## 🏗️ Hardware Support

| Hardware | VRAM | Max Model Size | Training Mode |
|----------|------|----------------|---------------|
| RTX 4060 | 8GB | 7B | QLoRA |
| RTX 4090 | 24GB | 20B | QLoRA/LoRA |
| A100 40GB | 40GB | 35B | QLoRA/LoRA |
| 2x H200 | 282GB | 70B+ | DeepSpeed |

---

## 🎛️ Quick Commands

```bash
make help           # See all available commands
make gpu-info       # Check your hardware
make check          # Validate setup
make train-bnb-tb   # Start training with monitoring
make infer          # Chat with your model
```

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

**Xyne-Play**: From 8GB to 800GB. From hobbyist to enterprise. One tool, infinite possibilities. 🚀
