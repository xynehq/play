# Xyne-Play Documentation

Welcome to the comprehensive documentation for Xyne-Play, the universal LLM fine-tuning CLI tool.

## 📚 Documentation Structure

### 🎯 Task-Specific Guides
- **[LLM Fine-Tuning](tasks/llm-fine-tuning.md)** - Instruction following and chat model training
- **[DAPT/CPT](tasks/dapt-cpt.md)** - Domain-adaptive pretraining and continual pretraining
- **[Embedding Fine-Tuning](tasks/embedding-fine-tuning.md)** - Semantic embedding training with contrastive learning

### 🚀 Advanced Guides
- **[Multi-GPU Training](advanced/multi-gpu.md)** - Distributed training across multiple GPUs
- **[Custom Configurations](advanced/custom-configs.md)** - Advanced configuration patterns
- **[Production Deployment](advanced/production.md)** - Deploying models to production

### ⚙️ Reference
- **[Configuration Reference](configs/)** - Complete configuration options
- **[Command Reference](commands.md)** - All CLI commands and Makefile targets
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🎯 Getting Started

### New to Xyne-Play?
1. Read the [main README](../README.md) for installation and quick start
2. Follow the [LLM Fine-Tuning guide](tasks/llm-fine-tuning.md) for your first model
3. Explore other task guides as needed

### Choose Your Task

#### I want to train a chat assistant
→ **[LLM Fine-Tuning](tasks/llm-fine-tuning.md)**

#### I want to create a domain expert model
→ **[DAPT/CPT](tasks/dapt-cpt.md)**

#### I want to improve semantic search
→ **[Embedding Fine-Tuning](tasks/embedding-fine-tuning.md)**

#### I have multiple GPUs and want to train large models
→ **[Multi-GPU Training](advanced/multi-gpu.md)**

## 🏗️ Hardware Requirements

| Task | Minimum GPU | Recommended GPU | VRAM Required |
|------|-------------|-----------------|---------------|
| LLM Fine-Tuning (1-7B) | RTX 4060 (8GB) | RTX 4070 (12GB) | 8-12GB |
| LLM Fine-Tuning (14B+) | RTX 4080 (16GB) | RTX 4090 (24GB) | 16-24GB |
| DAPT/CPT | RTX 4060 (8GB) | RTX 4070 (12GB) | 8-12GB |
| Embedding Fine-Tuning | RTX 3050 (6GB) | RTX 4060 (8GB) | 6-8GB |
| Multi-GPU Training | 2x RTX 4060 | 2x RTX 4090 | 16GB+ per GPU |

## 🎛️ Quick Command Reference

### Setup
```bash
make install          # Install dependencies
make setup-dirs       # Create directories
make check           # Validate setup
```

### Training
```bash
make train-bnb-tb    # Single GPU with monitoring
make train-distributed-tb  # Multi-GPU training
make train-embed-tb  # Embedding fine-tuning
```

### Evaluation & Inference
```bash
make eval            # Evaluate model
make infer           # Interactive chat
make tensorboard     # View training logs
```

## 📖 Learning Path

### Beginner Path
1. **Installation**: `make install && make setup-dirs`
2. **First Model**: Follow [LLM Fine-Tuning](tasks/llm-fine-tuning.md) with toy dataset
3. **Evaluation**: `make eval` and `make infer`
4. **Monitoring**: `make tensorboard`

### Intermediate Path
1. **Custom Data**: Create your own dataset
2. **Configuration**: Learn [Custom Configurations](advanced/custom-configs.md)
3. **Domain Adaptation**: Try [DAPT/CPT](tasks/dapt-cpt.md)
4. **Embedding Training**: Explore [Embedding Fine-Tuning](tasks/embedding-fine-tuning.md)

### Advanced Path
1. **Multi-GPU**: Set up [Multi-GPU Training](advanced/multi-gpu.md)
2. **Production**: Deploy with [Production Guide](advanced/production.md)
3. **Optimization**: Advanced configuration and tuning
4. **Customization**: Extend Xyne-Play for your needs

## 🔧 Common Workflows

### Research Workflow
```bash
# 1. Setup experiment
make setup-dirs
cp configs/config_base.yaml configs/experiment.yaml

# 2. Prepare data
make process && make style

# 3. Run experiment
make train CONFIG=configs/experiment.yaml

# 4. Evaluate results
make eval && make infer

# 5. Analyze with TensorBoard
make tensorboard
```

### Production Workflow
```bash
# 1. Multi-GPU training
make setup-accelerate
make train-deepspeed-tb CONFIG=configs/production.yaml

# 2. Merge for deployment
make merge

# 3. Test deployment
make merge-test

# 4. Deploy model
# (See production guide)
```

### Domain Expert Workflow
```bash
# 1. Prepare domain documents
make dapt-docx

# 2. Domain pretraining
make dapt-train

# 3. Add instruction following
make process && make train-bnb-tb

# 4. Test domain knowledge
make infer
```

## 📊 Performance Tips

### Memory Optimization
- Use QLoRA for memory-constrained training
- Reduce sequence length: `model.max_seq_len=256`
- Smaller batch sizes: `train.batch_size=1`

### Speed Optimization
- Use Unsloth backend: `make train-unsloth-tb`
- Multi-GPU training: `make train-distributed-tb`
- Gradient accumulation: `train.grad_accum=16`

### Quality Optimization
- More epochs: `train.epochs=5`
- Better learning rate: `train.learning_rate=1e-4`
- Larger datasets: Add more training examples

## 🚨 Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or sequence length
- **Slow training**: Use Unsloth or multi-GPU
- **Poor results**: Increase epochs or improve data quality
- **Setup issues**: Run `make check` to validate

### Get Help
- **Built-in help**: `make help`
- **Setup validation**: `make check`
- **GPU info**: `make gpu-info`
- **Memory check**: `make memory-check`

## 🤝 Contributing

### Documentation Contributions
- Fix typos and errors
- Add examples and use cases
- Improve explanations
- Update for new features

### How to Contribute
1. Fork the repository
2. Make your changes
3. Test your changes
4. Submit a pull request

## 📈 Roadmap

### Upcoming Features
- [ ] More model architectures support
- [ ] Advanced evaluation metrics
- [ ] Production deployment tools
- [ ] Web interface
- [ ] Cloud integration

### Documentation Plans
- [ ] Video tutorials
- [ ] Interactive examples
- [ ] API reference
- [ ] Best practices guide

---

**Need help?** Start with the [main README](../README.md) or choose a task-specific guide from the list above.
