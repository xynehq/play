# Xyne-LLM-Play Examples

Complete end-to-end examples for different use cases, models, and hardware configurations.

## Available Examples

### Quick Start Examples
- **[Basic SFT](basic-sft/)** - Your first fine-tuning project
- **[Customer Support Bot](customer-support/)** - Domain-specific assistant
- **[Multi-GPU Training](multi-gpu/)** - Large model training

### Advanced Examples
- **[Medical Domain Adaptation](medical-dapt/)** - Specialized knowledge training
- **[Code Generation](code-generation/)** - Programming assistant
- **[Multi-Turn Conversations](multi-turn/)** - Chat model training

## Example Structure

Each example follows this structure:
```
example_name/
├── README.md           # Step-by-step instructions
├── config.yaml         # Optimized configuration
├── data/              # Sample data (if applicable)
├── scripts/           # Custom scripts (if needed)
└── expected_results/  # What to expect
```

## Getting Started

### Choose Your Example
1. **Beginner**: Start with [Basic SFT](basic-sft/)
2. **Specific Use Case**: Choose from domain examples
3. **Large Models**: Use [Multi-GPU](multi-gpu/) examples

### Run an Example
```bash
# Navigate to example
cd examples/basic-sft

# Follow the README instructions
cat README.md

# Run the training
make train CONFIG=config.yaml
```

## Hardware Compatibility

| Example | Min GPU | Recommended GPU | Model Size |
|---------|---------|-----------------|------------|
| Basic SFT | RTX 4060 (8GB) | RTX 4090 (24GB) | 1.5B-7B |
| Customer Support | RTX 4070 (12GB) | RTX 4090 (24GB) | 3B-14B |
| Medical DAPT | RTX 4080 (16GB) | A100 (40GB) | 7B-27B |
| Multi-GPU | 2x RTX 4090 | 2x H100 | 27B-70B |

## Customization

### Adapt Examples to Your Needs
```bash
# Copy an example
cp -r examples/basic-sft examples/my-project

# Customize the configuration
nano examples/my-project/config.yaml

# Add your data
cp my_data.jsonl examples/my-project/data/

# Run your custom example
cd examples/my-project
make train CONFIG=config.yaml
```

### Parameter Overrides
```bash
# Quick customization without editing files
make train CONFIG=config.yaml \
  MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
  TRAIN_EPOCHS=5 \
  TRAIN_LEARNING_RATE=1e-4
```

## Best Practices

### For Beginners
1. Start with [Basic SFT](basic-sft/)
2. Use sample data first
3. Monitor with TensorBoard
4. Experiment with parameters

### For Production
1. Use appropriate hardware configurations
2. Validate data quality
3. Monitor training metrics
4. Test thoroughly before deployment

### For Research
1. Experiment with different models
2. Compare configurations
3. Document results
4. Share findings

## Troubleshooting

### Common Issues
- **Memory errors**: Reduce batch size or model size
- **Slow training**: Use Unsloth backend or multiple GPUs
- **Poor quality**: Increase epochs or improve data
- **Configuration errors**: Validate with `make check`

### Get Help
```bash
# Check your setup
make check

# Get help with commands
make help

# Validate configuration
python scripts/validate_config.py --config config.yaml
```

## Contributing Examples

### Share Your Examples
We welcome community contributions! To add an example:

1. Create a new folder in `examples/`
2. Follow the standard structure
3. Include clear documentation
4. Test on multiple hardware configurations
5. Submit a pull request

### Example Template
```bash
mkdir examples/my-example
cd examples/my-example

# Create README with step-by-step instructions
# Create optimized config.yaml
# Add sample data if needed
# Test thoroughly
```

## Next Steps

- **[Documentation](../docs/)** - Detailed guides and references
- **[Configuration Guide](../docs/configuration/)** - Configuration options
- **[Supervised Fine-Tuning](../docs/sft/)** - SFT techniques
- **[Domain Adaptation](../docs/dapt/)** - DAPT methods
- **[Multi-GPU Training](../docs/multi-gpu/)** - Distributed training

---

**Start with [Basic SFT](basic-sft/) to get hands-on experience! 🚀**
