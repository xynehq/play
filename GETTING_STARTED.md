# Getting Started with SFT-Play: Complete Beginner's Guide

Welcome to SFT-Play! This guide will walk you through everything you need to know to start fine-tuning language models, from small 1.5B models on consumer GPUs to massive 70B+ models on enterprise hardware.

## 🤔 **What is SFT-Play?**

SFT-Play is a **command-line tool** for fine-tuning large language models. Think of it as your Swiss Army knife for AI model training:

- **Supervised Fine-Tuning (SFT)**: Train models to follow instructions better
- **Domain-Adaptive Pretraining (DAPT)**: Teach models about specific domains (medical, legal, etc.)
- **Multi-GPU Training**: Scale from 1 GPU to multiple GPUs seamlessly
- **Memory Efficient**: Train large models on smaller hardware using smart techniques

## 🎯 **Who Should Use SFT-Play?**

### **Complete Beginners**
- Never trained an AI model before
- Want to learn how fine-tuning works
- Have a consumer GPU (RTX 4060, 4070, etc.)

### **Researchers & Students**
- Need to experiment with different models
- Want reproducible training setups
- Working on academic projects

### **Professionals & Enterprises**
- Training models for business applications
- Have high-end hardware (H100, H200, A100)
- Need production-ready solutions

### **AI Enthusiasts**
- Want to create custom AI assistants
- Interested in domain-specific models
- Love tinkering with the latest models

## 🏗️ **What Can You Train?**

### **Model Sizes & Hardware Requirements**

| Model Size | Examples | Minimum GPU | Recommended GPU | Training Mode |
|------------|----------|-------------|-----------------|---------------|
| **Small (1-7B)** | Qwen2.5-1.5B, Gemma-2B, Llama-7B | RTX 4060 (8GB) | RTX 4070 (12GB) | QLoRA |
| **Medium (8-20B)** | Qwen2.5-14B, Llama-13B | RTX 4080 (16GB) | RTX 4090 (24GB) | QLoRA |
| **Large (20-35B)** | Gemma-27B, Qwen-32B | A100 (40GB) | 2x H200 (282GB) | QLoRA + Multi-GPU |
| **Huge (70B+)** | Llama-70B, Qwen-72B | 2x A100 (160GB) | 4x H200 (564GB) | DeepSpeed + Multi-GPU |

### **Training Types**

1. **Supervised Fine-Tuning (SFT)**
   - **What**: Teach models to follow instructions
   - **Example**: Train a model to answer questions about your company
   - **Data**: Question-answer pairs

2. **Domain-Adaptive Pretraining (DAPT)**
   - **What**: Teach models domain-specific knowledge
   - **Example**: Train a model on medical literature
   - **Data**: Domain-specific documents (PDFs, DOCX)

3. **Mixed Training**
   - **What**: Combine domain knowledge with instruction following
   - **Example**: Medical expert that can chat naturally
   - **Data**: Medical documents + instruction data

## 🚀 **Quick Start: Your First Model**

### **Step 1: Installation (5 minutes)**

```bash
# Clone the repository
git clone <repository-url>
cd experiments/play

# Install dependencies (auto-detects pip or uv)
make install

# Create necessary directories
make setup-dirs
```

### **Step 2: Check Your Setup**

```bash
# See what GPUs you have
make gpu-info

# Example output:
# GPU 0: NVIDIA RTX 4060 (8.0 GB)
# Number of GPUs: 1
```

### **Step 3: Prepare Your Data**

You have three options:

#### **Option A: Use Sample Data (Easiest)**
```bash
# Run the interactive setup - it creates sample data for you
./workflows/quick_start.sh
```

#### **Option B: Bring Your Own Data**
Create a file `data/raw/my_data.jsonl` with your training examples:
```json
{"system": "You are a helpful assistant.", "user": "What is machine learning?", "assistant": "Machine learning is a subset of AI..."}
{"system": "You are a helpful assistant.", "user": "Explain neural networks.", "assistant": "Neural networks are..."}
```

#### **Option C: Domain Documents (DAPT)**
Put your PDF/DOCX files in `data/raw/` and run:
```bash
make dapt-docx  # Converts documents to training data
```

### **Step 4: Process Your Data**

```bash
# Convert raw data to training format
make process

# Add style instructions (optional)
make style STYLE="Answer in a friendly, helpful tone"

# Validate everything is ready
make check
```

### **Step 5: Start Training**

Choose based on your hardware:

#### **Single GPU (Most Common)**
```bash
# Basic training
make train

# Training with live monitoring
make train-bnb-tb
```

#### **Multiple GPUs**
```bash
# Setup multi-GPU (one-time)
make setup-accelerate

# Distributed training
make train-distributed-tb
```

### **Step 6: Monitor Training**

Open your browser to `http://localhost:6006` to see:
- Training loss going down
- GPU memory usage
- Training speed

### **Step 7: Test Your Model**

```bash
# Interactive chat with your model
make infer

# Example conversation:
# You: What is machine learning?
# Model: Machine learning is a subset of artificial intelligence...
```

## 🎛️ **Understanding the Commands**

### **Data Processing Commands**
```bash
make process        # Raw data → Training format
make style          # Add personality/instructions
make render         # Convert to model-specific format
make full-pipeline  # Do all data processing steps
```

### **Training Commands**
```bash
# Single GPU
make train          # Basic training
make train-bnb      # BitsAndBytes backend (stable)
make train-unsloth  # Unsloth backend (faster)

# Multi-GPU
make train-distributed    # Auto-detect GPUs
make train-deepspeed      # Maximum memory efficiency

# With Monitoring
make train-bnb-tb         # Single GPU + TensorBoard
make train-distributed-tb # Multi-GPU + TensorBoard
```

### **Evaluation & Testing**
```bash
make eval           # Test model performance
make eval-quick     # Quick test (200 samples)
make infer          # Chat with your model
make infer-batch    # Process multiple inputs
```

### **Utilities**
```bash
make gpu-info       # Check your hardware
make memory-check   # Monitor GPU memory
make check          # Validate setup
make clean          # Clean up files
make help           # See all commands
```

## 🔧 **Configuration Files Explained**

### **What are Config Files?**
Configuration files tell SFT-Play how to train your model. Think of them as recipes:

```yaml
# configs/my_config.yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct  # Which model to use
  max_seq_len: 512                   # How long conversations can be

tuning:
  mode: qlora                        # Memory-efficient training
  backend: bnb                       # Which training engine

train:
  epochs: 3                          # How many times to see the data
  learning_rate: 2e-4                # How fast to learn
  batch_size: auto                   # Let system decide
```

### **Pre-made Configs**
- `configs/run_bnb.yaml` - Small models, single GPU
- `configs/run_unsloth.yaml` - Faster training
- `configs/run_gemma27b_distributed.yaml` - Large models, multi-GPU
- `configs/run_dapt.yaml` - Domain adaptation

### **Using Different Configs**
```bash
# Use a specific config
make train CONFIG=configs/run_unsloth.yaml

# Works with any command
make train-distributed CONFIG=configs/run_gemma27b_distributed.yaml
```

## 🧠 **Understanding Training Modes**

### **QLoRA (Recommended for Beginners)**
- **What**: Memory-efficient training using 4-bit quantization
- **Pros**: Fits large models on small GPUs
- **Cons**: Slightly slower than full training
- **Use When**: You have limited GPU memory

### **LoRA**
- **What**: Efficient training by only updating small parts of the model
- **Pros**: Faster than QLoRA, still memory efficient
- **Cons**: Needs more GPU memory than QLoRA
- **Use When**: You have decent GPU memory (16GB+)

### **Full Fine-tuning**
- **What**: Update the entire model
- **Pros**: Best possible results
- **Cons**: Requires lots of GPU memory
- **Use When**: You have enterprise hardware

## 📊 **Understanding Your Hardware**

### **GPU Memory Requirements**

| Your GPU | VRAM | What You Can Train | Recommended Mode |
|----------|------|-------------------|------------------|
| RTX 4060 | 8GB | Up to 7B models | QLoRA |
| RTX 4070 | 12GB | Up to 7B models | QLoRA/LoRA |
| RTX 4080 | 16GB | Up to 13B models | QLoRA/LoRA |
| RTX 4090 | 24GB | Up to 20B models | QLoRA/LoRA |
| A100 | 40GB | Up to 30B models | QLoRA/LoRA/Full |
| A100 | 80GB | Up to 70B models | LoRA/Full |
| H100/H200 | 80-141GB | Any model | Full |

### **Multi-GPU Benefits**
- **More Memory**: 2x GPUs = 2x total memory
- **Faster Training**: Near 2x speedup for large models
- **Bigger Models**: Train models that don't fit on single GPU

## 🎯 **Common Use Cases & Examples**

### **Use Case 1: Customer Support Bot**
```bash
# 1. Prepare data
echo '{"system": "You are a helpful customer support agent.", "user": "How do I reset my password?", "assistant": "To reset your password, go to..."}' > data/raw/support.jsonl

# 2. Process and train
make process
make style STYLE="Be helpful and professional"
make train-bnb-tb

# 3. Test
make infer
```

### **Use Case 2: Domain Expert (Medical)**
```bash
# 1. Add medical documents to data/raw/
# 2. Process for domain adaptation
make dapt-docx
make dapt-train

# 3. Test domain knowledge
make infer
```

### **Use Case 3: Large Model on Multiple GPUs**
```bash
# 1. Setup multi-GPU
make setup-accelerate

# 2. Train large model
make train-distributed CONFIG=configs/run_gemma27b_distributed.yaml

# 3. Monitor on TensorBoard
# Visit: http://localhost:6006
```

## 🔍 **Troubleshooting Common Issues**

### **"CUDA out of memory"**
```bash
# Solution 1: Use smaller model
# Edit config: model.name: "Qwen/Qwen2.5-1.5B-Instruct"

# Solution 2: Reduce sequence length
# Edit config: model.max_seq_len: 256

# Solution 3: Use QLoRA
# Edit config: tuning.mode: qlora

# Solution 4: Check memory usage
make memory-check
```

### **"No module named 'accelerate'"**
```bash
# Install missing dependencies
pip install accelerate deepspeed
# or
make install
```

### **"Training is very slow"**
```bash
# Solution 1: Use Unsloth backend
make train CONFIG=configs/run_unsloth.yaml

# Solution 2: Use multiple GPUs
make train-distributed

# Solution 3: Reduce data size for testing
# Edit your data file to have fewer examples
```

### **"Model gives bad responses"**
```bash
# Solution 1: Train longer
# Edit config: train.epochs: 5

# Solution 2: Add more data
# Add more examples to your training file

# Solution 3: Improve data quality
# Make sure your examples are high-quality
```

## 📈 **Monitoring Your Training**

### **TensorBoard (Recommended)**
```bash
# Start training with monitoring
make train-bnb-tb

# Open browser to: http://localhost:6006
# You'll see:
# - Loss going down (good!)
# - Learning rate schedule
# - GPU memory usage
```

### **Command Line Monitoring**
```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Check memory
make memory-check

# Check training logs
tail -f outputs/*/training.log
```

## 🎓 **Next Steps**

### **Beginner → Intermediate**
1. Try different models (Gemma, Llama, Qwen)
2. Experiment with different training modes (QLoRA → LoRA)
3. Create domain-specific datasets
4. Learn to evaluate model quality

### **Intermediate → Advanced**
1. Multi-GPU training
2. Custom configurations
3. DeepSpeed optimization
4. Production deployment

### **Advanced → Expert**
1. Custom model architectures
2. Multi-node training
3. Custom training loops
4. Research contributions

## 🤝 **Getting Help**

### **Built-in Help**
```bash
make help           # See all commands
make check          # Validate your setup
./workflows/quick_start.sh  # Interactive setup
```

### **Documentation**
- `README.md` - Project overview
- `MULTI_GPU_GUIDE.md` - Multi-GPU training
- `AUTOMATION_GUIDE.md` - Advanced automation
- `SETUP_DOCUMENTATION.md` - Detailed setup

### **Common Commands Reference**
```bash
# Setup
make install && make setup-dirs

# Quick start
./workflows/quick_start.sh

# Single GPU training
make train-bnb-tb

# Multi-GPU training
make setup-accelerate
make train-distributed-tb

# Check everything
make gpu-info
make check
make help
```

## 🎉 **You're Ready!**

Congratulations! You now understand:
- What SFT-Play is and what it can do
- How to match your hardware to the right models
- How to prepare data and start training
- How to monitor and troubleshoot training
- How to scale from single to multiple GPUs

**Start with the quick start guide and experiment!** The best way to learn is by doing.

```bash
# Your first command:
./workflows/quick_start.sh
```

Welcome to the world of AI model fine-tuning! 🚀
