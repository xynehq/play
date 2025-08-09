# 🚀 SFT-Play v1.0 — Plug-and-Play LoRA/QLoRA Fine-Tuning in 8GB VRAM

**SFT-Play** is a reusable, beginner-friendly, 8GB-VRAM–friendly environment for supervised fine-tuning (SFT) of LLMs with **LoRA**/**QLoRA**. It's built for stability, speed, and minimal setup — so you can focus on experimentation instead of debugging.

---

## ✨ **What's New in v1.0**

### 🔹 Stable Backends with Auto-Fallback

* **BitsAndBytes** as the default stable backend
* **Unsloth** available for experimentation
* Automatic fallback to BitsAndBytes when Unsloth + XFormers incompatibilities are detected
* Backend-specific configs (`run_bnb.yaml`, `run_unsloth.yaml`) to avoid cross-contamination

### 🔹 Training Safety Features

* Backend stamping to prevent mismatched resume runs
* Config validation for precision, LoRA params, and paths
* Resume protection to avoid corrupted checkpoints
* Config defaults tuned for 8GB VRAM QLoRA

### 🔹 Logging & Monitoring

* **TensorBoard** integration out of the box
* Logging of `train/loss`, `eval/loss`, `train/lr`, and metrics
* Preconfigured Makefile commands:

  ```bash
  make train-bnb-tb     # Stable + TensorBoard
  make train-unsloth-tb # Experimentation + Fallback
  ```

### 🔹 Data & Templates

* Supports `system/user/assistant` structured chat data
* Built-in Jinja templating for instruction formatting
* CLI tools for:
  * Processing raw → structured data
  * Adding style prompts
  * Rendering templates

### 🔹 **High-Quality Inference Engine** ⭐ NEW

* **Proper Stopping Conditions**: Forces stop at EOS tokens and chat template boundaries
* **Clean Output Extraction**: Strips everything after first `<|assistant|>` → `<|user|>` boundary
* **Template Consistency**: Matches training template format exactly during inference
* **Single-Turn Generation**: Prevents multi-turn conversation bleed-through

**Example of clean inference:**
```python
# Raw generation might include:
# "<|system|>You are helpful</|system|><|user|>Hello</|user|><|assistant|>Hi there!<|user|>..."

# Cleaned output extracts only:
# "Hi there!"
```

### 🔹 Inference & Evaluation

* Clean single-turn inference with:
  ```bash
  make infer
  ```
* Evaluation script with ROUGE-L, SARI, EM, schema compliance
* LoRA merge script to export full FP16 model

### 🔹 **Complete Automation System**

* **20+ Makefile commands** for every aspect of the pipeline
* **Interactive setup** with `./workflows/quick_start.sh`
* **Batch processing** automation
* **Comprehensive validation** with `make check`

**Key automation commands:**
```bash
make install                 # Install dependencies (auto-detects uv/pip)
make setup-dirs             # Create all necessary directories
make full-pipeline          # Complete data processing pipeline
make train-with-tb          # Train with TensorBoard monitoring
make eval                   # Evaluate on validation set
make infer                  # Interactive inference
make merge                  # Merge LoRA adapters to FP16 model
```

### 🔹 **Model Management**

* **Automatic model downloading** from Hugging Face Hub
* **Local model caching** to avoid re-downloads
* **Hugging Face token integration** with multiple auth methods
* **Offline mode support** after initial download

---

## 📂 **Folder Structure**

```
sft-play/
├─ configs/              # Base + run configs
│  ├─ config_base.yaml   # Reusable defaults
│  ├─ config_run.yaml    # Per-run overrides
│  ├─ run_bnb.yaml      # BitsAndBytes backend config
│  └─ run_unsloth.yaml  # Unsloth backend config
├─ data/                 # Raw + processed datasets
│  ├─ raw/              # Input sources (json/csv/jsonl)
│  ├─ processed/        # Structured chat data
│  └─ rendered/         # Materialized templates
├─ chat_templates/       # Jinja templates
│  └─ default.jinja     # Default chat template
├─ scripts/              # Train, eval, infer, merge
│  ├─ train.py          # QLoRA/LoRA/Full training
│  ├─ eval.py           # Comprehensive evaluation
│  ├─ infer.py          # High-quality inference
│  ├─ process_data.py   # Data processing pipeline
│  └─ merge_lora.py     # Adapter merging
├─ outputs/              # Checkpoints & logs
├─ adapters/             # LoRA adapters (~50-200MB)
├─ workflows/            # Automation scripts
├─ Makefile             # Complete automation
└─ requirements.txt
```

---

## ✅ **Why v1.0 is Ready**

* ✅ Runs QLoRA on RTX 4060 (8GB VRAM) without OOM
* ✅ No manual debugging for XFormers/Unsloth issues
* ✅ Fully documented setup and commands
* ✅ High-quality inference with proper stopping
* ✅ Complete automation system
* ✅ Ready for public use as a **starter kit** for SFT

---

## 🚀 **Quick Start**

### Option 1: One-Command Setup (Recommended)

```bash
git clone https://github.com/your-username/sft-play
cd sft-play
./workflows/quick_start.sh
```

This interactive script will:
- Install dependencies (auto-detects uv or pip)
- Create all necessary directories
- Generate sample data if none exists
- Process data through the complete pipeline
- Guide you to training

### Option 2: Manual Setup

```bash
git clone https://github.com/your-username/sft-play
cd sft-play
pip install -r requirements.txt

make setup-dirs         # Create directories
make process           # Prepare dataset
make check             # Validate setup
make train-bnb-tb      # Start stable QLoRA training with TensorBoard
```

### Option 3: Advanced Workflow

```bash
make install && make setup-dirs
make process
make style STYLE="Be concise and professional"
make check
make train CONFIG=configs/run_bnb.yaml
make eval-test
make merge && make merge-test
```

---

## 🎯 **Supported Models & Hardware**

### **Tested Models:**
- Qwen/Qwen2.5-3B-Instruct
- Meta-Llama models (with proper tokens)
- Mistral models
- Any HuggingFace causal LM

### **Hardware Requirements:**
- **Minimum**: 8GB VRAM (RTX 4060, RTX 3070, etc.)
- **Recommended**: 12GB+ VRAM for larger models
- **CPU**: Any modern CPU with 16GB+ RAM

### **VRAM Usage Examples:**
- Qwen2.5-3B + QLoRA + BitsAndBytes: ~6.5GB VRAM
- Llama-7B + QLoRA + BitsAndBytes: ~7.8GB VRAM

---

## 🛠️ **Configuration Examples**

### Stable Training (Recommended)
```yaml
# configs/run_bnb.yaml
include: configs/config_base.yaml

tuning:
  backend: bnb             # BitsAndBytes backend
  mode: qlora

train:
  bf16: false              # BitsAndBytes works best with fp16
  fp16: true
  output_dir: outputs/run-bnb
```

### Experimental Training
```yaml
# configs/run_unsloth.yaml
include: configs/config_base.yaml

tuning:
  backend: unsloth         # Unsloth backend (auto-fallback to bnb)
  mode: qlora

train:
  bf16: true               # Unsloth works better with bfloat16
  fp16: false
  output_dir: outputs/run-unsloth
```

---

## 📊 **Performance & Quality**

### **Training Performance:**
- **Memory Efficient**: Only saves LoRA adapters (~50-200MB vs full model GB)
- **VRAM Optimized**: Auto-tunes batch size and gradient accumulation
- **Fast Resume**: Lightweight checkpoints enable quick restarts

### **Inference Quality:**
- **Clean Output**: Proper template boundary detection
- **Consistent Format**: Training-inference template matching
- **Single-Turn Focus**: Prevents multi-turn conversation bleed
- **Production Ready**: High-quality responses suitable for deployment

### **Automation Benefits:**
- **Zero Config**: Works out of the box with sensible defaults
- **Error Prevention**: Comprehensive validation and safety checks
- **Developer Friendly**: 20+ automation commands for every workflow

---

## 💡 **Pro Tips**

* **For Maximum Stability**: Use `make train-bnb-tb`
* **For Experimentation**: Try `make train-unsloth-tb` (auto-fallback included)
* **For Quick Validation**: Use `make check` before training
* **For Clean Inference**: The new inference engine handles all edge cases
* **For Automation**: Explore `make help` for all available commands

---

## 🔧 **Troubleshooting**

### Common Issues Resolved in v1.0:

* ✅ **XFormers Compatibility**: Auto-fallback prevents crashes
* ✅ **Backend Confusion**: Separate configs prevent cross-contamination
* ✅ **Memory Issues**: Auto-tuning handles VRAM constraints
* ✅ **Template Mismatches**: Inference engine ensures consistency
* ✅ **Setup Complexity**: One-command setup and comprehensive automation

### Quick Fixes:
```bash
make check                    # Validate everything
make clean && make setup-dirs # Reset if needed
./workflows/quick_start.sh    # Guided setup
```

---

## 📚 **Documentation**

- **README.md** - Complete usage guide with examples
- **AUTOMATION_GUIDE.md** - Detailed automation system documentation
- **SETUP_DOCUMENTATION.md** - Complete project setup guide
- **This Release Note** - What's new in v1.0

---

## 🎉 **Ready to Fine-Tune?**

SFT-Play v1.0 is production-ready for:
- **AI Hobbyists** — Fine-tune on your own hardware
- **Researchers** — Quick experimentation before scaling
- **Educators** — Teaching LLM fine-tuning
- **Developers** — Prototyping AI features
- **Open Source Contributors** — Building and sharing models

**Get started now:**
```bash
git clone https://github.com/your-username/sft-play
cd sft-play
./workflows/quick_start.sh
```

---

*Happy fine-tuning! 🚀*
