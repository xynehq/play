# SFT-Play (QLoRA-ready, 8-GB Friendly)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.42+-orange.svg)](https://huggingface.co/transformers/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Plug-and-play Supervised Fine-Tuning** on small GPUs.
Single config, QLoRA/LoRA/Full switches, bitsandbytes/Unsloth backends, Jinja chat templating, TensorBoard live UI, and lean checkpoints (save adapters, not full models).

---

## 🎯 Who Should Use SFT-Play?

- **AI hobbyists** — fine-tune models on your own dataset without cloud GPUs
- **Researchers** — run small-scale experiments before scaling to larger infrastructure
- **Educators** — teach LLM fine-tuning with a minimal, reproducible setup
- **Open-source contributors** — build datasets + share fine-tuned models efficiently
- **Developers** — prototype AI features with custom models on local hardware

---

## ✨ Features

* **Runs on any single GPU (8 GB+)** — VRAM probe auto-tunes batch/grad-accum.
* **Two-config UX** — `config_base.yaml` (defaults) + `config_run.yaml` (you edit).
* **Tuning modes** — `qlora | lora | full` (config switch).
* **Backends** — `bitsandbytes` (default) or `unsloth` (optional; auto-fallback to bnb).
* **Data pipeline** — raw → structured chat (`system,user,assistant`) → Jinja render on-the-fly.
* **UI** — **TensorBoard** only (loss/metrics/LR; optional GPU stats).
* **Tiny checkpoints** — LoRA adapters only (~50-200 MB vs. full model's multiple GB).
* **Complete automation** — Makefile + workflows for zero-config setup.

---

## 🗂️ Repo Layout

```
sft-play/
├─ configs/
│  ├─ config_base.yaml          # reusable defaults (rarely change)
│  └─ config_run.yaml           # per-run overrides (you edit)
├─ data/
│  ├─ raw/                      # input sources (json/csv/jsonl)
│  ├─ processed/                # structured chat (system,user,assistant)
│  ├─ processed_with_style/     # optional: after style injection
│  └─ rendered/                 # optional: materialized seq2seq (input,target)
├─ chat_templates/
│  └─ default.jinja             # single Jinja template
├─ scripts/
│  ├─ process_data.py           # raw → structured chat + split
│  ├─ style_prompt.py           # inject/override system/style rule
│  ├─ render_template.py        # (optional) Jinja → seq2seq jsonl
│  ├─ train.py                  # QLoRA/LoRA/Full; bnb/Unsloth; TB logging
│  ├─ eval.py                   # ROUGE-L/SARI/Exact-Match (+ schema checks)
│  ├─ infer.py                  # batch/interactive inference (same template)
│  └─ merge_lora.py             # merge adapters → single FP16 model (optional)
├─ env/
│  └─ accelerate_config.yaml    # fp16, single-GPU defaults
├─ outputs/                     # TB logs, metrics, sample preds
├─ adapters/                    # LoRA adapter checkpoints
├─ workflows/                   # automation scripts
│  ├─ quick_start.sh            # interactive setup with sample data
│  └─ batch_process.sh          # batch processing automation
├─ Makefile                     # complete automation commands
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Configs

### `configs/config_base.yaml` (defaults)

* Training: epochs, warmup, weight\_decay, fp16, gradient\_checkpointing
* Checkpoint/eval: `save_strategy`, `save_steps`, `eval_strategy`, `save_total_limit`, `metric_for_best_model`, `load_best_model_at_end`
* Data: `format: chat`, `template_path`, **split ratios** (train/val/test)
* Logging: `backend: tensorboard`, `log_interval`

### `configs/config_run.yaml` (you edit)

```yaml
include: configs/config_base.yaml

model:
  name: Qwen/Qwen2.5-3B-Instruct
  type: causal              # or seq2seq
  max_seq_len: 512

tuning:
  mode: qlora               # qlora | lora | full
  backend: bnb              # bnb | unsloth
  lora:
    r: 32
    alpha: 32
    dropout: 0.05
    target_modules: auto

data:
  train_path: data/processed/train.jsonl
  val_path:   data/processed/val.jsonl
  test_path:  data/processed/test.jsonl

gen:
  max_new_tokens: 200
  temperature: 0.2
  top_p: 0.9
```

> **What usually changes per run?** `model.name`, `tuning.mode`, `tuning.backend`, occasionally `max_seq_len`, `lora.{r,alpha,dropout}`, or `gen.*`.
> Everything else stays the same.

---

## 🚀 Quickstart

### Option 1: Automated Setup (Recommended)

**Complete setup in one command:**

```bash
./workflows/quick_start.sh
```

This interactive script will:
- Install dependencies (auto-detects uv or pip)
- Create all necessary directories
- Generate sample data if none exists
- Process data through the complete pipeline
- Guide you to training

**Or use individual Makefile commands:**

```bash
make help                    # See all available commands
make install                 # Install dependencies
make setup-dirs             # Create directories
make full-pipeline          # Complete data processing
make check                  # Validate setup before training
make train-with-tb          # Train with TensorBoard monitoring
```

### Option 2: Manual Step-by-Step

#### 0) Install

```bash
pip install -r requirements.txt
# or
# uv venv && uv pip install -e .
```

(Optional) Configure Accelerate:

```bash
accelerate config  # or use env/accelerate_config.yaml
```

#### 1) Add raw data

Create `data/raw/raw.jsonl` or `raw.json`. Example (JSONL):

```json
{"system":"You are a helpful assistant.","user":"What is machine learning?","assistant":"Machine learning is..."}
```

(Your `process_data.py` also supports simple dicts like `{"question":"...","answer":"..."}`.)

#### 2) Process raw → structured chat

```bash
python scripts/process_data.py --config configs/config_run.yaml --raw_path data/raw/raw.jsonl
# writes data/processed/{train,val,test}.jsonl
```

#### 3) (Optional) Inject style/system rule

```bash
python scripts/style_prompt.py --config configs/config_run.yaml \
  --style "Answer in ≤2 concise sentences. No markdown." \
  --in data/processed/train.jsonl \
  --out data/processed_with_style/train.jsonl
# repeat for val/test if desired and update config_run.yaml data paths
```

#### 4) Validate setup

```bash
make check                   # Comprehensive sanity check
```

#### 5) Train (TensorBoard logs)

```bash
python scripts/train.py --config configs/config_run.yaml
tensorboard --logdir outputs/
```

* See `train/loss`, `eval/loss`, `train/lr`, and `eval/rougeL` live.
* Checkpoints saved every `save_steps` (adapters + trainer state only).

#### 6) Evaluate

```bash
python scripts/eval.py --config configs/config_run.yaml --split val
# writes outputs/metrics.json and outputs/samples.jsonl
```

#### 7) Inference

Interactive:

```bash
python scripts/infer.py --config configs/config_run.yaml
```

Batch:

```bash
echo "Explain QLoRA in two lines." > demo_inputs.txt
python scripts/infer.py --config configs/config_run.yaml --mode batch --input_file demo_inputs.txt --output_file outputs/preds.txt
```

#### 8) (Optional) Merge adapters → FP16 model

```bash
python scripts/merge_lora.py --config configs/config_run.yaml \
  --adapters adapters/last \
  --out outputs/merged_fp16 \
  --dtype fp16
```

---

## 🛠️ Automation Commands

### Complete Makefile Reference

```bash
# Setup
make install                 # Install dependencies (auto-detects uv/pip)
make setup-dirs             # Create all necessary directories

# Data Pipeline
make process                # Process raw data to structured chat
make style                  # Apply style prompts to all splits
make render                 # Render chat templates
make full-pipeline          # Complete data processing pipeline

# Training & Evaluation
make train                  # Start training
make train-with-tb          # Train with TensorBoard monitoring
make eval                   # Evaluate on validation set
make eval-test              # Evaluate on test set
make eval-quick             # Quick evaluation (200 samples)
make eval-full              # Full evaluation (no limit)

# Inference
make infer                  # Interactive inference (chat mode)
make infer-batch            # Batch inference from file
make infer-interactive      # Interactive inference (explicit)

# Model Management
make merge                  # Merge LoRA adapters to FP16 model
make merge-bf16             # Merge LoRA adapters to BF16 model
make merge-test             # Test merged model loading

# Monitoring
make tensorboard            # Start TensorBoard manually
make stop-tb                # Stop background TensorBoard

# Utilities
make check                  # Validate project setup
make clean                  # Clean generated files
make help                   # Show all commands
```

### Workflow Scripts

```bash
# Interactive setup with sample data
./workflows/quick_start.sh

# Batch processing for multiple datasets
./workflows/batch_process.sh
```

### Customization

```bash
# Custom style prompts
make style STYLE="Answer in JSON format only"

# Custom configuration
make train CONFIG=configs/my_config.yaml

# Custom workflows
make process && make style && make train
```

---

## 🧠 Design Notes

* **Memory-efficient checkpoints**: we save **only LoRA adapters** + trainer state. Result: tiny checkpoints, fast resume. Merge at the end only if you need a single FP16 folder.
* **VRAM-aware**: when `batch_size/grad_accum` are `auto`, training probes free VRAM and picks safe values (starts at `bs=1`, increases accumulation).
* **Template flexibility**: training renders Jinja on-the-fly, so you can change `chat_templates/default.jinja` without reprocessing.
  ```jinja
  {{ system }}
  User: {{ user }}
  Assistant: {{ assistant }}
  ```
* **Backends**: set `tuning.backend: unsloth` if installed; otherwise it auto-falls back to bnb with a warning.
* **Complete automation**: Makefile provides 20+ commands for every aspect of the pipeline.
* **VRAM efficiency**: Qwen2.5-3B + QLoRA + bnb → ~6.5 GB VRAM at seq_len=512
* **Modes**:

  * `qlora` → 4-bit base + LoRA (best for 8 GB on 7B/3B causal LMs)
  * `lora`  → fp16/bf16 base + LoRA (fine for 1–3B, or enough VRAM)
  * `full`  → full fine-tune (use for small seq2seq, e.g., FLAN-T5-base)

---

## 🧪 Troubleshooting

* **CUDA OOM**

  * Lower `model.max_seq_len` (e.g., 512 → 384).
  * Keep `mode: qlora`, `backend: bnb`, `batch_size: auto`, `grad_accum: auto`.
  * Ensure TensorBoard isn't eating VRAM on the same GPU (runs on CPU, but double-check).

* **Unsloth import fails**

  * Use `backend: bnb` (default).
  * If you insist on Unsloth, match its CUDA/PTX requirements.

* **Weird formatting in generations**

  * Check `chat_templates/default.jinja` and your style prompt.
  * Remember causal LMs may echo the prompt; `infer.py` strips assistant tags heuristically.

* **Metrics too low**

  * Increase epochs to 3–5.
  * Tune LoRA `r` (16→32) or LR (2e-4 → 1e-4).
  * Ensure your processed data is clean and task-consistent.

* **Setup issues**

  * Run `make check` to validate your setup
  * Use `./workflows/quick_start.sh` for guided setup
  * Check `AUTOMATION_GUIDE.md` for detailed automation docs

---

## ✅ Definition of Done (v0.1)

* End-to-end run on **Qwen2.5-3B (QLoRA+bnb)** on an **8 GB** GPU without OOM.
* Live TensorBoard charts.
* `outputs/metrics.json` with ROUGE-L (and others).
* `infer.py` produces sensible answers.
* (Optional) `outputs/merged_fp16` exists and loads with HF.
* **Complete automation** with Makefile and workflow scripts.
* **Sanity checking** with `make check` validation.

---

## 📚 Documentation

- **AUTOMATION_GUIDE.md** - Detailed automation system documentation
- **SETUP_DOCUMENTATION.md** - Complete project setup guide
- **LICENSE** - MIT License for open source use

---

## 🎯 Quick Examples

**Complete beginner workflow:**
```bash
./workflows/quick_start.sh     # One command setup
make check                     # Validate everything
make train-with-tb            # Train with monitoring
```

**Advanced user workflow:**
```bash
make install && make setup-dirs
make process
make style STYLE="Be concise and professional"
make check
make train CONFIG=configs/my_qlora.yaml
make eval-test
make merge && make merge-test
```

**Development workflow:**
```bash
make full-pipeline            # Process all data
make eval-quick               # Fast validation
make infer                    # Test interactively
```

That's it! The automation system makes SFT-Play truly plug-and-play. Run `make help` to see all available commands, or start with `./workflows/quick_start.sh` for a guided experience.
