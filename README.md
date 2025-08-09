# SFT-Play (QLoRA-ready, 8-GB Friendly) - QLoRA-ready SFT starter kit

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
* **Model Caching** - Automatically download models from Hugging Face Hub and cache them locally.
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
│  ├─ merge_lora.py             # merge adapters → single FP16 model (optional)
│  └─ utils/
│     └─ model_store.py        # handles model downloading/caching
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

### Backend-Specific Configs (Recommended)

**For BitsAndBytes (Stable, Broad Compatibility):**
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

**For Unsloth (Faster, Requires Compatible CUDA):**
```yaml
# configs/run_unsloth.yaml
include: configs/config_base.yaml

tuning:
  backend: unsloth         # Unsloth backend
  mode: qlora

train:
  bf16: true               # Unsloth works better with bfloat16
  fp16: false
  output_dir: outputs/run-unsloth
```

### `configs/config_run.yaml` (legacy/custom)

```yaml
include: configs/config_base.yaml

model:
  name: Qwen/Qwen2.5-3B-Instruct     # HF repo_id OR local folder path
  local_dir: models/qwen2.5-3b       # where to store/download the model
  revision: main                     # optional: tag/sha
  trust_remote_code: true            # some repos need this
  type: causal
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

### Backend Safety Features

**Automatic Backend Stamping:**
- Each training run creates `outputs/<run>/backend.json` with backend info
- Prevents accidental resume across different backends
- Validates configuration consistency on resume

**XFormers Safety:**
- Unsloth automatically disables XFormers to prevent compatibility issues
- BitsAndBytes uses standard PyTorch attention mechanisms
- Automatic fallback from Unsloth to BitsAndBytes if import fails

**Precision Auto-Detection:**
- Auto-enables bf16 on Ada GPUs (RTX 40xx series) when neither precision is set
- Auto-enables fp16 on other GPUs as fallback
- Prevents both bf16 and fp16 being enabled simultaneously

> **Recommended Usage:** Use `configs/run_bnb.yaml` for stability or `configs/run_unsloth.yaml` for speed. The backend-specific configs ensure optimal settings and prevent configuration conflicts.

---

## 🤗 Hugging Face Token

To download models from the Hugging Face Hub, you need to provide an access token. You can do this in one of three ways:

1.  **CLI Login (Recommended):**
    ```bash
    huggingface-cli login
    ```
    This will store your token securely on your machine.

2.  **Environment Variable:**
    ```bash
    export HUGGINGFACE_HUB_TOKEN=hf_...
    ```
    You can add this to your shell profile (e.g., `.bashrc`, `.zshrc`) or a `.env` file.

3.  **Offline Mode:**
    After downloading a model once, you can work offline:
    ```bash
    export HF_HUB_OFFLINE=1
    ```

## 📈 TensorBoard

We log to a fixed path: `outputs/tb`.

Start it:
```bash
make tensorboard            # uses port 6006
make tensorboard TB_PORT=6007
```

Stop it:
```bash
make tb-stop
```

If TB shows "No dashboards…" check you're pointing at the absolute path:
```bash
tensorboard --logdir "$(pwd)/outputs/tb" --port 6006
```

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

You can also pre-download models using the Makefile:
```bash
make download-model MODEL=Qwen/Qwen2.5-3B-Instruct
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

##### Inference Quality Improvements

The inference script includes several optimizations to ensure high-quality, single-turn responses:

**1. Proper Stopping Conditions**
- Forces stop at EOS tokens using `eos_token_id=tokenizer.eos_token_id`
- Adds custom stop tokens for chat template boundaries (`<|user|>`, `</|assistant|>`)
- Prevents multi-turn generation where the model continues beyond the assistant's response

**2. Template Boundary Parsing**
- Extracts only the assistant's response from the full generation
- Strips everything after the first `<|assistant|>` → `<|user|>` boundary
- Handles both `</|assistant|>` end tags and natural conversation boundaries

**3. Template Consistency**
- Loads the same Jinja chat template used during training
- Ensures inference format exactly matches training format
- Prevents template mismatches that can cause poor generation quality

**Example of clean output extraction:**
```python
# Raw generation might include:
# "<|system|>You are helpful</|system|><|user|>Hello</|user|><|assistant|>Hi there!<|user|>..."

# Cleaned output extracts only:
# "Hi there!"
```

These improvements ensure that:
- Models stop generating at appropriate conversation boundaries
- Output is clean and contains only the intended assistant response
- Template consistency is maintained between training and inference
- Multi-turn conversations don't bleed into single responses

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
make train                  # Start training with current config
make train-bnb              # Start training with BitsAndBytes backend
make train-unsloth          # Start training with Unsloth backend
make train-with-tb          # Train with TensorBoard monitoring
make train-bnb-tb           # BitsAndBytes training with TensorBoard
make train-unsloth-tb       # Unsloth training with TensorBoard
make eval                   # Evaluate on validation set
make eval-test              # Evaluate on test set
make eval-quick             # Quick evaluation (200 samples)
make eval-full              # Full evaluation (no limit)

# Inference
make infer                  # Interactive inference (chat mode)
make infer-batch            # Batch inference from file
make infer-interactive      # Interactive inference (explicit)

# Model Management
make download-model         # Download a model from Hugging Face Hub
make merge                  # Merge LoRA adapters to FP16 model
make merge-bf16             # Merge LoRA adapters to BF16 model
make merge-test             # Test merged model loading

# Monitoring
make tensorboard            # Start TensorBoard on outputs/tb
make tb-stop                # Kill any running TensorBoard
make tb-clean               # Remove TB event files
make tb-open                # Print exact path & suggest URL

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

### Configuration Issues

* **Training Arguments Mismatch Error**
  ```
  ValueError: --load_best_model_at_end requires the save and eval strategy to match
  ```
  **Solution**: This was fixed in the training script. The issue occurred when `evaluation_strategy` and `save_strategy` didn't match. The script now properly handles both `evaluation_strategy` and `eval_strategy` keys from config files.

* **Backend Switching: BitsAndBytes ↔ Unsloth**

  **To switch from BitsAndBytes to Unsloth:**
  ```yaml
  # In configs/config_run.yaml
  tuning:
    backend: unsloth
  
  # In configs/config_base.yaml
  train:
    bf16: true    # Unsloth works better with bfloat16
    fp16: false
  ```

  **To switch from Unsloth to BitsAndBytes:**
  ```yaml
  # In configs/config_run.yaml
  tuning:
    backend: bnb
  
  # In configs/config_base.yaml
  train:
    bf16: false   # BitsAndBytes is more stable with float16
    fp16: true
  ```

  **Key differences:**
  - **Unsloth**: Faster training, requires specific CUDA versions, works best with `bf16: true`
  - **BitsAndBytes**: More stable, broader compatibility, works best with `fp16: true`
  - **Auto-fallback**: If Unsloth fails to import, the system automatically falls back to BitsAndBytes

### Memory and Performance Issues

* **CUDA OOM**

  * Lower `model.max_seq_len` (e.g., 512 → 384).
  * Keep `mode: qlora`, `backend: bnb`, `batch_size: auto`, `grad_accum: auto`.
  * Ensure TensorBoard isn't eating VRAM on the same GPU (runs on CPU, but double-check).

* **Unsloth import fails**

  * Use `backend: bnb` (default).
  * If you insist on Unsloth, match its CUDA/PTX requirements.

* **XFormers compatibility issues**
  ```
  NotImplementedError: No operator found for `memory_efficient_attention_backward`
  ```
  **Solution**: This is a known compatibility issue with Unsloth + XFormers on certain GPU/CUDA configurations. The system now automatically falls back to BitsAndBytes when Unsloth is requested:
  
  **Automatic Fallback**: When you use `configs/run_unsloth.yaml`, the system detects XFormers issues and automatically switches to BitsAndBytes with a warning message.
  
  **Recommended Approach**: Use BitsAndBytes directly for maximum stability:
  ```bash
  make train-bnb        # Direct BitsAndBytes training
  make train-bnb-tb     # BitsAndBytes with TensorBoard
  ```
  
  **Manual Configuration**: If you want to force BitsAndBytes:
  ```yaml
  tuning:
    backend: bnb
  train:
    bf16: false
    fp16: true
  ```

### Data and Training Issues

* **Weird formatting in generations**

  * Check `chat_templates/default.jinja` and your style prompt.
  * Remember causal LMs may echo the prompt; `infer.py` strips assistant tags heuristically.

* **Metrics too low**

  * Increase epochs to 3–5.
  * Tune LoRA `r` (16→32) or LR (2e-4 → 1e-4).
  * Ensure your processed data is clean and task-consistent.

* **ROUGE metrics showing 0.0**
  ```
  [train] Warning: Could not compute ROUGE metrics: argument 'ids': 'list' object cannot be interpreted as an integer
  ```
  **Note**: This is a known issue with the ROUGE evaluation library and doesn't affect training. The model is still learning (check the decreasing loss values).

### Setup Issues

* **Setup issues**

  * Run `make check` to validate your setup
  * Use `./workflows/quick_start.sh` for guided setup
  * Check `AUTOMATION_GUIDE.md` for detailed automation docs

### Configuration Validation

* **Before training, always validate your config:**
  ```bash
  make check                    # Comprehensive validation
  python scripts/train.py --config configs/config_run.yaml --help  # Check arguments
  ```

* **Common config mistakes:**
  - Mismatched `evaluation_strategy` and `save_strategy` (now auto-fixed)
  - Wrong precision settings for your backend (`bf16` vs `fp16`)
  - Missing data files or incorrect paths
  - Incompatible model settings with available VRAM

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
