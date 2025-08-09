# SFT-Play (QLoRA-ready, 8-GB Friendly)

**A plug-and-play Supervised Fine-Tuning template for small GPUs. Single config, QLoRA/LoRA/Full switches, bitsandbytes/Unsloth backends, Jinja chat templating, live training UI, and lean checkpointing (save adapters, not the world).**

## Use Cases

* **Prompt rewriter / canonicalizer** (messy → structured schema)
* **Intent → JSON** (enterprise assistant, mail/search ops)
* **Summarization with constraints** (tone/length/format)
* **Domain instruction tuning** (fintech, support, ed-tech)
* **Query rewrite for retrieval** (Vespa/RAG recall boost)
* **Style control** (formal tone, 2-line answers, JSON output)

## Why This Project

* **Runs on any single GPU (8 GB+)** with auto VRAM-aware defaults
* **Single config** drives *everything* (model, method, backend, checkpoints)
* **Batteries included**: data processing → templating → train → eval → infer
* **Live dashboards** (Trackio/TensorBoard/W&B) with zero code changes
* **Memory-efficient**: checkpoints save LoRA adapters & trainer state only

## Folder Structure

```
sft-play/
├─ configs/
│  ├─ config_base.yaml        # reusable defaults (you rarely touch)
│  └─ config_run.yaml         # per-run overrides (you edit this)
├─ data/
│  ├─ raw/                    # your raw inputs (json/csv/etc.)
│  ├─ processed/              # structured chat JSONL (system/user/assistant)
│  ├─ processed_with_style/   # optional: after injecting style/system rules
│  └─ rendered/               # optional: seq2seq JSONL after Jinja rendering
├─ chat_templates/
│  └─ default.jinja           # single Jinja template (swap if needed)
├─ scripts/
│  ├─ process_data.py         # raw → structured chat + splits
│  ├─ style_prompt.py         # inject/override system/style instructions
│  ├─ render_template.py      # (optional) materialize Jinja to seq2seq
│  ├─ train.py                # SFT w/ QLoRA/LoRA/Full; bnb/Unsloth switch
│  ├─ infer.py                # batch/interactive inference
│  ├─ eval.py                 # ROUGE/SARI/EM + optional schema checks
│  └─ merge_lora.py           # optionally export merged FP16 model
├─ env/
│  └─ accelerate_config.yaml  # single-GPU fp16 defaults
├─ outputs/                   # logs, metrics, samples, TB/Trackio runs
├─ adapters/                  # LoRA adapter checkpoints
├─ requirements.txt
└─ README.md
```

## Configs (Two-File System)

* **`configs/config_base.yaml`** — global defaults you keep stable:
  * logging backend + intervals
  * checkpoint/eval strategy
  * warmup/weight_decay/fp16/grad-ckpt
  * data format, Jinja template path
  * split ratios (e.g., 0.8/0.1/0.1)

* **`configs/config_run.yaml`** — you edit per run:
  * `model.name`, `model.type` (causal/seq2seq), `max_seq_len`
  * `tuning.mode: qlora|lora|full`
  * `tuning.backend: bnb|unsloth`
  * LoRA knobs (`r/alpha/dropout/target_modules: auto`)
  * data paths (train/val/test)

*(Loader merges run → base; any key in run overrides base.)*

## File-by-File (What Each Does)

### `scripts/process_data.py`

* Reads `data/raw/*` (JSON/CSV/HF dataset)
* Normalizes to **structured chat** rows:
  ```json
  {"system":"...","user":"question text","assistant":"answer text"}
  ```
* Splits using ratios from config (train/val/test)
* Writes JSONL to `data/processed/`

### `scripts/style_prompt.py`

* Injects or replaces a **system/style** instruction (e.g., "Answer in 2 concise lines, no markdown.")
* Saves to `data/processed_with_style/` so you keep originals intact

### `scripts/render_template.py` (optional)

* Applies `chat_templates/default.jinja` to convert structured chat → **rendered seq2seq** pairs:
  ```json
  {"input":"<rendered prompt>", "target":"gold answer"}
  ```
* Good for auditing exactly what the model sees

### `chat_templates/default.jinja`

* One Jinja file that defines how to format system+user for your model family
* Swap or tweak without touching Python code

### `scripts/train.py`

* Loads merged config; probes VRAM to auto-set safe `batch_size/grad_accum`
* Selects **bnb** or **Unsloth** backend
* Selects **QLoRA / LoRA / Full** path:
  * `qlora` → 4-bit base + PEFT adapters
  * `lora`  → fp16/bf16 (or 8-bit) base + PEFT adapters
  * `full`  → full FT (good for tiny seq2seq like FLAN-T5-base)
* Renders Jinja **on the fly** in a collate (so templates can change anytime)
* Wires **Trackio/TensorBoard/W&B** from config
* Checkpoints on **steps/epoch** as configured

### `scripts/eval.py`

* Computes ROUGE-L, SARI, Exact-Match; optional **schema-compliance %** via regex hooks
* Writes `metrics.json` + a few qualitative examples

### `scripts/infer.py`

* Loads base + adapters; runs batch or interactive prompts
* Respects the same Jinja template and generation params

### `scripts/merge_lora.py`

* (Optional) Merges adapters into base FP16 weights for deployment outside PEFT
* Useful if you want a single model file for serving

### `env/accelerate_config.yaml`

* Single-GPU fp16 default; works out of the box

## Checkpointing Strategy (Memory-Efficient)

* **Default:** save only **LoRA adapters** + **Trainer state** (optimizer/scheduler/step) to `adapters/` and `outputs/`
* **Why:** adapter checkpoints are tiny (MBs), resume is fast, disk stays clean
* **Best model:** `metric_for_best_model` + `load_best_model_at_end: true` keep just top-K checkpoints
* **When to merge:** if you must deploy as a single model, run `merge_lora.py` once at the end (exports FP16)

## Live UI (How to View)

Choose once in config:

```yaml
logging:
  backend: trackio    # or tensorboard | wandb
  project: sft-play
  run_name: qwen3b_qlora_bnb_run1
  log_interval: 20
```

### Trackio (default)
* Start training: `make train` (or run the python command)
* View live: `trackio show --project sft-play` (opens interactive dashboard)
* See: train/eval loss, LR, ROUGE/SARI, schema-compliance, GPU util/VRAM

### TensorBoard
* `tensorboard --logdir outputs/`

### W&B
* `wandb login` once; metrics stream automatically

## Installation

### Option 1: pip (standard)

```bash
pip install -r requirements.txt
accelerate config  # (or use provided env/accelerate_config.yaml)
```

### Option 2: uv (faster installs, lockfile)

```bash
uv venv
uv pip install -e .
```

## Requirements

```
torch>=2.2
transformers>=4.42
datasets
accelerate
peft
bitsandbytes
jinja2
evaluate
rouge-score
sacrebleu
tensorboard
pyyaml
tqdm
```

## Quickstart (End-to-End)

### Option 1: Automated Setup (Recommended)

```bash
# Quick start with interactive setup
./workflows/quick_start.sh

# Or use Makefile commands
make help                    # See all available commands
make install                 # Install dependencies
make setup-dirs             # Create directory structure
make full-pipeline          # Run complete data processing
make train                  # Start training
make eval                   # Evaluate model
make infer                  # Run inference
```

### Option 2: Manual Step-by-Step

```bash
# 0) Install
pip install -r requirements.txt
accelerate config  # (or use provided env/accelerate_config.yaml)

# 1) Process raw → structured chat
python scripts/process_data.py --config configs/config_run.yaml

# 2) (Optional) inject style/system rule
python scripts/style_prompt.py --config configs/config_run.yaml \
  --style "Answer concisely in 2 lines. No markdown." \
  --in data/processed/train.jsonl \
  --out data/processed_with_style/train.jsonl

# 3) Train (renders Jinja on the fly; logs to Trackio/TB/W&B)
python scripts/train.py --config configs/config_run.yaml

# 4) Evaluate
python scripts/eval.py --config configs/config_run.yaml

# 5) Inference
python scripts/infer.py --config configs/config_run.yaml \
  --input demo_inputs.txt --out outputs/preds.txt

# 6) (Optional) Merge adapters to a single FP16 model
python scripts/merge_lora.py --config configs/config_run.yaml \
  --out outputs/merged_fp16
```

## Automation & Workflows

### Makefile Commands

The project includes a comprehensive Makefile for easy automation:

```bash
# Setup and installation
make install                 # Auto-detect and use uv or pip
make setup-dirs             # Create all necessary directories

# Data processing pipeline
make process                # Process raw data
make style                  # Apply style prompts to all splits
make render                 # Render chat templates
make full-pipeline          # Run complete data processing

# Training and evaluation
make train                  # Start training
make eval                   # Run evaluation
make infer                  # Run inference with demo data

# Utilities
make clean                  # Clean all generated files
make merge                  # Merge LoRA adapters

# Customization
make style STYLE="Your custom style prompt"
make train CONFIG=configs/your_config.yaml
```

### Quick Start Script

Interactive setup with sample data:

```bash
./workflows/quick_start.sh
```

This script will:
- Set up directories
- Install dependencies
- Create sample data if none exists
- Guide you through the complete pipeline
- Provide next steps

### Batch Processing

For processing multiple datasets:

```bash
./workflows/batch_process.sh
```

Features:
- Process multiple datasets with different configurations
- Apply different style prompts per dataset
- Run multiple training experiments
- Automated experiment tracking

## Configuration Examples

### QLoRA with bitsandbytes (most memory-efficient)

```yaml
# configs/config_run.yaml
model:
  name: "microsoft/DialoGPT-medium"
  type: "causal"
  max_seq_len: 512

tuning:
  mode: "qlora"
  backend: "bnb"
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: "auto"

data:
  train_path: "data/processed/train.jsonl"
  val_path: "data/processed/val.jsonl"
```

### LoRA with Unsloth (faster training)

```yaml
# configs/config_run.yaml
model:
  name: "unsloth/llama-2-7b-bnb-4bit"
  type: "causal"
  max_seq_len: 1024

tuning:
  mode: "lora"
  backend: "unsloth"
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
```

### Full Fine-tuning (small models)

```yaml
# configs/config_run.yaml
model:
  name: "google/flan-t5-base"
  type: "seq2seq"
  max_seq_len: 512

tuning:
  mode: "full"
  backend: "bnb"
```

## Tips & Best Practices

1. **Start small**: Use QLoRA with r=16 for initial experiments
2. **Monitor VRAM**: The scripts auto-adjust batch size, but watch GPU memory
3. **Template testing**: Use `render_template.py` to verify your Jinja formatting
4. **Incremental style**: Use `style_prompt.py` to test different system prompts without reprocessing data
5. **Checkpoint management**: Set `save_total_limit` in config to avoid filling disk
6. **Evaluation frequency**: Balance `eval_steps` vs training speed

## Troubleshooting

### CUDA Out of Memory
* Reduce `per_device_train_batch_size` in config
* Enable `gradient_checkpointing: true`
* Try QLoRA instead of LoRA

### Poor Training Performance
* Check learning rate (try 5e-5 to 1e-4 for LoRA)
* Verify data quality with `render_template.py`
* Monitor loss curves in dashboard

### Template Issues
* Test templates with small samples first
* Check for proper tokenization boundaries
* Ensure system/user/assistant roles are clear

## License

MIT License - feel free to use for commercial projects.

## Contributing

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

Built on top of:
* [Transformers](https://github.com/huggingface/transformers)
* [PEFT](https://github.com/huggingface/peft)
* [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
* [Unsloth](https://github.com/unslothai/unsloth)
