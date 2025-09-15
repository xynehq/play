# DPIP-QnA (Ollama x 3 endpoints)

A lean, scalable pipeline to generate **Q–A pairs from DPIP docs** (PDF/DOCX) with **two generation endpoints** (fallback or round‑robin) and **one judge endpoint**. 
Prompts are **Python modules** for easy swapping via config.

## Endpoints & Models (pre-wired)
- **Gen A**: `Add your endpoints link` → **qwen3:32b** (primary)
- **Gen B**: `Add your endpoints link` → **gemma3:27b** (secondary)
- **Judge**: `Add your endpoints link` → **hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M**

> You can switch routing to round‑robin, or swap models in `configs/config_run.yaml`.

## Quickstart
```bash
# 1) Create venv & install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Drop your PDF/DOCX into:
#    data/raw_docs/

# 3) Run the pipeline
make prep
make gen
make v1
make v2
make export

# Final SFT dataset: data/sft/train.jsonl
```

## Folder Layout
```
PLAY/  
├─ data-generation/            # QnA data generation pipeline
│  ├─ configs/                # Generation configs
│  ├─ prompts/                # Python prompt modules
│  ├─ qna_core/               # Core modules (loaders, chunking, validation)
│  ├─ scripts/                # Generation scripts
│  ├─ data/
│  │  ├─ generated/
│  │  ├─ processed/
│  │  └─ raw_docs/            # Put PDF/DOCX here
│  └─ generated-data/
│     └─ qna_high_confidence_validated.jsonl  # 1001 samples
├─ scripts/                    # SFT training scripts
│  ├─ train.py                # Main training script
│  ├─ eval.py
│  ├─ infer.py
│  └─ utils/
└─ requirements.txt            # Python dependencies
```

## Notes
- Answers are **forced to include citations** (chunk_ids). Items without valid citations are rejected.
- Step‑1 validator checks lexical overlap, semantic similarity, and uncited-sentence ratio.
- Step‑2 judge scores faithfulness/completeness/specificity/style and citations_valid (0/1).
- Retrieval defaults to **TF‑IDF** for simplicity (swap with Vespa later).
