export PYTHONPATH := $(CURDIR)

prep:
	python3 scripts/00_prepare_corpus.py --config configs/config_run.yaml

gen:
	python3 scripts/10_generate_qna.py --config configs/config_run.yaml

clean:
	python3 scripts/clean_qna.py --input data/generated/qna_raw.jsonl --output data/generated/qna_cleaned.jsonl

v1:
	python3 scripts/20_validate_citations.py --config configs/config_run.yaml

v2:
	python3 scripts/30_judge_eval.py --config configs/config_run.yaml

export:
	python3 scripts/40_export_sft.py --config configs/config_run.yaml

all: prep gen v1 v2 export
