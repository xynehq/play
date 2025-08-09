.PHONY: help install process style render train train-with-tb stop-tb tensorboard eval eval-test eval-val eval-quick eval-full infer infer-batch infer-interactive merge clean setup-dirs

# Default config file
CONFIG ?= configs/config_run.yaml

# Style prompt (can be overridden)
STYLE ?= "Answer concisely in 2 lines. No markdown. If unsure, say 'Not sure'."

help:
	@echo "SFT-Play Makefile Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install dependencies (pip or uv)"
	@echo "  setup-dirs    Create necessary directories"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  process       Process raw data to structured chat format"
	@echo "  style         Apply style/system prompts to processed data"
	@echo "  render        Render chat templates to seq2seq format"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  train         Start training with current config"
	@echo "  train-with-tb Start training with TensorBoard monitoring"
	@echo "  eval          Evaluate trained model (validation set)"
	@echo "  eval-test     Evaluate on test set"
	@echo "  eval-val      Evaluate on validation set"
	@echo "  eval-quick    Quick evaluation (200 samples)"
	@echo "  eval-full     Full evaluation (no sample limit)"
	@echo "  infer         Interactive inference (chat mode)"
	@echo "  infer-batch   Batch inference from file"
	@echo "  infer-interactive Interactive inference (explicit)"
	@echo ""
	@echo "Monitoring:"
	@echo "  tensorboard   Start TensorBoard (manual)"
	@echo "  stop-tb       Stop background TensorBoard"
	@echo ""
	@echo "Model Management:"
	@echo "  merge         Merge LoRA adapters to single model"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         Clean generated files"
	@echo "  full-pipeline Run complete data processing pipeline"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=path   Specify config file (default: configs/config_run.yaml)"
	@echo "  STYLE=text    Specify style prompt for style command"

install:
	@echo "Installing dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for installation..."; \
		uv venv --python 3.10; \
		uv pip install -e .; \
	else \
		echo "Using pip for installation..."; \
		pip install -r requirements.txt; \
	fi

setup-dirs:
	@echo "Creating necessary directories..."
	@mkdir -p data/raw data/processed data/processed_with_style data/rendered
	@mkdir -p outputs adapters
	@touch data/raw/.gitkeep data/processed/.gitkeep data/processed_with_style/.gitkeep data/rendered/.gitkeep
	@touch outputs/.gitkeep adapters/.gitkeep
	@echo "Directories created successfully!"

process:
	@echo "Processing raw data..."
	python scripts/process_data.py --config $(CONFIG)

style:
	@echo "Applying style prompts..."
	@if [ ! -f data/processed/train.jsonl ]; then \
		echo "Error: data/processed/train.jsonl not found. Run 'make process' first."; \
		exit 1; \
	fi
	python scripts/style_prompt.py \
		--config $(CONFIG) \
		--style $(STYLE) \
		--in data/processed/train.jsonl \
		--out data/processed_with_style/train.jsonl \
		--mode prepend
	@for split in val test; do \
		if [ -f data/processed/$$split.jsonl ]; then \
			echo "Processing $$split split..."; \
			python scripts/style_prompt.py \
				--config $(CONFIG) \
				--style $(STYLE) \
				--in data/processed/$$split.jsonl \
				--out data/processed_with_style/$$split.jsonl \
				--mode prepend; \
		fi; \
	done

render:
	@echo "Rendering chat templates..."
	@if [ ! -f data/processed/train.jsonl ]; then \
		echo "Error: data/processed/train.jsonl not found. Run 'make process' first."; \
		exit 1; \
	fi
	python scripts/render_template.py \
		--config $(CONFIG) \
		--in data/processed/train.jsonl \
		--out data/rendered/train.jsonl
	@for split in val test; do \
		if [ -f data/processed/$$split.jsonl ]; then \
			echo "Rendering $$split split..."; \
			python scripts/render_template.py \
				--config $(CONFIG) \
				--in data/processed/$$split.jsonl \
				--out data/rendered/$$split.jsonl; \
		fi; \
	done

train:
	@echo "Starting training..."
	python scripts/train.py --config $(CONFIG)

train-with-tb:
	@echo "Starting training with TensorBoard..."
	@echo "TensorBoard will be available at http://localhost:6006"
	@echo "Starting TensorBoard in background..."
	@nohup tensorboard --logdir outputs/ --port 6006 > /dev/null 2>&1 & echo $$! > .tensorboard.pid
	@echo "Starting training..."
	python scripts/train.py --config $(CONFIG)
	@echo "Training completed. TensorBoard is still running."
	@echo "To stop TensorBoard: make stop-tb"

stop-tb:
	@if [ -f .tensorboard.pid ]; then \
		echo "Stopping TensorBoard..."; \
		kill `cat .tensorboard.pid` 2>/dev/null || true; \
		rm -f .tensorboard.pid; \
		echo "TensorBoard stopped."; \
	else \
		echo "TensorBoard PID file not found."; \
	fi

tensorboard:
	@echo "Starting TensorBoard..."
	@echo "TensorBoard will be available at http://localhost:6006"
	tensorboard --logdir outputs/ --port 6006

eval:
	@echo "Running evaluation on validation set..."
	python scripts/eval.py --config $(CONFIG) --split val

eval-test:
	@echo "Running evaluation on test set..."
	python scripts/eval.py --config $(CONFIG) --split test

eval-val:
	@echo "Running evaluation on validation set..."
	python scripts/eval.py --config $(CONFIG) --split val

eval-quick:
	@echo "Running quick evaluation (200 samples)..."
	python scripts/eval.py --config $(CONFIG) --split val --limit 200

eval-full:
	@echo "Running full evaluation (no limit)..."
	python scripts/eval.py --config $(CONFIG) --split val --limit 0

infer:
	@echo "Starting interactive inference..."
	@echo "Type your questions. Press Ctrl+C to exit."
	python scripts/infer.py --config $(CONFIG) --mode interactive

infer-batch:
	@echo "Running batch inference..."
	@if [ ! -f demo_inputs.txt ]; then \
		echo "Creating demo_inputs.txt..."; \
		echo "What is machine learning?" > demo_inputs.txt; \
		echo "Explain neural networks briefly." >> demo_inputs.txt; \
		echo "What is QLoRA?" >> demo_inputs.txt; \
	fi
	python scripts/infer.py --config $(CONFIG) --mode batch --input_file demo_inputs.txt --output_file outputs/preds.txt
	@echo "Results saved to outputs/preds.txt"

infer-interactive:
	@echo "Starting interactive inference..."
	@echo "Type your questions. Press Ctrl+C to exit."
	python scripts/infer.py --config $(CONFIG) --mode interactive

merge:
	@echo "Merging LoRA adapters..."
	python scripts/merge_lora.py --config $(CONFIG) --out outputs/merged_fp16

clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/* data/processed_with_style/* data/rendered/*
	rm -rf outputs/* adapters/*
	rm -f demo_inputs.txt
	@echo "Clean complete!"

full-pipeline: setup-dirs process style render
	@echo "Full data processing pipeline completed!"
	@echo "Next steps:"
	@echo "  1. Review processed data in data/processed_with_style/"
	@echo "  2. Run 'make train' to start training"
	@echo "  3. Run 'make eval' to evaluate the model"
