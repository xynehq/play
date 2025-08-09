.PHONY: help install process style render train train-with-tb stop-tb tensorboard eval eval-test eval-val eval-quick eval-full infer infer-batch infer-interactive merge merge-bf16 merge-test check clean setup-dirs download-model

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
	@echo "  merge         Merge LoRA adapters to FP16 model"
	@echo "  merge-bf16    Merge LoRA adapters to BF16 model"
	@echo "  merge-test    Test merged model loading"
	@echo ""
	@echo "Utilities:"
	@echo "  check         Validate project setup before training"
	@echo "  clean         Clean generated files"
	@echo "  full-pipeline Run complete data processing pipeline"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=path   Specify config file (default: configs/config_run.yaml)"
	@echo "  STYLE=text    Specify style prompt for style command"

.PHONY: download-model
MODEL?=Qwen/Qwen2.5-3B-Instruct
LOCAL?=models/$(shell echo $(MODEL) | sed 's/[\/:@#]\+/-/g')

download-model:
	@python -c "from huggingface_hub import snapshot_download; import os; \
token=os.getenv('HUGGINGFACE_HUB_TOKEN', None); \
snapshot_download(repo_id='$(MODEL)', local_dir='$(LOCAL)', local_dir_use_symlinks=False, token=token); \
print('Saved to: $(LOCAL)')"

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
	python scripts/merge_lora.py --config $(CONFIG) --adapters adapters/last --out outputs/merged_fp16 --dtype fp16

merge-bf16:
	@echo "Merging LoRA adapters (bfloat16)..."
	python scripts/merge_lora.py --config $(CONFIG) --adapters adapters/last --out outputs/merged_bf16 --dtype bf16

merge-test:
	@echo "Testing merged model..."
	@if [ ! -d outputs/merged_fp16 ]; then \
		echo "Error: Merged model not found. Run 'make merge' first."; \
		exit 1; \
	fi
	@echo "Testing merged model loading..."
	@python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		print('Loading merged model...'); \
		m = AutoModelForCausalLM.from_pretrained('outputs/merged_fp16', torch_dtype='auto', device_map='auto'); \
		t = AutoTokenizer.from_pretrained('outputs/merged_fp16', use_fast=True); \
		print('✓ Model loaded successfully!'); \
		print(f'Model type: {m.config.model_type}'); \
		print(f'Pad token: {t.pad_token}'); \
		print('✓ Merged model is ready for deployment!')"

check:
	@echo "🔍 SFT-Play Sanity Check"
	@echo "========================"
	@echo ""
	@echo "Checking project setup..."
	@echo ""
	@# Check config file exists
	@if [ ! -f $(CONFIG) ]; then \
		echo "❌ Config file not found: $(CONFIG)"; \
		exit 1; \
	else \
		echo "✅ Config file found: $(CONFIG)"; \
	fi
	@echo ""
	@# Check processed data exists
	@if [ ! -f data/processed/train.jsonl ]; then \
		echo "❌ Training data not found: data/processed/train.jsonl"; \
		echo "   Run 'make process' or './workflows/quick_start.sh' first"; \
		exit 1; \
	else \
		echo "✅ Training data found: data/processed/train.jsonl"; \
		@wc -l data/processed/train.jsonl | awk '{print "   📊 Training samples: " $$1}'; \
	fi
	@echo ""
	@# Check validation data
	@if [ -f data/processed/val.jsonl ]; then \
		echo "✅ Validation data found: data/processed/val.jsonl"; \
		@wc -l data/processed/val.jsonl | awk '{print "   📊 Validation samples: " $$1}'; \
	else \
		echo "⚠️  Validation data not found: data/processed/val.jsonl"; \
	fi
	@echo ""
	@# Check test data
	@if [ -f data/processed/test.jsonl ]; then \
		echo "✅ Test data found: data/processed/test.jsonl"; \
		@wc -l data/processed/test.jsonl | awk '{print "   📊 Test samples: " $$1}'; \
	else \
		echo "⚠️  Test data not found: data/processed/test.jsonl"; \
	fi
	@echo ""
	@# Check template file
	@if [ -f chat_templates/default.jinja ]; then \
		echo "✅ Chat template found: chat_templates/default.jinja"; \
	else \
		echo "❌ Chat template not found: chat_templates/default.jinja"; \
		exit 1; \
	fi
	@echo ""
	@# Check directories
	@if [ -d outputs ] && [ -d adapters ]; then \
		echo "✅ Output directories ready"; \
	else \
		echo "⚠️  Output directories missing - run 'make setup-dirs'"; \
	fi
	@echo ""
	@echo "🎉 Sanity check completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  • Run 'make train' to start training"
	@echo "  • Run 'make train-with-tb' for training with TensorBoard"
	@echo "  • Run 'make help' to see all available commands"

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
	@echo "  2. Run 'make check' to validate setup"
	@echo "  3. Run 'make train' to start training"
	@echo "  4. Run 'make eval' to evaluate the model"
