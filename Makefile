.PHONY: help install process style render train train-with-tb stop-tb tensorboard tb tb-stop tb-clean tb-open train-and-watch eval eval-test eval-val eval-quick eval-full infer infer-batch infer-interactive merge merge-bf16 merge-test check clean setup-dirs download-model print-python dapt-docx dapt-train test test-unit test-integration test-coverage test-fast test-eval-optimization test-eval-core test-eval-performance test-eval-integration

# Python detection - use python3 if available, otherwise python
PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)

# Default config file
CONFIG ?= configs/run_bnb.yaml

# Style prompt (can be overridden)
STYLE ?= Answer concisely in 2 lines. No markdown. If unsure, say 'Not sure'.

# ---- TensorBoard config ----
TB_PORT ?= 6006
TB_LOGDIR ?= $(shell mkdir -p outputs/tb && realpath outputs/tb)

help:
	@echo "SFT-Play Makefile Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install dependencies (pip or uv)"
	@echo "  setup-dirs    Create necessary directories"
	@echo "  setup-accelerate Configure Accelerate for multi-GPU training"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  process       Process raw data to structured chat format"
	@echo "  style         Apply style/system prompts to processed data"
	@echo "  render        Render chat templates to seq2seq format"
	@echo ""
	@echo "Single-GPU Training:"
	@echo "  train         Start training with current config"
	@echo "  train-bnb     Start training with BitsAndBytes backend"
	@echo "  train-unsloth Start training with Unsloth backend"
	@echo "  train-with-tb Start training with TensorBoard monitoring"
	@echo "  train-bnb-tb  Start BitsAndBytes training with TensorBoard"
	@echo "  train-unsloth-tb Start Unsloth training with TensorBoard"
	@echo ""
	@echo "Multi-GPU Distributed Training:"
	@echo "  train-distributed    Distributed training (auto-detect GPUs)"
	@echo "  train-distributed-tb Distributed training with TensorBoard"
	@echo "  train-deepspeed      DeepSpeed distributed training"
	@echo "  train-deepspeed-tb   DeepSpeed training with TensorBoard"
	@echo ""
	@echo "Evaluation & Inference:"
	@echo "  eval          Evaluate trained model (validation set)"
	@echo "  eval-test     Evaluate on test set"
	@echo "  eval-val      Evaluate on validation set"
	@echo "  eval-quick    Quick evaluation (200 samples)"
	@echo "  eval-full     Full evaluation (no sample limit)"
	@echo "  infer         Interactive inference (chat mode)"
	@echo "  infer-batch   Batch inference from file"
	@echo "  infer-interactive Interactive inference (explicit)"
	@echo ""
	@echo "DAPT (Domain-Adaptive Pretraining):"
	@echo "  dapt-docx     Process DOCX files for DAPT CPT datasets"
	@echo "  dapt-train    Start DAPT training with mixed CPT + instruction data"
	@echo ""
	@echo "GPU Monitoring & Diagnostics:"
	@echo "  gpu-info      Show GPU information and CUDA details"
	@echo "  memory-check  Check GPU memory usage"
	@echo ""
	@echo "Monitoring:"
	@echo "  tensorboard   Start TensorBoard on outputs/tb"
	@echo "  tb-stop       Kill any running TensorBoard"
	@echo "  tb-clean      Remove TB event files"
	@echo "  tb-open       Print exact path & suggest URL"
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
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  test-fast     Run fast tests (excluding slow tests)"
	@echo "  test-commands Test all Makefile commands"
	@echo "  test-configs  Test configuration files"
	@echo ""
	@echo "Evaluation Optimization Testing:"
	@echo "  test-eval-optimization    Run all evaluation optimization tests"
	@echo "  test-eval-core            Run core evaluation optimization tests"
	@echo "  test-eval-callback        Run EvalSpeedCallback tests"
	@echo "  test-eval-compatibility   Run compatibility tests"
	@echo "  test-eval-trainer         Run trainer configuration tests"
	@echo "  test-eval-sync            Run post-evaluation synchronization tests"
	@echo "  test-eval-params          Run evaluation parameter optimization tests"
	@echo "  test-eval-config          Run configuration integration tests"
	@echo "  test-eval-integration     Run evaluation integration tests"
	@echo "  test-eval-performance     Run performance characteristics tests"
	@echo "  test-eval-coverage        Run evaluation optimization tests with coverage"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=path   Specify config file (default: configs/run_bnb.yaml)"
	@echo "  STYLE=text    Specify style prompt for style command"
	@echo ""
	@echo "Multi-GPU Examples:"
	@echo "  make train-distributed CONFIG=configs/run_bnb.yaml"
	@echo "  make train-deepspeed-tb CONFIG=configs/run_gemma27b_distributed.yaml"
	@echo "  make gpu-info  # Check your GPU setup"

print-python:
	@echo "which python: `which python`"
	@python -c "import sys; print(f'sys.executable: {sys.executable}')"

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
	$(PYTHON) scripts/process_data.py --config $(CONFIG)

style:
	@echo "Applying style prompts..."
	@if [ ! -f data/processed/train.jsonl ]; then \
		echo "Error: data/processed/train.jsonl not found. Run 'make process' first."; \
		exit 1; \
	fi
	$(PYTHON) scripts/style_prompt.py \
		--config $(CONFIG) \
		--style "$(STYLE)" \
		--in data/processed/train.jsonl \
		--out data/processed_with_style/train.jsonl \
		--mode prepend
	@for split in val test; do \
		if [ -f data/processed/$$split.jsonl ]; then \
			echo "Processing $$split split..."; \
			$(PYTHON) scripts/style_prompt.py \
				--config $(CONFIG) \
				--style "$(STYLE)" \
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
	$(PYTHON) scripts/render_template.py \
		--config $(CONFIG) \
		--in data/processed/train.jsonl \
		--out data/rendered/train.jsonl
	@for split in val test; do \
		if [ -f data/processed/$$split.jsonl ]; then \
			echo "Rendering $$split split..."; \
			$(PYTHON) scripts/render_template.py \
				--config $(CONFIG) \
				--in data/processed/$$split.jsonl \
				--out data/rendered/$$split.jsonl; \
		fi; \
	done

train:
	@echo "Starting training..."
	PYTHONPATH=. $(PYTHON) scripts/train.py --config $(CONFIG)

train-bnb:
	@echo "Starting training with BitsAndBytes backend..."
	PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/run_bnb.yaml

train-unsloth:
	@echo "Starting training with Unsloth backend (XFormers disabled)..."
	XFORMERS_DISABLED=1 UNSLOTH_DISABLE_FAST_ATTENTION=1 PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/run_unsloth.yaml

## train-with-tb: Train + print how to launch TB
train-with-tb:
	@echo "Starting training…"
	PYTHONPATH=. $(PYTHON) scripts/train.py --config $(CONFIG)
	@echo ""
	@echo "✅ Training finished. To view logs:"
	@echo "   make tensorboard TB_PORT=$(TB_PORT)"

train-bnb-tb:
	@echo "Starting BitsAndBytes training with TensorBoard..."
	@mkdir -p outputs/tb
	@echo "📈 Starting TensorBoard at http://localhost:$(TB_PORT)"
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
	@sleep 3
	@echo "📈 TensorBoard should be running at http://localhost:$(TB_PORT)"
	PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/run_bnb.yaml
	@echo ""
	@echo "✅ Training finished. TensorBoard may still be running at:"
	@echo "   http://localhost:$(TB_PORT)"
	@echo "   To stop TensorBoard: make tb-stop"

train-unsloth-tb:
	@echo "Starting Unsloth training with TensorBoard..."
	@mkdir -p outputs/tb
	@pgrep -f "tensorboard.*$(TB_PORT)" | xargs -r kill || true
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
	@sleep 2
	@echo "📈 TensorBoard started at http://localhost:$(TB_PORT)"
	XFORMERS_DISABLED=1 UNSLOTH_DISABLE_FAST_ATTENTION=1 PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/run_unsloth.yaml
	@echo ""
	@echo "✅ Training finished. TensorBoard is still running at:"
	@echo "   http://localhost:$(TB_PORT)"
	@echo "   To stop TensorBoard: make tb-stop"

## train-and-watch: Start TB (bg) then train
train-and-watch:
	@mkdir -p outputs/tb
	@pgrep -f "tensorboard.*$(TB_PORT)" | xargs -r kill || true
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
	@sleep 2
	@echo "📈 TensorBoard at http://localhost:$(TB_PORT)"
	PYTHONPATH=. $(PYTHON) scripts/train.py --config $(CONFIG)

## tensorboard: Start TensorBoard on outputs/tb (override TB_PORT=6007 if needed)
tensorboard tb:
	@if [ ! -d "outputs/tb" ]; then mkdir -p outputs/tb; fi
	@echo "👉 Launching TensorBoard at http://localhost:$(TB_PORT) (logdir=$(TB_LOGDIR))"
	@tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0

## tb-stop: Kill any running TensorBoard
tb-stop:
	@pkill -f tensorboard || true
	@echo "✅ Stopped any running TensorBoard"

## tb-clean: Remove TB event files
tb-clean:
	@rm -rf outputs/tb
	@mkdir -p outputs/tb
	@echo "🧹 Cleaned outputs/tb"

## tb-open: Print exact path & suggest URL
tb-open:
	@echo "Logdir: $(TB_LOGDIR)"
	@echo "Visit:  http://localhost:$(TB_PORT)"

eval:
	@echo "Running evaluation on validation set..."
	PYTHONPATH=. $(PYTHON) scripts/eval.py --config $(CONFIG) --split val

eval-test:
	@echo "Running evaluation on test set..."
	PYTHONPATH=. $(PYTHON) scripts/eval.py --config $(CONFIG) --split test

eval-val:
	@echo "Running evaluation on validation set..."
	PYTHONPATH=. $(PYTHON) scripts/eval.py --config $(CONFIG) --split val

eval-quick:
	@echo "Running quick evaluation (200 samples)..."
	PYTHONPATH=. $(PYTHON) scripts/eval.py --config $(CONFIG) --split val --limit 200

eval-full:
	@echo "Running full evaluation (no limit)..."
	PYTHONPATH=. $(PYTHON) scripts/eval.py --config $(CONFIG) --split val --limit 0

infer:
	@echo "Starting interactive inference..."
	@echo "Type your questions. Press Ctrl+C to exit."
	PYTHONPATH=. python scripts/infer.py --config $(CONFIG) --mode interactive

infer-batch:
	@echo "Running batch inference..."
	@if [ ! -f demo_inputs.txt ]; then \
		echo "Creating demo_inputs.txt..."; \
		echo "What is machine learning?" > demo_inputs.txt; \
		echo "Explain neural networks briefly." >> demo_inputs.txt; \
		echo "What is QLoRA?" >> demo_inputs.txt; \
	fi
	PYTHONPATH=. python scripts/infer.py --config $(CONFIG) --mode batch --input_file demo_inputs.txt --output_file outputs/preds.txt
	@echo "Results saved to outputs/preds.txt"

infer-interactive:
	@echo "Starting interactive inference..."
	@echo "Type your questions. Press Ctrl+C to exit."
	PYTHONPATH=. python scripts/infer.py --config $(CONFIG) --mode interactive

merge:
	@echo "Merging LoRA adapters..."
	PYTHONPATH=. python scripts/merge_lora.py --config $(CONFIG) --adapters adapters/last --out outputs/merged_fp16 --dtype fp16

merge-bf16:
	@echo "Merging LoRA adapters (bfloat16)..."
	PYTHONPATH=. python scripts/merge_lora.py --config $(CONFIG) --adapters adapters/last --out outputs/merged_bf16 --dtype bf16

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

full-pipeline: setup-dirs
	@echo "Creating necessary directories"
	@$(MAKE) process
	@$(MAKE) style  
	@$(MAKE) render
	@echo "Full data processing pipeline completed"
	@echo "Next steps:"
	@echo "  1. Review processed data in data/processed_with_style/"
	@echo "  2. Run 'make check' to validate setup"
	@echo "  3. Run 'make train' to start training"
	@echo "  4. Run 'make eval' to evaluate the model"

# DAPT (Domain-Adaptive Pretraining) targets
dapt-docx:
	@echo "Processing DOCX files for DAPT..."
	$(PYTHON) scripts/ingest_docx.py

dapt-train:
	@echo "Starting DAPT training..."
	PYTHONPATH=. $(PYTHON) scripts/train.py --config configs/run_dapt.yaml

# Multi-GPU Distributed Training (Generic for any model/config)
train-distributed:
	@echo "Starting distributed training with config: $(CONFIG)"
	@echo "Auto-detecting available GPUs..."
	PYTHONPATH=. accelerate launch --multi_gpu scripts/train_distributed.py --config $(CONFIG)

train-distributed-tb:
	@echo "Starting distributed training with TensorBoard monitoring..."
	@mkdir -p outputs/tb
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
	@sleep 2
	@echo "📈 TensorBoard started at http://localhost:$(TB_PORT)"
	PYTHONPATH=. accelerate launch --multi_gpu scripts/train_distributed.py --config $(CONFIG)
	@echo ""
	@echo "✅ Distributed training finished. TensorBoard is still running at:"
	@echo "   http://localhost:$(TB_PORT)"
	@echo "   To stop TensorBoard: make tb-stop"

train-deepspeed:
	@echo "Starting DeepSpeed distributed training with config: $(CONFIG)"
	PYTHONPATH=. accelerate launch --multi_gpu --use_deepspeed scripts/train_distributed.py --config $(CONFIG) --deepspeed --deepspeed_config configs/deepspeed_z2.json

train-deepspeed-tb:
	@echo "Starting DeepSpeed training with TensorBoard monitoring..."
	@mkdir -p outputs/tb
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
	@sleep 2
	@echo "📈 TensorBoard started at http://localhost:$(TB_PORT)"
	PYTHONPATH=. accelerate launch --multi_gpu --use_deepspeed scripts/train_distributed.py --config $(CONFIG) --deepspeed --deepspeed_config configs/deepspeed_z2.json
	@echo ""
	@echo "✅ DeepSpeed training finished. TensorBoard is still running at:"
	@echo "   http://localhost:$(TB_PORT)"
	@echo "   To stop TensorBoard: make tb-stop"

# GPU monitoring and diagnostics
gpu-info:
	@echo "GPU Information:"
	@nvidia-smi
	@echo ""
	@echo "CUDA Version:"
	@nvcc --version || echo "NVCC not found"
	@echo ""
	@echo "PyTorch CUDA Info:"
	@python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

memory-check:
	@echo "GPU Memory Usage:"
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Accelerate configuration
setup-accelerate:
	@echo "Setting up Accelerate for multi-GPU training..."
	@accelerate config

# Testing targets
test:
	@echo "Running all tests..."
	pytest

test-unit:
	@echo "Running unit tests..."
	pytest -m unit

test-integration:
	@echo "Running integration tests..."
	pytest -m integration

test-coverage:
	@echo "Running tests with coverage report..."
	pytest --cov=scripts --cov-report=html --cov-report=term-missing

test-fast:
	@echo "Running fast tests (excluding slow tests)..."
	pytest -m "not slow"

test-commands:
	@echo "Testing all Makefile commands..."
	pytest tests/test_commands.py -v

test-configs:
	@echo "Testing configuration files..."
	pytest tests/test_configs.py -v

# Evaluation Optimization Testing Targets
test-eval-optimization:
	@echo "Running all evaluation optimization tests..."
	pytest tests/test_evaluation_optimization.py -v

test-eval-core:
	@echo "Running core evaluation optimization tests..."
	pytest tests/test_evaluation_optimization.py::TestEvaluationOptimization -v

test-eval-callback:
	@echo "Running EvalSpeedCallback tests..."
	pytest tests/test_evaluation_optimization.py::TestEvalSpeedCallback -v

test-eval-compatibility:
	@echo "Running compatibility tests..."
	pytest tests/test_evaluation_optimization.py::TestTrainingArgumentsCompatibility -v

test-eval-trainer:
	@echo "Running trainer configuration tests..."
	pytest tests/test_evaluation_optimization.py::TestTrainerConfiguration -v

test-eval-sync:
	@echo "Running post-evaluation synchronization tests..."
	pytest tests/test_evaluation_optimization.py::TestPostEvaluationSynchronization -v

test-eval-params:
	@echo "Running evaluation parameter optimization tests..."
	pytest tests/test_evaluation_optimization.py::TestEvaluationParameterOptimization -v

test-eval-config:
	@echo "Running configuration integration tests..."
	pytest tests/test_evaluation_optimization.py::TestConfigurationIntegration -v

test-eval-integration:
	@echo "Running evaluation integration tests..."
	pytest tests/test_evaluation_optimization.py::TestEvaluationOptimizationIntegration -v

test-eval-performance:
	@echo "Running performance characteristics tests..."
	pytest tests/test_evaluation_optimization.py::TestPerformanceCharacteristics -v

test-eval-coverage:
	@echo "Running evaluation optimization tests with coverage..."
	pytest tests/test_evaluation_optimization.py --cov=scripts.train_distributed --cov-report=html --cov-report=term-missing
