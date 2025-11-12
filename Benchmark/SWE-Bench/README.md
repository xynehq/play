# SWE-bench Automation Script

A one-step Python script to automate the complete SWE-bench evaluation pipeline - from setup to results. This script simplifies the process of running software engineering benchmarks on code generation models.

---

## 🎯 What is This Script?

This script (`swe.py`) automates the entire **SWE-bench** (Software Engineering Benchmark) evaluation workflow. SWE-bench is a benchmark that evaluates AI models' ability to solve real-world GitHub issues by generating code patches.

### What the Script Does:

1. 🔍 **Prerequisites Check**: Validates system requirements before starting (Docker, Python, disk space, RAM)
2. ✅ **Automatic Setup**: Installs Python 3.10-3.12, creates virtual environment, installs all dependencies
3. 🤖 **Model Inference**: Generates code patches using either:
   - API-based models (OpenAI, Anthropic, or custom APIs)
   - Local models (Llama, CodeLlama, etc.)
   - Existing prediction files
4. 🐳 **Docker Environment**: Builds isolated Docker containers for each test instance
5. 🧪 **Evaluation**: Runs test suites to validate generated patches
6. 📊 **Results**: Produces detailed evaluation metrics and success rates


## 📦 Prerequisites

### Required:
- **Docker**: Docker Desktop or OrbStack (**MUST be installed and running**)
  - The script checks Docker at startup
  - Provides OS-specific installation instructions if missing

### Automatically Handled:
- **Python**: 3.10-3.12 (script finds or installs automatically)
- **Dependencies**: All packages installed automatically
- **Virtual Environment**: Created automatically
- **SWE-bench Repository**: Cloned automatically

### Quick Docker Check:

```bash
# Verify Docker is installed and running
docker info
```

**If Docker is missing**, the script will show installation options for your OS and exit.
**If Docker is not running**, you can choose to continue (predictions only, no evaluation).

---

## 🚀 Quick Start

```bash
# 1. Download the script
cd ~/Desktop
mkdir swe && cd swe
# Place swe.py in this directory

# 2. Run the script
python3 swe.py
```

⚠️ **Note**: Docker must be installed and running. The script will:
- Check Docker immediately at startup
- Show OS-specific installation instructions if Docker is missing
- Allow you to continue without Docker (predictions only, no evaluation)


## 🛠️ Setup Instructions

### Step 1: First Run

Simply run the script:

```bash
python3 swe.py
```

The script will automatically:
1. ✅ Check Docker (must be installed and running)
2. ✅ Find or install Python 3.10-3.12
3. ✅ Create virtual environment (`venv_swebench/`)
4. ✅ Install all required packages (torch, transformers, datasets, etc.)
5. ✅ Clone SWE-bench repository

### Step 2: API Configuration (Optional)

During first run, you'll be prompted to configure API settings:

```
Do you want to configure API settings now? (y/n, default: y): y
```

**For API-based inference**, provide:
- API Base URL (e.g., `https://api.openai.com/v1` or your custom endpoint)
- API Key
- Model Name (e.g., `gpt-4`, `claude-3-sonnet`, or your custom model)
- Temperature (default: 0.2)
- Max Tokens (default: 1000, recommended: 500-2000)
- Number of Instances (optional, will be asked each time if empty)

Configuration is saved to `.env` file and can be edited anytime.

### Step 3: Choose Dataset

Select one of three benchmark datasets:

1. **SWE-bench Lite** (300 instances) - **Recommended for testing**
2. **SWE-bench Verified** (500 instances) - Verified test cases
3. **SWE-bench Full** (2,294 instances) - Complete benchmark (⚠️ takes hours)

---

## 📖 Script Usage

The script supports **three evaluation modes**. Run `python3 swe.py` and choose:

| Mode | Choice | Description | Best For |
|------|--------|-------------|----------|
| **Local Model** | `1` | Use local HuggingFace model | GPU access, offline evaluation |
| **API Inference** | `2` | Use OpenAI/Anthropic/Custom API | Quick testing, cloud models |
| **Existing File** | `3` | Evaluate pre-generated predictions | Reproducibility, debugging |

---

### Mode 1: Local Model Inference

Use your own locally stored models (HuggingFace format).

**Requirements:**
- Local model files (e.g., Llama, CodeLlama)
- CUDA GPU (16GB+ VRAM recommended)
- 32GB+ RAM

**Usage:**
```bash
python3 swe.py
# Choose: 1
# Enter model path: ./models/codellama-13b
# Enter number of instances: 10
```

⚠️ **Note**: Resource-intensive and may take hours for large datasets.

---

### Mode 2: API-Based Inference (Recommended)

Use cloud APIs for fast, scalable inference.

**Supported Providers:**
- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`
- **Custom APIs**: Any OpenAI-compatible endpoint

**Usage:**
```bash
python3 swe.py
# Choose: 2
# Script uses .env configuration automatically
```

**Configuration** (`.env` file):
```bash
API_BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-api-key-here
MODEL_NAME=gpt-4
TEMPERATURE=0.2
MAX_TOKENS=1000
NUM_INSTANCES=10  # Limit instances for testing
```

**Example Output:**
```
🌐 API Model Inference
⚙️ Using Model from .env: gpt-4
🤖 Processing 10 instances...
Inference: 100%|██████| 10/10 [00:30<00:00, 3.0s/it]
✅ Done!
```

---

### Mode 3: Existing Predictions File

Evaluate predictions you've already generated (skip inference).

**Usage:**
```bash
python3 swe.py
# Choose: 3
# Enter file path: ./predictions/my_results.jsonl
```

**File Format (JSONL):**
```jsonl
{"instance_id": "django__django-12856", "model_patch": "diff --git a/file.py...", "model_name_or_path": "my-model"}
{"instance_id": "sympy__sympy-18532", "model_patch": "diff --git a/file.py...", "model_name_or_path": "my-model"}
```

**File Format (JSON):**
```json
[
  {
    "instance_id": "django__django-12856",
    "model_patch": "diff --git a/django/file.py b/django/file.py\n...",
    "model_name_or_path": "my-model"
  }
]
```

---

## ⚙️ Configuration

### Environment Variables (`.env` file)

Create or edit `.env` in the script directory:

```bash
# API Configuration
API_BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-api-key-here
MODEL_NAME=gpt-4

# Inference Parameters
TEMPERATURE=0.2
MAX_TOKENS=1000
NUM_INSTANCES=10

# Optional: Leave empty to be prompted each time
# NUM_INSTANCES=
```

### Manual Configuration:

Edit `.env` file directly:

```bash
nano .env
```

Or delete `.env` to be prompted again during next run.

---

## 📊 Input/Output Format

### Input: GitHub Issues

SWE-bench instances contain:
- **Problem Statement**: Issue description from GitHub
- **Repository**: Source code context
- **Base Commit**: Git commit to apply patch to
- **Test Suite**: Tests to validate the solution

### Output: Model Predictions

Generated predictions are saved in **JSONL format**:

**File**: `SWE-bench/<model_name>__<dataset>__test.jsonl`

```jsonl
{
  "instance_id": "django__django-12856",
  "model_name_or_path": "your-model-name",
  "model_patch": "diff --git a/django/contrib/auth/forms.py b/django/contrib/auth/forms.py\nindex abc123..def456 100644\n--- a/django/contrib/auth/forms.py\n+++ b/django/contrib/auth/forms.py\n@@ -100,7 +100,7 @@ class UserCreationForm(forms.ModelForm):\n-        return user\n+        return self.save(commit=False)"
}
```

**Fields:**
- `instance_id`: Unique identifier for the GitHub issue
- `model_name_or_path`: Model used for generation
- `model_patch`: Git diff patch in unified format

---

## 📁 Results Location

After running the script, you'll see output like:

```
================================================================================
✅ SWE-bench Evaluation Complete!
================================================================================
🏆 Results file: /full/path/to/SWE-bench/your-model-name.run_20251112_170834.json
📂 Logs directory: /full/path/to/SWE-bench/logs/run_evaluation/run_20251112_170834
📄 Predictions file: /full/path/to/SWE-bench/your-model-name__SWE-bench_Lite__test.jsonl
```

**All paths are displayed as absolute paths** for easy access - you can click them directly in most terminals!

---

## 📚 Benchmark Details

### SWE-bench Overview

**SWE-bench** (Software Engineering Benchmark) evaluates models on real-world software engineering tasks:

- **Source**: Real GitHub issues from popular Python repositories
- **Task**: Generate code patches that fix the issues
- **Evaluation**: Patches are applied and tested against the actual test suites
- **Metric**: Pass@1 (percentage of instances where generated patch passes tests)

### Dataset Variants:

| Dataset | Instances | Description | Recommended For |
|---------|-----------|-------------|-----------------|
| **SWE-bench Lite** | 300 | Curated subset, balanced difficulty | Testing, quick evaluation |
| **SWE-bench Verified** | 500 | Human-verified test cases | High-quality evaluation |
| **SWE-bench Full** | 2,294 | Complete benchmark | Research, comprehensive eval |

### Repositories Included:

- Django (web framework)
- SymPy (symbolic mathematics)
- Matplotlib (plotting library)
- Pytest (testing framework)
- Scikit-learn (machine learning)
- Flask (web framework)
- And 20+ more popular Python projects

### Evaluation Process:

1. **Patch Application**: Generated patch is applied to the base commit
2. **Environment Setup**: Dependencies installed in isolated Docker container
3. **Test Execution**: Full test suite runs against patched code
4. **Result**: Pass/Fail based on test outcomes

### Success Criteria:

✅ **Resolved**: All tests pass after applying patch
❌ **Failed**: Any test fails or patch cannot be applied

---

## 🐛 Troubleshooting

### Quick Issue Index:

1. [Docker Not Running](#1-docker-not-running--critical) ⚠️ Critical
2. [Python Version Issues](#2-python-version-issues)
3. [API Rate Limits](#3-api-rate-limits)
4. [Out of Memory (Local Model)](#4-out-of-memory-local-model)
5. [Docker Image Build Fails (Exit Code 137)](#5-docker-image-build-fails-exit-code-137) - Memory issue
6. [Other Docker Image Build Failures](#6-other-docker-image-build-failures)
7. [JSON Parsing Error](#7-json-parsing-error-existing-predictions-file) - Empty lines in file
8. [File Paths Not Clickable](#8-file-paths-not-clickable--files-not-opening) - Fixed!
9. [Malformed Patch Errors](#9-malformed-patch-errors) - Model quality issue

---

### Common Issues:

#### 1. Docker Not Running ⚠️ CRITICAL

```
❌ Docker not detected or not running. Please start Docker Desktop.
```

**When this happens**: The script detects this **immediately** during prerequisites check (before any setup)

**Solution**: 
1. Install Docker Desktop (https://www.docker.com/products/docker-desktop) or OrbStack (macOS)
2. Start Docker and ensure it's running
3. Verify with: `docker info`
4. Re-run the script

**Note**: The script now checks Docker **at the very beginning** (Step 0) and shows OS-specific installation instructions if missing.

#### 2. Python Version Issues

```
⚠️ Python 3.9 is too old. Minimum required: 3.10
```

**Solution**: Script will automatically detect Python 3.10-3.12. Install if needed:

```bash
# macOS
brew install python@3.12

# Ubuntu/Debian
sudo apt install python3.12 python3.12-venv
```

#### 3. API Rate Limits

```
Rate limit exceeded. Waiting 60 seconds...
```

**Solution**: 
- Reduce `NUM_INSTANCES` in `.env`
- Use a different API provider
- Wait and retry

#### 4. Out of Memory (Local Model)

```
CUDA out of memory
```

**Solution**:
- Use smaller model
- Reduce batch size
- Use API-based inference instead

#### 5. Docker Image Build Fails (Exit Code 137)

```
BuildImageError: The command returned a non-zero code: 137
```

**This is a memory issue!** Exit code 137 = Docker Out of Memory (OOM)

**Solution**:
- **Increase Docker memory**: 
  - Docker Desktop: Settings → Resources → Memory → Increase to 8GB+ (16GB recommended)
  - OrbStack: Settings → Resources → Memory → Increase to 8GB+
- **Reduce parallel workers**: Edit `swe.py` line with `--max_workers` and change `"8"` to `"2"` or `"4"`
- **Process fewer instances**: Set `NUM_INSTANCES=5` in `.env` to test with smaller batches
- **Restart Docker**: Sometimes helps clear memory
- **Check available RAM**: Ensure your system has 16GB+ RAM

#### 6. Other Docker Image Build Failures

```
Error building Docker image for instance...
```

**Solution**:
- Check internet connection
- Ensure Docker has enough disk space (20GB+ free)
- Check `SWE-bench/logs/run_evaluation/run_*/` for detailed error logs
- Verify Docker daemon is running properly

#### 7. JSON Parsing Error (Existing Predictions File)

```
json.decoder.JSONDecodeError: Expecting value: line 2 column 1 (char 1)
```

**Cause**: Previous run created a predictions file with empty lines or incomplete entries

**Solution** (Automatic!):
The script now **automatically detects** this issue and offers options:
```
⚠️  Existing predictions file found: your-model-name__SWE-bench_Lite__test.jsonl
   This may cause issues if the file has incomplete entries.
   ⚠️  File has 2 empty line(s) - this will cause JSON parsing errors!
   
Options: [r]emove and start fresh, [k]eep and resume, [q]uit (default: r):
```

**Recommendation**: Choose **[r]** to remove and start fresh (safest option)

#### 8. File Paths Not Clickable / Files Not Opening

**Symptom**: Displayed paths have underscores (`my_model_v2`) but actual files have hyphens (`my-model-v2`)

**Solution**: ✅ **Fixed!** The script now:
- Keeps hyphens in model names (e.g., `my-model-v2` stays as-is)
- Shows **absolute paths** for all result files (clickable in most terminals)
- Correctly points to `logs/run_evaluation/run_ID/` (not `logs/run_ID/`)

**Example output** (after fix):
```
🏆 Results file: /Users/you/Desktop/swe/SWE-bench/your-model-name.run_20251112_170834.json
📂 Logs directory: /Users/you/Desktop/swe/SWE-bench/logs/run_evaluation/run_20251112_170834
📄 Predictions file: /Users/you/Desktop/swe/SWE-bench/your-model-name__SWE-bench_Lite__test.jsonl
```

All paths are now **correct and clickable**! 🎉

#### 9. Malformed Patch Errors

```
❌ Patch Apply Failed: malformed patch at line 29
```

**Cause**: Model generated an invalid patch (not in proper unified diff format)

**Symptoms**:
- Patch contains code comments or docstrings inside diff hunks
- Repeated function definitions
- Missing `diff --git` header
- Truncated or incomplete patches

**Solution**:
This is a **model quality issue**, not a script issue. The model needs better training on:
1. Unified diff format (`diff --git a/... b/...`)
2. Proper hunk markers (`@@ -X,Y +A,B @@`)
3. Only including changed lines with `+` or `-` prefixes

**Workarounds**:
- Use a different, better-trained model (GPT-4, Claude-3.5-Sonnet, DeepSeek-Coder)
- Fine-tune your model with proper diff examples
- Increase `MAX_TOKENS` if patches are being truncated
- Check `logs/run_evaluation/run_*/model-name/instance-id/patch.diff` to see the actual malformed patch

**Example of a good patch**:
```diff
diff --git a/src/file.py b/src/file.py
index abc123..def456 100644
--- a/src/file.py
+++ b/src/file.py
@@ -10,7 +10,7 @@ def function():
     old_line = "value"
-    return old_line
+    return "new_value"
```

**Example of a bad patch** (DO NOT generate):
```diff
diff --git a/src/file.py b/src/file.py
@@ -10,7 +10,7 @@
def function():
    """This is a docstring"""  # ❌ Should NOT be in diff!
    old_line = "value"
-    return old_line
+    return "new_value"
```

### Getting Help:

1. Check `SWE-bench/logs/run_evaluation/run_*/` for detailed error logs
2. View `docker_images.log` for Docker-related issues
3. Ensure all prerequisites are met
4. Try with fewer instances first (`NUM_INSTANCES=2`)
5. For patch errors, check the `patch.diff` file in the instance log directory

