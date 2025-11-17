# RustEvo: Benchmarking LLMs on Rust API Evolution

## What is RustEvo?

**RustEvo** is a comprehensive benchmark designed to evaluate Large Language Models (LLMs) on their ability to work with evolving Rust APIs. As Rust libraries evolve, APIs change—signatures are modified, parameters are added or removed, return types change, and functions are deprecated. RustEvo tests whether LLMs can generate correct code that adapts to these API changes.

### Key Features

- **588 Real-World Tasks**: Curated from actual Rust API evolution across popular crates
- **Comprehensive Test Coverage**: Each task includes queries, function signatures, and test programs
- **Multiple Evaluation Scenarios**: Tests models with varying levels of API documentation
- **Detailed Metrics**: Pass@1, API usage accuracy, borrow-checker error rates, and more
- **Automated Testing**: Built-in Rust compilation and test execution

### Why RustEvo?

Rust's unique features (ownership, borrowing, lifetimes) make API evolution particularly challenging. RustEvo helps:
- **Researchers**: Evaluate LLM performance on systems programming tasks
- **Tool Developers**: Benchmark code generation tools for Rust
- **Model Creators**: Assess and improve LLM understanding of Rust semantics

---

## Research Questions (RQ)

RustEvo evaluates models through different research questions:

### RQ1: Full Documentation Evaluation

**Goal**: Can LLMs generate correct code when provided with complete API documentation?

**What's Provided**:
- Complete API signature
- Full API documentation
- Source code implementation
- Version change information

**Use Case**: Simulates scenarios where developers have access to comprehensive API documentation and need to migrate code to new API versions.

---

### RQ3: Minimal Documentation Evaluation

**Goal**: Can LLMs generate correct code with minimal API information?

**What's Provided**:
- API name
- Module path only

**Use Case**: Simulates real-world scenarios where developers must work with limited documentation or infer API usage from context.

---

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows with WSL
- **Python**: 3.8 or higher
- **Rust**: Latest stable version (install via [rustup](https://rustup.rs/))

### Required Tools

1. **Install Rust and Cargo**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   rustc --version  # Verify installation
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install openai tqdm
   ```

## Quick Start

> **📝 Note: Configuring Model and API Credentials**
> 
> To use different models or API endpoints, you can either:
> 1. **Use environment variables** (recommended):
>    ```bash
>    export API_KEY="your-api-key"
>    export BASE_URL="https://your-endpoint.com"
>    ```
> 2. **Edit the evaluation scripts directly**:
>    - Open `Evaluate/eval_models_rq1.py`
>    - Open `Evaluate/eval_models_rq3.py`
>    - Modify these lines:
>      ```python
>      API_KEY = os.getenv('API_KEY', 'your-default-key-here')
>      BASE_URL = os.getenv('BASE_URL', 'https://your-default-endpoint.com')
>      MODELS = ["your-model-name"]  # Change default model
>      ```

---

### 🚀 Method 1: Automated Setup Script (Recommended)

**Choose Your Preferred Script:**
- **setup.py** - Python script (Cross-platform: macOS, Linux, Windows) - Plain text output
- **setup.sh** - Bash script (Unix/Linux/macOS only) - Colored terminal output

Both scripts provide identical functionality with different output styles!

---

#### Option A: Using Python Script (Recommended - Works Everywhere)

**About setup.py:**
The `setup.py` script is a cross-platform Python automation tool that handles complete RustEvo benchmarking workflow:

**Key Features:**
- ✅ **Cross-Platform**: Works on macOS, Linux, and Windows (no bash required)
- ✅ **Automatic Installation**: Installs Rust toolchain and Python dependencies
- ✅ **Dataset Validation**: Verifies all required files exist
- ✅ **Interactive Mode**: Menu-driven interface for easy operation
- ✅ **Command-Line Mode**: Full automation with arguments
- ✅ **Error Handling**: Clear error messages and status reporting

**One-Command Setup & Run:**
```bash
# Set your API credentials
export API_KEY="your-api-key-here"
export BASE_URL="https://your-api-endpoint.com"

# Run full setup and both evaluations
python3 setup.py --all
```

**Step-by-Step Commands:**
```bash
# 1. Set API credentials
export API_KEY="your-api-key-here"
export BASE_URL="https://your-api-endpoint.com"

# 2. Install dependencies only
python3 setup.py --install

# 3. Run RQ1 evaluation
python3 setup.py --rq1 kat-dev-hs-72b 8

# 4. Run RQ3 evaluation
python3 setup.py --rq3 kat-dev-hs-72b 8

# 5. Interactive menu mode
python3 setup.py
```

**All Available Commands:**
```bash
python3 setup.py --install          # Install dependencies only
python3 setup.py --rq1 MODEL WORKERS # Run RQ1 only
python3 setup.py --rq3 MODEL WORKERS # Run RQ3 only
python3 setup.py --all MODEL WORKERS # Run both RQ1 & RQ3
python3 setup.py                     # Interactive menu mode
python3 setup.py --help              # Show help
```

---

#### Option B: Using Bash Script (Unix/Linux/macOS)

**One-Command Setup & Run:**
```bash
# Set your API credentials
export API_KEY="your-api-key-here"
export BASE_URL="https://your-api-endpoint.com"

# Run full setup and both evaluations
chmod +x setup.sh && ./setup.sh --all
```

**All Available Commands:**
```bash
./setup.sh --install          # Install dependencies only
./setup.sh --rq1 MODEL WORKERS # Run RQ1 only
./setup.sh --rq3 MODEL WORKERS # Run RQ3 only
./setup.sh --all MODEL WORKERS # Run both RQ1 & RQ3
./setup.sh                     # Interactive menu mode
./setup.sh --help              # Show help
```

---

**What the scripts do:**
- ✅ Installs Rust toolchain (rustc, cargo)
- ✅ Installs Python dependencies (openai, tqdm)
- ✅ Verifies dataset files exist
- ✅ Creates Results directory
- ✅ Runs evaluations with progress tracking
- ✅ Displays comprehensive results summary

---

### 🛠️ Method 2: Manual Setup and Execution

**Quick Start:**
```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Install Python dependencies
pip3 install openai tqdm

# 3. Set API credentials
export API_KEY="your-api-key-here"
export BASE_URL="https://your-api-endpoint.com"

# 4. Run RQ1 evaluation
python3 Evaluate/eval_models_rq1.py \
  --file_a ./Dataset/RustEvo^2.json \
  --file_b ./Dataset/APIDocs.json \
  --output ./Results/rq1_results.json \
  --models kat-dev-hs-72b \
  --max_workers 8

# 5. Run RQ3 evaluation
python3 Evaluate/eval_models_rq3.py \
  --file_a ./Dataset/RustEvo^2.json \
  --file_b ./Dataset/APIDocs.json \
  --output ./Results/rq3_results.json \
  --models kat-dev-hs-72b \
  --max_workers 8
```

**Detailed Manual Setup:**

1. **Install Rust Toolchain**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   rustc --version  # Verify installation
   ```

2. **Install Python Dependencies**:
   ```bash
   pip3 install --upgrade pip
   pip3 install openai tqdm
   ```

3. **Configure API Credentials**:
   ```bash
   export API_KEY="your-api-key-here"
   export BASE_URL="https://your-api-endpoint.com"
   ```

4. **Verify Dataset Files**:
   Ensure these files exist:
   - `Dataset/RustEvo^2.json` - Task definitions
   - `Dataset/APIDocs.json` - API documentation

## Running Evaluations

### RQ1: Full Documentation Evaluation

Evaluates models with complete API documentation including signatures, documentation, and source code.

#### Using Automated Script:
```bash
./setup.sh --rq1 kat-dev-hs-72b 8
```

#### Using Manual Command:
```bash
# Set API credentials first
export API_KEY="your-api-key-here"
export BASE_URL="https://your-api-endpoint.com"

# Run RQ1 evaluation
python3 Evaluate/eval_models_rq1.py \
  --file_a ./Dataset/RustEvo^2.json \
  --file_b ./Dataset/APIDocs.json \
  --output ./Results/rq1_results.json \
  --models kat-dev-hs-72b \
  --max_workers 8 \
  --api_key $API_KEY \
  --base_url $BASE_URL
```

**Parameters**:
- `--file_a`: Path to task dataset (queries and tests)
- `--file_b`: Path to API documentation dataset
- `--output`: Path for results output (JSON format)
- `--models`: List of model names to evaluate (space-separated)
- `--max_workers`: Number of parallel workers (default: 8)
- `--api_key`: API key for LLM service (optional if set in environment)
- `--base_url`: Base URL for LLM service (optional if set in environment)

**Output Files**:
- `Results/rq1_results.json` - Detailed results for each task
- `Results/rq1_results_metrics.json` - Aggregated metrics and statistics

---

### RQ3: Minimal Documentation Evaluation

Evaluates models with minimal API information (only API name and module).

#### Using Automated Script:
```bash
./setup.sh --rq3 kat-dev-hs-72b 8
```

#### Using Manual Command:
```bash
# Set API credentials first
export API_KEY="your-api-key-here"
export BASE_URL="https://your-api-endpoint.com"

# Run RQ3 evaluation
python3 Evaluate/eval_models_rq3.py \
  --file_a ./Dataset/RustEvo^2.json \
  --file_b ./Dataset/APIDocs.json \
  --output ./Results/rq3_results.json \
  --models kat-dev-hs-72b \
  --max_workers 8 \
  --api_key $API_KEY \
  --base_url $BASE_URL
```

**Parameters**: Same as RQ1

**Output Files**:
- `Results/rq3_results.json` - Detailed results for each task
- `Results/rq3_results_metrics.json` - Aggregated metrics and statistics

---

### Running Both RQ1 and RQ3

#### Using Automated Script:
```bash
./setup.sh --all kat-dev-hs-72b 8
```

#### Using Manual Commands:
```bash
# Set API credentials first
export API_KEY="your-api-key-here"
export BASE_URL="https://your-api-endpoint.com"

# Run RQ1
python3 Evaluate/eval_models_rq1.py \
  --file_a ./Dataset/RustEvo^2.json \
  --file_b ./Dataset/APIDocs.json \
  --output ./Results/rq1_results.json \
  --models kat-dev-hs-72b \
  --max_workers 8 \
  --api_key $API_KEY \
  --base_url $BASE_URL

# Run RQ3
python3 Evaluate/eval_models_rq3.py \
  --file_a ./Dataset/RustEvo^2.json \
  --file_b ./Dataset/APIDocs.json \
  --output ./Results/rq3_results.json \
  --models kat-dev-hs-72b \
  --max_workers 8 \
  --api_key $API_KEY \
  --base_url $BASE_URL
```

## Understanding Results

### Metrics Calculated

Both evaluation scripts calculate comprehensive metrics:

1. **Pass@1**: Percentage of tasks that passed on first attempt
   - Formula: `(Success count / Total tasks) × 100`
   
2. **API Usage Accuracy**: Percentage of tasks that correctly used the target API
   - Formula: `(Tasks using API / Total tasks) × 100`

3. **API Coverage (Distinct)**: Percentage of unique APIs successfully used
   - Formula: `(Distinct APIs used / Total distinct APIs) × 100`

4. **Borrow-checker Failure Rate**: Percentage of failures due to borrow checker errors
   - Formula: `(Borrow checker failures / Total failures) × 100`

5. **Compilation Errors**: Count of compilation failures

6. **Test Failures**: Count of runtime test failures

### Metrics by Change Type

Results are also broken down by API change type:
- `signature` - Signature changes
- `parameter_addition` - New parameters added
- `return_type_change` - Return type modifications
- `deprecation` - Deprecated APIs
- And more...

### Example Metrics Output

```json
{
  "kat-dev-hs-72b": {
    "total_tasks": 588,
    "success_count": 141,
    "pass_at_1": 23.98,
    "api_usage_accuracy": 82.82,
    "api_coverage_distinct": 75.50,
    "api_coverage_distinct_count": "150/200",
    "borrow_checker_failure_rate_over_failures": 76.73,
    "compilation_errors": 250,
    "test_failures": 197
  }
}
```

## Checkpoint and Resume

Both scripts support checkpointing:
- Results are automatically saved every 10 tasks
- If interrupted, re-run the same command to resume from the last checkpoint
- Already completed tasks are skipped automatically

## Troubleshooting

### Common Issues

1. **Rust Not Found**:
   ```
   rustup: command not found
   ```
   Solution: Install Rust using rustup (see Prerequisites)

2. **API Key Error**:
   ```
   Warning: No API key provided
   ```
   Solution: Set API_KEY environment variable or use --api_key parameter

3. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'openai'
   ```
   Solution: Install required packages with `pip install openai tqdm`

4. **Timeout Errors**:
   - Compilation/test timeouts are set to 30-45 seconds
   - Large projects may need timeout adjustments in the script

### Debug Mode

To see detailed error messages, check the `validation_output` field in results JSON.

## Project Structure

```
RustEvo/
├── Dataset/
│   ├── RustEvo^2.json      # Task dataset
│   └── APIDocs.json        # API documentation
├── Evaluate/
│   ├── eval_models_rq1.py  # RQ1 evaluation script
│   ├── eval_models_rq3.py  # RQ3 evaluation script
│   └── ...                 # Other evaluation utilities
├── Results/
│   ├── rq1_results.json    # RQ1 detailed results
│   ├── rq1_results_metrics.json  # RQ1 metrics
│   ├── rq3_results.json    # RQ3 detailed results
│   └── rq3_results_metrics.json  # RQ3 metrics
└── README.md               # This file
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{rustevo2024,
  title={RustEvo: A Benchmark for Evaluating LLMs on Rust API Evolution},
  author={...},
  booktitle={...},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [maintainer email].
