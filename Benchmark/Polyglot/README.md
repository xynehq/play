
# Aider benchmark harness

Aider uses benchmarks to quantitatively measure how well it works
with various LLMs.
This directory holds the harness and tools needed to run the benchmarking suite.

## Background

The benchmark is based on the [Exercism](https://github.com/exercism/python) coding exercises.
This
benchmark evaluates how effectively aider and LLMs can translate a
natural language coding request into executable code saved into
files that pass unit tests.
It provides an end-to-end evaluation of not just
the LLM's coding ability, but also its capacity to *edit existing code*
and *format those code edits* so that aider can save the
edits to the local source files.

See [this writeup for a longer discussion about the benchmark](https://aider.chat/2024/12/21/polyglot.html).

The benchmark is intended to be run *inside a docker container*.
This is because the benchmarking harness will be
taking code written by an LLM
and executing it without any human review or supervision!
The LLM could generate dangerous python that harms your system, like this: `import os; os.system("sudo rm -rf /")`.
Running inside a docker container helps limit the damage that could be done.

## Quick Start (Automated)

🚀 **NEW: One-Command Automation Script**

For the easiest benchmarking experience, use the automated `polyglot` script that handles everything:

```bash
# Make the script executable and run it
chmod +x polyglot
./polyglot
```

The automation script will:
1. ✅ **Check Docker installation** and provide installation guides if needed
2. ✅ **Clone repositories** (Aider + polyglot benchmark exercises)
3. ✅ **Build Docker container** with fallback handling for build issues
4. ✅ **Interactive configuration** with simple prompts for:
   - Model name (e.g., gpt-4, claude-3.5-sonnet, custom models)
   - API credentials (URL + API key for any provider)
   - Language selection (All, C++, Go, Java, JavaScript, Python, Rust)
   - Edit format, threads, and other settings
5. ✅ **Run benchmark** safely in Docker container
6. ✅ **Generate results** with comprehensive statistics

**Features:**
- 🌐 **Universal API Support**: Works with OpenAI, Anthropic, custom endpoints, or any OpenAI-compatible API
- 🎯 **Language Filtering**: Test specific programming languages instead of all
- 🔧 **Smart Model Handling**: Automatically formats model names for litellm compatibility
- 📊 **Comprehensive Reports**: Displays pass rates, costs, and detailed statistics
- 🛡️ **Safe Execution**: All LLM code runs in isolated Docker containers

### How the Automation Script Works

The `polyglot` script is a comprehensive Python automation tool that handles the entire benchmarking pipeline. Here's the detailed technical breakdown:

#### **1. Prerequisites Check (`check_prerequisites()`)**
```python
# Automatically detects operating system (macOS, Linux)
# Checks Docker installation and daemon status
# Provides platform-specific installation guides if Docker is missing
# Offers to continue without Docker (with warnings) if daemon isn't running
```

**What it does:**
- Runs `docker --version` and `docker info` to verify installation and status
- Provides detailed installation instructions for macOS (OrbStack, Docker Desktop, Homebrew)
- Provides installation commands for Linux (Ubuntu/Debian, Fedora/RHEL)
- Exits gracefully if Docker is required but not available

#### **2. Repository Setup**
```python
# Clones https://github.com/Aider-AI/aider.git
# Creates tmp.benchmarks directory structure
# Clones https://github.com/Aider-AI/polyglot-benchmark
# Handles existing repositories with git pull updates
# Includes error recovery for corrupted git repositories
```

**Directory Structure Created:**
```
aider/
├── benchmark/          # Benchmarking scripts
├── tmp.benchmarks/     # Results storage
│   └── polyglot-benchmark/  # Exercise files
│       ├── cpp/        # C++ exercises
│       ├── go/         # Go exercises
│       ├── java/       # Java exercises
│       ├── javascript/ # JavaScript exercises
│       ├── python/     # Python exercises
│       └── rust/       # Rust exercises
```

#### **3. Docker Container Management**
```python
# Checks for existing 'aider-benchmark' Docker image
# Builds container using ./benchmark/docker_build.sh
# Includes fallback build with setuptools-scm version override
# Handles build failures gracefully with detailed error messages
```

**Docker Build Process:**
- Uses `benchmark/Dockerfile` to create isolated environment
- Installs all language runtimes (Python, Go, Java, JavaScript, C++, Rust)
- Sets up testing frameworks for each language
- Includes fallback for setuptools-scm version detection issues

#### **4. Interactive Configuration System**
```python
# Model Configuration
config["model"] = input("Model name: ")  # e.g., gpt-4, claude-3.5-sonnet

# API Configuration (Universal)
api_base = input("API base URL (optional): ")  # Custom endpoints
api_key = input("API key: ")  # Required for all providers

# Language Selection Menu
language_map = {
    "2": "cpp", "3": "go", "4": "java", 
    "5": "javascript", "6": "python", "7": "rust"
}

# Additional Settings
edit_format = ["whole", "diff", "udiff"]  # Code editing approach
threads = 10  # Parallel execution
num_tests = "all"  # Test count limit
keywords = ""  # Additional filtering
```

**Configuration Logic:**
- **API Detection**: Automatically detects provider based on model name
- **Model Formatting**: Adds `openai/` prefix for custom models with litellm
- **Environment Variables**: Sets appropriate API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **Language Filtering**: Uses `--keywords` parameter to filter by language directory

#### **5. Benchmark Execution Engine**
```python
# Constructs Docker command with proper isolation
docker_run_cmd = [
    "docker", "run", "--rm",
    "--memory=12g", "--memory-swap=12g",  # Resource limits
    "--add-host=host.docker.internal:host-gateway",  # Network access
    "-v", f"{aider_dir}:/aider",  # Mount source code
    "-v", f"{benchmarks_dir}:/benchmarks",  # Mount results
    "-e", "AIDER_DOCKER=1",  # Environment flags
    "-e", f"OPENAI_API_KEY={api_key}",  # API credentials
    "aider-benchmark",  # Container image
    "bash", "-c", benchmark_command  # Execution command
]

# Benchmark command structure
benchmark_command = [
    "pip install -e .[dev] &&",  # Install aider in development mode
    "./benchmark/benchmark.py",  # Run benchmark script
    run_name,  # Timestamped run identifier
    "--model", model_name,  # LLM model to test
    "--edit-format", edit_format,  # Code editing approach
    "--threads", str(threads),  # Parallel execution
    "--tries", "2",  # Retry attempts per test
    "--exercises-dir", "polyglot-benchmark"  # Exercise location
]
```

**Execution Process:**
- **Isolation**: All LLM-generated code runs in Docker container
- **Resource Management**: 12GB memory limit prevents system overload
- **Parallel Processing**: Configurable thread count for faster execution
- **API Integration**: Passes credentials securely via environment variables
- **Progress Tracking**: Real-time output from benchmark execution

#### **6. Results Processing and Reporting**
```python
# Locates results directory: tmp.benchmarks/YYYY-MM-DD-HH-MM-SS--model-format
# Reads YAML statistics files automatically generated by benchmark
# Parses JSON result files for individual test outcomes
# Displays comprehensive summary with pass rates and costs
```

**Result Files Generated:**
```
tmp.benchmarks/2024-11-12-22-30-15--gpt-4-whole/
├── stats.yaml              # Overall statistics
├── cpp/
│   └── exercise-name/
│       ├── .aider.results.json    # Individual test results
│       ├── .aider.chat.history.md # LLM conversation log
│       └── source_files.*         # Generated code
└── [other languages...]
```

**Statistics Reported:**
- **Pass Rates**: Percentage of tests passing on try 1, try 2
- **Cost Analysis**: Total API costs and cost per test case
- **Error Analysis**: Syntax errors, malformed responses, timeouts
- **Performance Metrics**: Seconds per case, total execution time
- **Model Information**: Exact model, edit format, commit hash

#### **7. Error Handling and Recovery**
```python
# Docker build failures: Automatic retry with version override
# Git repository corruption: Automatic re-cloning
# API connection issues: Clear error messages with troubleshooting
# Missing dependencies: Detailed installation instructions
# Interrupted execution: Safe cleanup and partial results recovery
```

**Robust Error Management:**
- **Graceful Degradation**: Continues with warnings when possible
- **Clear Diagnostics**: Detailed error messages with solutions
- **Automatic Recovery**: Handles common issues without user intervention
- **Safe Cleanup**: Proper Docker container cleanup on exit

#### **8. Security and Safety Features**
```python
# Docker Isolation: All LLM code execution in containers
# Resource Limits: Memory and CPU constraints
# Network Isolation: Limited external access
# File System Protection: Read-only mounts where possible
# API Key Security: Environment variable injection, no disk storage
```

**Security Measures:**
- **Sandboxed Execution**: LLM-generated code cannot access host system
- **Resource Protection**: Prevents resource exhaustion attacks
- **Credential Safety**: API keys never written to disk
- **Clean Environment**: Fresh container for each run

This automation script transforms a complex 15+ step manual process into a single command while maintaining all the safety, flexibility, and power of the original benchmarking system.

## Manual Usage (Advanced)

For advanced users who want manual control, there are 3 main tasks involved in benchmarking aider:

1. Install and setup for benchmarking.

2. Run the benchmark to measure performance across all the exercises.

3. Generate a summary report of how many of the exercises succeeded or failed.

### Setup for benchmarking

First, prepare all the groundwork for running the benchmarks.
These steps only need to be done once.

```
# Clone the aider repo
git clone https://github.com/Aider-AI/aider.git

# Create the scratch dir to hold benchmarking results inside the main aider dir:
cd aider
mkdir tmp.benchmarks

# Clone the repo with the exercises
git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark

# Build the docker container
./benchmark/docker_build.sh
```

### Running the benchmark

Launch the docker container and run the benchmark inside it:

```
# Launch the docker container
./benchmark/docker.sh

# Inside the container, install aider as a development build.
# This way you're running the code that you cloned above, including any local changes.
pip install -e .[dev]

# Run the benchmark:
./benchmark/benchmark.py a-helpful-name-for-this-run --model gpt-3.5-turbo --edit-format whole --threads 10 --exercises-dir polyglot-benchmark
```

The above will create a folder `tmp.benchmarks/YYYY-MM-DD-HH-MM-SS--a-helpful-name-for-this-run` with benchmarking results.
Run like this, the script will run all the exercises in a random order.

You can run `./benchmark/benchmark.py --help` for a list of all the arguments, but here are the most useful to keep in mind:

- `--model` is the name of the model, same as you would pass directly to `aider`.
- `--edit-format` is the name of the edit format, same as you would pass directly to `aider`. When working with an experimental LLM, I recommend starting with `whole`
- `--threads` specifies how many exercises to benchmark in parallel. Start with a single thread if you are working out the kinks on your benchmarking setup or working with a new model, etc. Once you are getting reliable results, you can speed up the process by running with more threads. 10 works well against the OpenAI APIs.
- `--num-tests` specifies how many of the tests to run before stopping. This is another way to start gently as you debug your benchmarking setup.
- `--keywords` filters the tests to run to only the ones whose name match the supplied argument (similar to `pytest -k xxxx`).
- `--read-model-settings=<filename.yml>` specify model settings, see here: https://aider.chat/docs/config/adv-model-settings.html#model-settings

### Benchmark report

You can generate stats about any benchmark, including ones which are still running.
You don't need to run this inside the docker container, as it is just
collecting stats not executing unsafe python.

```
# Generate stats for a specific benchmarking directory
./benchmark/benchmark.py --stats tmp.benchmarks/YYYY-MM-DD-HH-MM-SS--a-helpful-name-for-this-run
```

The benchmark report is a yaml record with statistics about the run:

```yaml
- dirname: 2024-07-04-14-32-08--claude-3.5-sonnet-diff-continue
  test_cases: 225
  model: claude-3.5-sonnet
  edit_format: diff
  commit_hash: 35f21b5
  pass_rate_1: 57.1
  pass_rate_2: 77.4
  percent_cases_well_formed: 99.2
  error_outputs: 23
  num_malformed_responses: 4
  num_with_malformed_responses: 1
  user_asks: 2
  lazy_comments: 0
  syntax_errors: 1
  indentation_errors: 0
  exhausted_context_windows: 0
  test_timeouts: 1
  command: aider --sonnet
  date: 2024-07-04
  versions: 0.42.1-dev
  seconds_per_case: 17.6
  total_cost: 3.6346
```

The key statistics are the `pass_rate_#` entries, which report the
percent of the tasks which had all tests passing.
There will be multiple of these pass rate stats,
depending on the value of the `--tries` parameter.

The yaml also includes all the settings which were in effect for the benchmark run.
It also reports the git hash of the repo at the time that the benchmark was
run, with `(dirty)` if there were uncommitted changes.
It's good practice to commit the repo before starting a benchmark run.
This way the `model`, `edit_format` and `commit_hash`
should be enough to reliably reproduce any benchmark run.

You can see examples of the benchmark report yaml in the
[aider leaderboard data files](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/).


## Limitations, notes

- Contributions of benchmark results are welcome! Submit results by opening a PR with edits to the
[aider leaderboard data files](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/).
- These scripts are not intended for use by typical aider end users.
- Some of these tools are written as `bash` scripts, so it will be hard to use them on Windows.
