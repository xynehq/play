#!/bin/bash

# run_all_benchmarks.sh
# Script to run Tau and Tau2 benchmarks using configuration from model_config.yaml
# Creates tmux session with separate windows for each benchmark

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to parse YAML using python (fallback if yq not available)
parse_yaml() {
    python3 -c "
import yaml
import sys

try:
    with open('model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    key = '$1'
    if '.' in key:
        keys = key.split('.')
        value = config
        for k in keys:
            value = value.get(k, '')
    else:
        value = config.get(key, '')
    
    print(value)
except Exception as e:
    print(f'', end='')
    sys.exit(1)
"
}

# Check dependencies
print_status "Checking dependencies..."

if ! command_exists tmux; then
    print_error "tmux is not installed. Please install tmux first."
    exit 1
fi

# Check for YAML parser (prefer yq, fallback to python)
if command_exists yq; then
    YAML_PARSER="yq"
    print_status "Using yq for YAML parsing"
elif command_exists python3 && python3 -c "import yaml" 2>/dev/null; then
    YAML_PARSER="python"
    print_status "Using python for YAML parsing"
else
    print_error "Neither yq nor python with PyYAML is available. Please install one of them."
    exit 1
fi

# Parse configuration from model_config.yaml
print_status "Reading configuration from model_config.yaml..."

if [ "$YAML_PARSER" = "yq" ]; then
    API_BASE=$(yq '.api_base' model_config.yaml 2>/dev/null || echo "")
    MODEL_NAME=$(yq '.model_name' model_config.yaml 2>/dev/null || echo "")
    API_KEY=$(yq '.api_key' model_config.yaml 2>/dev/null || echo "")
else
    API_BASE=$(parse_yaml 'api_base')
    MODEL_NAME=$(parse_yaml 'model_name')
    API_KEY=$(parse_yaml 'api_key')
fi

# Validate configuration
if [ -z "$API_BASE" ] || [ -z "$MODEL_NAME" ] || [ -z "$API_KEY" ]; then
    print_error "Failed to parse configuration from model_config.yaml"
    print_error "Please ensure the file exists and contains api_base, model_name, and api_key"
    exit 1
fi

print_success "Configuration loaded successfully"
print_status "API Base: $API_BASE"
print_status "Model Name: $MODEL_NAME"
print_status "API Key: ${API_KEY:0:10}..." # Show only first 10 chars for security

# Format model name for commands (add openai/ prefix if not present)
if [[ ! "$MODEL_NAME" =~ ^openai/ ]]; then
    FORMATTED_MODEL_NAME="openai/$MODEL_NAME"
else
    FORMATTED_MODEL_NAME="$MODEL_NAME"
fi

# Setup OpenHands workspace before running benchmarks
print_status "Setting up OpenHands workspace..."

# Create openhands_workspace directory in SWE-Bench if it doesn't exist
if [ ! -d "SWE-Bench/openhands_workspace" ]; then
    print_status "Creating openhands_workspace directory..."
    mkdir -p SWE-Bench/openhands_workspace
fi

# Clone OpenHands repo if it doesn't exist
if [ ! -d "SWE-Bench/openhands_workspace/OpenHands" ]; then
    print_status "Cloning OpenHands repository..."
    cd SWE-Bench/openhands_workspace
    git clone https://github.com/OpenHands/OpenHands
    cd ../..
    print_success "OpenHands repository cloned successfully"
else
    print_status "OpenHands repository already exists"
fi

# Setup virtual environment for OpenHands
if [ ! -d "SWE-Bench/openhands_workspace/venv" ]; then
    print_status "Creating virtual environment for OpenHands..."
    python3 -m venv SWE-Bench/openhands_workspace/venv
    # Install OpenHands in the virtual environment
    print_status "Installing OpenHands in virtual environment..."
    cd SWE-Bench/openhands_workspace/OpenHands
    source ../venv/bin/activate
    python -m pip install -e .
    deactivate
    cd ../../..
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

print_success "OpenHands setup completed successfully"

# Check if directories exist
if [ ! -d "tau-bench" ]; then
    print_error "tau-bench directory not found"
    exit 1
fi

if [ ! -d "tau2-bench" ]; then
    print_error "tau2-bench directory not found"
    exit 1
fi

if [ ! -d "Polyglot" ]; then
    print_error "Polyglot directory not found"
    exit 1
fi

if [ ! -d "hyperswitch-benchmark" ]; then
    print_error "hyperswitch-benchmark directory not found"
    exit 1
fi

# Check if virtual environments exist
if [ ! -d "tau-bench/venv" ]; then
    print_warning "tau-bench/venv not found. Please ensure virtual environment is set up."
fi

if [ ! -d "tau2-bench/venv" ]; then
    print_warning "tau2-bench/venv not found. Please ensure virtual environment is set up."
fi

# Clean up existing tmux session if it exists
if tmux has-session -t benchmarks 2>/dev/null; then
    print_warning "Existing tmux session 'benchmarks' found. Killing it..."
    tmux kill-session -t benchmarks
fi

# Create new tmux session
print_status "Creating tmux session 'benchmarks'..."
tmux new-session -d -s benchmarks

# Function to send command to tmux pane
send_command() {
    local session=$1
    local window=$2
    local pane=$3
    local command=$4
    
    tmux send-keys -t "$session:$window.$pane" "$command" Enter
}

# Create Tau window with 2 panes (airline and retail)
print_status "Setting up Tau window with 2 panes..."
tmux new-window -t benchmarks:1 -n "Tau"
tmux split-window -h

# Tau commands
TAU_AIRLINE_CMD="cd tau-bench && source venv/bin/activate && OPENAI_API_BASE=\"$API_BASE\" OPENAI_API_KEY=\"$API_KEY\" python3 run.py --env airline --model \"$FORMATTED_MODEL_NAME\" --model-provider openai --user-model \"$FORMATTED_MODEL_NAME\" --user-model-provider openai --agent-strategy tool-calling --max-concurrency 3"

TAU_RETAIL_CMD="cd tau-bench && source venv/bin/activate && OPENAI_API_BASE=\"$API_BASE\" OPENAI_API_KEY=\"$API_KEY\" python3 run.py --env retail --model \"$FORMATTED_MODEL_NAME\" --model-provider openai --user-model \"$FORMATTED_MODEL_NAME\" --user-model-provider openai --agent-strategy tool-calling --max-concurrency 3"

# Send Tau commands
send_command "benchmarks" "1" "0" "$TAU_AIRLINE_CMD"
send_command "benchmarks" "1" "1" "$TAU_RETAIL_CMD"

# Set pane titles for Tau
tmux select-pane -t benchmarks:1.0 -T "Tau-Airline"
tmux select-pane -t benchmarks:1.1 -T "Tau-Retail"

# Create Tau2 window with 4 panes (retail, airline, mock, telecom)
print_status "Setting up Tau2 window with 4 panes..."
tmux new-window -t benchmarks:2 -n "Tau2"
tmux split-window -h
tmux split-window -v
tmux select-pane -t benchmarks:2.0
tmux split-window -v

# Tau2 commands
TAU2_RETAIL_CMD="cd tau2-bench && source venv/bin/activate && OPENAI_API_BASE=\"$API_BASE\" OPENAI_API_KEY=\"$API_KEY\" tau2 run --domain retail --agent-llm \"$FORMATTED_MODEL_NAME\" --user-llm \"$FORMATTED_MODEL_NAME\" --max-concurrency 3"

TAU2_AIRLINE_CMD="cd tau2-bench && source venv/bin/activate && OPENAI_API_BASE=\"$API_BASE\" OPENAI_API_KEY=\"$API_KEY\" tau2 run --domain airline --agent-llm \"$FORMATTED_MODEL_NAME\" --user-llm \"$FORMATTED_MODEL_NAME\" --max-concurrency 3"

TAU2_MOCK_CMD="cd tau2-bench && source venv/bin/activate && OPENAI_API_BASE=\"$API_BASE\" OPENAI_API_KEY=\"$API_KEY\" tau2 run --domain mock --agent-llm \"$FORMATTED_MODEL_NAME\" --user-llm \"$FORMATTED_MODEL_NAME\" --max-concurrency 3"

TAU2_TELECOM_CMD="cd tau2-bench && source venv/bin/activate && OPENAI_API_BASE=\"$API_BASE\" OPENAI_API_KEY=\"$API_KEY\" tau2 run --domain telecom --agent-llm \"$FORMATTED_MODEL_NAME\" --user-llm \"$FORMATTED_MODEL_NAME\" --max-concurrency 3"

# Send Tau2 commands
send_command "benchmarks" "2" "0" "$TAU2_RETAIL_CMD"
send_command "benchmarks" "2" "1" "$TAU2_AIRLINE_CMD"
send_command "benchmarks" "2" "2" "$TAU2_MOCK_CMD"
send_command "benchmarks" "2" "3" "$TAU2_TELECOM_CMD"

# Set pane titles for Tau2
tmux select-pane -t benchmarks:2.0 -T "Tau2-Retail"
tmux select-pane -t benchmarks:2.1 -T "Tau2-Airline"
tmux select-pane -t benchmarks:2.2 -T "Tau2-Mock"
tmux select-pane -t benchmarks:2.3 -T "Tau2-Telecom"

# Create Polyglot window with single pane
print_status "Setting up Polyglot window..."
tmux new-window -t benchmarks:3 -n "Polyglot"

# Polyglot command
POLYGLOT_CMD="cd Polyglot && python3 -m venv venv && source venv/bin/activate && pip install pyyaml && ./polyglot"

# Send Polyglot command
send_command "benchmarks" "3" "0" "$POLYGLOT_CMD"

# Set pane title for Polyglot
tmux select-pane -t benchmarks:3.0 -T "Polyglot"

# Create Hyperswitch window with single pane
print_status "Setting up Hyperswitch window..."
tmux new-window -t benchmarks:4 -n "Hyperswitch"

# Hyperswitch command
HYPERSWITCH_CMD="cd hyperswitch-benchmark && python3 -m venv venv && source venv/bin/activate && pip install openai pyyaml && python3 inference_3.py"

# Send Hyperswitch command
send_command "benchmarks" "4" "0" "$HYPERSWITCH_CMD"

# Set pane title for Hyperswitch
tmux select-pane -t benchmarks:4.0 -T "Hyperswitch"

# Create Archit-Eval window with single pane
print_status "Setting up Archit-Eval window..."
tmux new-window -t benchmarks:5 -n "Archit-Eval"

# Archit-Eval command
ARCHIT_EVAL_CMD="cd Archit-Eval && python3 -m venv venv && source venv/bin/activate && pip install openai pyyaml requests datasets && python3 script.py"

# Send Archit-Eval command
send_command "benchmarks" "5" "0" "$ARCHIT_EVAL_CMD"

# Set pane title for Archit-Eval
tmux select-pane -t benchmarks:5.0 -T "Archit-Eval"

# Create RustEvo window with single pane
print_status "Setting up RustEvo window..."
tmux new-window -t benchmarks:6 -n "RustEvo"

# RustEvo command
RUSTEVO_CMD="cd RustEvo && python3 -m venv venv && source venv/bin/activate && pip install openai pyyaml && python3 setup.py"

# Send RustEvo command
send_command "benchmarks" "6" "0" "$RUSTEVO_CMD"

# Set pane title for RustEvo
tmux select-pane -t benchmarks:6.0 -T "RustEvo"


# Create SWE-Bench window with single pane
print_status "Setting up SWE-Bench window..."
tmux new-window -t benchmarks:7 -n "SWE-Bench"

# SWE-Bench command
SWE_BENCH_CMD="cd SWE-Bench && source ./openhands_workspace/venv/bin/activate && pip install datasets && python3 script.py"

# Send SWE-Bench command
send_command "benchmarks" "7" "0" "$SWE_BENCH_CMD"

# Set pane title for SWE-Bench
tmux select-pane -t benchmarks:7.0 -T "SWE-Bench"


# Select first window and attach to session
tmux select-window -t benchmarks:1
tmux attach-session -t benchmarks

print_success "Benchmark session started successfully!"
print_status "Session name: benchmarks"
print_status "Windows:"
print_status "  1. Tau (2 panes: airline, retail)"
print_status "  2. Tau2 (4 panes: retail, airline, mock, telecom)"
print_status "  3. Polyglot (1 pane: polyglot)"
print_status "  4. Hyperswitch (1 pane: inference)"
print_status "  5. Archit-Eval (1 pane: archit-eval)"
print_status "  6. RustEvo (1 pane: rust-evo)"
print_status "  7. SWE-Bench (1 pane: swe-bench)"
print_status ""
print_status "Use Ctrl+B then:"
print_status "  - n: next window"
print_status "  - p: previous window"
print_status "  - 1/2/3/4/5/6/7: go to window 1/2/3/4/5/6/7"
print_status "  - o: switch panes"
print_status "  - d: detach session"
print_status ""
print_status "To reattach later: tmux attach-session -t benchmarks"
print_status "To kill session: tmux kill-session -t benchmarks"