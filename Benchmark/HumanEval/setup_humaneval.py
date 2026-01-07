#!/usr/bin/env python3
"""
setup_humaneval.py

One-step HumanEval setup + evaluation script.

✅ Features:
- Installs Python 3.7+
- Sets up venv and dependencies
- Clones HumanEval repo
- Configures API settings
- Runs Rust/Python benchmarking
- Generates comprehensive results
"""

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
import json
import shutil
import yaml
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def run(cmd, cwd=None, check=True, env=None, capture_output=False):
    """Run a command"""
    print(f"🔹 {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    if capture_output:
        result = subprocess.run(cmd, cwd=cwd, check=check, env=env, capture_output=True, text=True)
        return result.stdout.strip()
    else:
        subprocess.run(cmd, cwd=cwd, check=check, env=env)
        return None

def find_compatible_python():
    """Find a compatible Python 3.7+ installation"""
    print("\n🔍 Looking for compatible Python installation...")
    
    current_major, current_minor = sys.version_info[:2]
    print(f"🐍 Current Python: {current_major}.{current_minor}")
    
    # Python 3.7+ is required
    if current_major == 3 and current_minor >= 7:
        print(f"✅ Current Python {current_major}.{current_minor} is compatible")
        return sys.executable
    
    # Search for compatible Python
    python_candidates = [
        "python3.12", "python3.11", "python3.10", "python3.9", "python3.8", "python3.7",
        "/usr/local/bin/python3", "/opt/homebrew/bin/python3"
    ]
    
    for py_cmd in python_candidates:
        try:
            result = subprocess.run([py_cmd, "--version"], capture_output=True, text=True, check=True)
            version_str = result.stdout.strip()
            import re
            match = re.search(r'Python 3\.(\d+)', version_str)
            if match and int(match.group(1)) >= 7:
                print(f"✅ Found compatible Python: {version_str}")
                return py_cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("\n❌ No compatible Python 3.7+ found")
    print("📦 Please install Python 3.7 or higher")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Step 1: Find Python + Create Virtual Environment
# -----------------------------------------------------------------------------
print("="*70)
print("HumanEval Benchmark Setup Script")
print("="*70)

python_executable = find_compatible_python()
venv_dir = Path("venv_humaneval")

if not venv_dir.exists():
    print(f"\n🔧 Creating virtual environment...")
    subprocess.run([python_executable, "-m", "venv", str(venv_dir)], check=True)
    print("✅ Virtual environment created")
else:
    print("\n✅ Virtual environment already exists")

python_path = venv_dir / "bin" / "python"
pip_path = venv_dir / "bin" / "pip"

# Verify Python version
result = subprocess.run([str(python_path), "--version"], capture_output=True, text=True)
print(f"✅ Using: {result.stdout.strip()}")

# -----------------------------------------------------------------------------
# Step 2: Install dependencies
# -----------------------------------------------------------------------------
print("\n📦 Installing dependencies...")

# Upgrade pip
run([str(pip_path), "install", "--upgrade", "pip"])

# Install required packages
packages = [
    "numpy",
    "requests",
    "tqdm",
    "python-dotenv",
    "fire",
    "pyyaml",
]

run([str(pip_path), "install"] + packages)

# -----------------------------------------------------------------------------
# Step 3: Clone HumanEval repository
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("=== Clone HumanEval Repository ===")
print("="*70)

repo_url = "https://github.com/openai/human-eval.git"
repo_dir = Path("human-eval")

if not repo_dir.exists():
    print(f"📥 Cloning HumanEval repository to {repo_dir}...")
    run(["git", "clone", repo_url, str(repo_dir)])
    print("✅ Repository cloned successfully")
else:
    print(f"✅ HumanEval repo already exists at {repo_dir}. Skipping clone.")

# Copy Rust dataset to parent data directory if it doesn't exist
print("\nℹ️  Benchmark script will run from parent directory (has Rust support)")

# Ensure data directory exists
data_dir = Path("Dataset")
data_dir.mkdir(exist_ok=True)

# Copy Rust dataset from cloned repo to parent directory
rust_dataset_src = repo_dir / "data" / "humaneval-rust.jsonl.gz"
rust_dataset_dst = data_dir / "humaneval-rust.jsonl.gz"

if rust_dataset_src.exists() and not rust_dataset_dst.exists():
    print(f"\n📦 Copying Rust dataset to {rust_dataset_dst}...")
    shutil.copy2(rust_dataset_src, rust_dataset_dst)
    print("✅ Rust dataset copied successfully")
elif rust_dataset_dst.exists():
    print(f"✅ Rust dataset already exists at {rust_dataset_dst}")

# -----------------------------------------------------------------------------
# Step 4: Install HumanEval package
# -----------------------------------------------------------------------------
print("\n📦 Installing human_eval package...")
try:
    run([str(pip_path), "install", "-e", str(repo_dir)])
    print("✅ human_eval package installed")
except subprocess.CalledProcessError:
    print("⚠️  Package installation had warnings (continuing anyway)")
    print("ℹ️  The package is still usable despite the entry point error")

# -----------------------------------------------------------------------------
# Step 5: API Configuration
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("=== API Configuration ===")
print("="*70)

env_file = Path(".env")

config_path = Path(Path(__file__).parent.parent / "model_config.yaml")
config_file = Path(config_path)
ap_k = ""
b_url = ""
humaneval_config = {}
with open(config_file, 'r') as f:
    yaml_data=yaml.safe_load(f)
    ap_k=yaml_data.get("api_key")
    b_url=yaml_data.get("api_base")
    humaneval_config = yaml_data.get("humaneval_config", {})

if env_file.exists():
    print(f"\n⚠️  .env file already exists!")
    overwrite = humaneval_config.get("overwrite_env", "n")
    if overwrite != "y":
        print("✅ Keeping existing .env file")
        configure_api = "n"
    else:
        configure_api = "y"
else:
    configure_api = humaneval_config.get("configure_api", "y")

if configure_api == "y":
    print("\n⚙️  API configuration:")
    
    api_key = ap_k
    base_url = b_url
    fine_tuned_model = humaneval_config.get("fine_tuned_model","").strip()
    base_model = humaneval_config.get("base_model","").strip()
    
    with open(env_file, 'w') as f:
        f.write("# HumanEval API Configuration\n\n")
        f.write(f"API_KEY={api_key}\n")
        f.write(f"BASE_URL={base_url}\n")
        f.write(f"FINE_TUNED_MODEL={fine_tuned_model}\n")
        if base_model:
            f.write(f"BASE_MODEL={base_model}\n")
    
    print(f"\n✅ Configuration saved to .env")

# -----------------------------------------------------------------------------
# Step 6: Choose benchmark language
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("=== Choose Benchmark Language ===")
print("="*70)
print("1. Rust (requires Cargo and humaneval-rust.jsonl.gz dataset)")
print("2. Python (standard HumanEval)")
print("="*70)

lang_choice = humaneval_config.get("lang_choice",1)

if lang_choice == 1:
    language = "rust"
    print("✅ Selected: Rust benchmark")
    
    # Check for Rust installation
    try:
        run(["cargo", "--version"], capture_output=True)
        print("✅ Cargo is installed")
    except:
        print("\n⚠️  Cargo not found!")
        print("📦 Install Rust from: https://rustup.rs/")
        print("   Then run this script again")
        sys.exit(1)
    
    # Check for Rust dataset in parent data directory
    rust_dataset_parent = data_dir / "humaneval-rust.jsonl.gz"
    if not rust_dataset_parent.exists():
        # Try to copy from cloned repo if it exists there
        rust_dataset_cloned = repo_dir / "data" / "humaneval-rust.jsonl.gz"
        if rust_dataset_cloned.exists():
            print(f"\n📦 Copying Rust dataset from {rust_dataset_cloned}...")
            shutil.copy2(rust_dataset_cloned, rust_dataset_parent)
            print("✅ Rust dataset copied successfully")
        else:
            print(f"\n⚠️  Rust dataset not found at: {rust_dataset_parent}")
            print("📥 Download from MultiPL-E:")
            print("   https://github.com/nuprl/MultiPL-E")
            print(f"   Then place it at: {rust_dataset_parent}")
            sys.exit(1)
    print(f"✅ Rust dataset found: {rust_dataset_parent}")
    
else:
    language = "python"
    print("✅ Selected: Python benchmark")



# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ HumanEval Setup Complete!")
print("\n💡 To run the Benchmark:")
print(f"   python3 run_humaneval_api.py")
print("="*70)
