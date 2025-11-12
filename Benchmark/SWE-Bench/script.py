#!/usr/bin/env python3
"""
setup_swebench.py

One-step SWE-bench setup + evaluation script.

✅ Features:
- Installs Python 3.10+ (if needed)
- Sets up venv and dependencies
- Clones SWE-bench repo
- Detects CPU/GPU/macOS
- Handles FlashAttention setup (only if supported)
- Loads dataset directly from Hugging Face
- Runs evaluation with local model or API-based model
- Skips rebuilds if images already exist
"""

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
import json
import datetime

# -----------------------------------------------------------------------------
# PREREQUISITES CHECK - Run before anything else
# -----------------------------------------------------------------------------
def check_prerequisites():
    """
    Check Docker installation and running status.
    All other requirements (Python, disk space, etc.) are handled automatically by the script.
    """
    print("="*80)
    print("🔍 CHECKING DOCKER (Required for Evaluation)")
    print("="*80)
    
    system = platform.system()
    docker_installed = False
    docker_running = False
    
    print(f"\n📍 Operating System: {system} {platform.release()}")
    
    # Check Docker Installation and Status
    print("\n🐳 Docker Check:")
    
    try:
        # Check if docker command exists
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        docker_version = result.stdout.strip()
        print(f"   ✅ Docker installed: {docker_version}")
        docker_installed = True
        
        # Check if Docker daemon is running
        try:
            subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5
            )
            print("   ✅ Docker daemon is running")
            docker_running = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("   ⚠️  Docker is installed but NOT running")
            
    except FileNotFoundError:
        print("   ❌ Docker is NOT installed")
    except subprocess.CalledProcessError:
        print("   ❌ Docker command failed")
    
    # Print Summary
    print("\n" + "="*80)
    
    if docker_installed and docker_running:
        print("✅ DOCKER CHECK PASSED")
        print("="*80)
        print("� Proceeding with setup...")
        print("="*80)
        return True
    
    # Docker not installed or not running
    if not docker_installed:
        print("❌ DOCKER NOT INSTALLED")
        print("="*80)
        print("\n🐳 DOCKER INSTALLATION REQUIRED")
        print("="*80)
        
        if system == "Darwin":
            print("\n📦 macOS Installation Options:\n")
            print("Option 1: OrbStack (Recommended - Lightweight)")
            print("   1. Visit: https://orbstack.dev")
            print("   2. Download and install")
            print("   3. Open OrbStack from Applications")
            print("   4. Verify: docker info\n")
            
            print("Option 2: Docker Desktop")
            print("   1. Visit: https://www.docker.com/products/docker-desktop")
            print("   2. Download Docker Desktop for Mac")
            if platform.machine() == "arm64":
                print("      → Choose 'Apple Silicon' version")
            else:
                print("      → Choose 'Intel Chip' version")
            print("   3. Install and launch Docker Desktop")
            print("   4. Verify: docker info\n")
            
            print("Option 3: Homebrew")
            print("   brew install --cask orbstack")
            print("   # OR")
            print("   brew install --cask docker\n")
            
        elif system == "Linux":
            print("\n📦 Linux Installation:\n")
            try:
                with open("/etc/os-release") as f:
                    os_info = f.read().lower()
                
                if "ubuntu" in os_info or "debian" in os_info:
                    print("Ubuntu/Debian:")
                    print("   sudo apt update")
                    print("   sudo apt install -y docker.io")
                    print("   sudo systemctl start docker")
                    print("   sudo systemctl enable docker")
                    print("   sudo usermod -aG docker $USER")
                    print("   # Log out and back in\n")
                elif "fedora" in os_info or "rhel" in os_info:
                    print("Fedora/RHEL/CentOS:")
                    print("   sudo dnf install -y docker")
                    print("   sudo systemctl start docker")
                    print("   sudo systemctl enable docker")
                    print("   sudo usermod -aG docker $USER\n")
                else:
                    print("Visit: https://docs.docker.com/engine/install/\n")
            except:
                print("Visit: https://docs.docker.com/engine/install/\n")
        
        print("\n" + "="*80)
        print("❌ Please install Docker and re-run the script.")
        print("="*80)
        sys.exit(1)
    
    elif not docker_running:
        print("⚠️  DOCKER IS NOT RUNNING")
        print("="*80)
        print("\nDocker is required for evaluation. Please start Docker:\n")
        if system == "Darwin":
            print("   → Open Docker Desktop or OrbStack from Applications")
            print("   → Wait for the status icon to show 'Docker is running'")
        elif system == "Linux":
            print("   → sudo systemctl start docker")
        print("\nThen re-run this script.")
        print("="*80)
        
        response = input("\nContinue anyway? (Predictions will be generated but not evaluated) [y/N]: ").strip().lower()
        if response != 'y':
            sys.exit(0)
        
        print("\n" + "="*80)
        print("⚠️  Proceeding without Docker... (evaluation will be skipped)")
        print("="*80)
    
    return True

# Run prerequisites check immediately
check_prerequisites()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def run(cmd, cwd=None, check=True, env=None):
    print(f"🔹 {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check, env=env)
    return result

def find_compatible_python():
    """Find a compatible Python 3.10-3.12 installation"""
    print("\n🔍 Looking for compatible Python installation...")
    
    # Check current Python version
    current_major, current_minor = sys.version_info[:2]
    print(f"🐍 Current Python: {current_major}.{current_minor}")
    
    # List of Python versions to try (3.10-3.12 recommended for torch 2.2.2)
    python_candidates = [
        "python3.12",
        "python3.11", 
        "python3.10",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.11",
        "/usr/local/bin/python3.10",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
    ]
    
    # If current Python is 3.10-3.12, use it
    if 10 <= current_minor <= 12 and current_major == 3:
        print(f"✅ Current Python {current_major}.{current_minor} is compatible")
        return sys.executable
    
    # For Python < 3.10 or >= 3.13, search for compatible version
    if current_major == 3 and (current_minor < 10 or current_minor >= 13):
        if current_minor < 10:
            print(f"⚠️ Python {current_major}.{current_minor} is too old. Minimum required: 3.10")
        else:
            print(f"⚠️ Python 3.13+ detected. Torch 2.2.2 requires Python 3.10-3.12")
        
        print("🔍 Looking for Python 3.10-3.12 on your system...")
        
        for py_cmd in python_candidates:
            try:
                result = subprocess.run(
                    [py_cmd, "--version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                version_str = result.stdout.strip()
                # Extract version number
                import re
                match = re.search(r'Python 3\.(\d+)', version_str)
                if match:
                    minor = int(match.group(1))
                    if 10 <= minor <= 12:
                        # Get full path
                        try:
                            path_result = subprocess.run(
                                ["which", py_cmd],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            python_path = path_result.stdout.strip()
                        except:
                            # If 'which' fails, use the command as-is if it's an absolute path
                            python_path = py_cmd if py_cmd.startswith('/') else None
                            if not python_path:
                                continue
                        
                        print(f"✅ Found compatible Python: {version_str} at {python_path}")
                        return python_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        # No compatible Python found
        print("\n❌ No compatible Python 3.10-3.12 found on your system.")
        print("📦 Please install Python 3.12 using one of these methods:")
        print("\nOn macOS:")
        print("   brew install python@3.12")
        print("\nOn Ubuntu/Debian:")
        print("   sudo apt update && sudo apt install python3.12 python3.12-venv")
        print("\nOn any system with pyenv:")
        print("   pyenv install 3.12.0 && pyenv global 3.12.0")
        print("\nAfter installing, run this script again.")
        sys.exit(1)
    
    return sys.executable

def create_venv(venv_dir="venv_swebench", python_executable=None):
    """Create virtual environment using specified Python"""
    venv_path = Path(venv_dir).resolve()  # Get absolute path
    
    if not venv_path.exists():
        print(f"🔧 Creating virtual environment with {python_executable}...")
        if python_executable and python_executable != sys.executable:
            # Use subprocess to create venv with different Python
            subprocess.run([python_executable, "-m", "venv", str(venv_path)], check=True)
        else:
            venv.EnvBuilder(with_pip=True).create(str(venv_path))
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    
    print(f"📍 Virtual environment: {venv_path}")
    return venv_path

def pip_install(venv_dir, *packages):
    pip_path = venv_dir / "bin" / "pip"
    run([str(pip_path), "install", "--upgrade", "pip"])
    run([str(pip_path), "install", *packages])

def get_torch_version_for_python(python_path):
    """Determine compatible torch version based on Python version"""
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        major, minor = map(int, version.split('.'))
        
        print(f"🐍 Virtual environment Python: {major}.{minor}")
        
        if (major, minor) >= (3, 13):
            print("⚠️ Python 3.13+ detected - using latest torch")
            return "torch>=2.6.0"
        elif (major, minor) == (3, 12):
            print("✅ Python 3.12 detected - using torch 2.2.x")
            return "torch>=2.2.0,<2.6.0"
        else:  # 3.10, 3.11
            print("✅ Python 3.10/3.11 detected - using torch 2.2.2")
            return "torch==2.2.2"
    except Exception as e:
        print(f"⚠️ Could not detect Python version: {e}")
        return "torch>=2.2.0"

def log_docker_images(log_file="docker_images.log", mode="append"):
    """Log SWE-bench related Docker images to a file"""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedAt}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Filter for SWE-bench images only
        all_lines = result.stdout.split('\n')
        sweb_images = [line for line in all_lines if 'sweb' in line.lower()]
        
        # Open file in append or write mode
        file_mode = "a" if mode == "append" else "w"
        with open(log_file, file_mode) as f:
            if mode == "write":
                f.write("="*80 + "\n")
                f.write(f"SWE-bench Docker Images Log\n")
                f.write("="*80 + "\n\n")
            
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
            f.write("REPOSITORY:TAG\t\t\t\tIMAGE ID\tSIZE\t\tCREATED\n")
            f.write("-"*80 + "\n")
            if sweb_images:
                f.write('\n'.join(sweb_images) + "\n")
            else:
                f.write("No SWE-bench images found.\n")
            f.write("\n")
        
        print(f"🐳 SWE-bench images: {len(sweb_images)}")
        print(f"📝 Docker images logged to: {log_file}")
        
        return len(sweb_images)
    except Exception as e:
        print(f"⚠️ Could not log Docker images: {e}")
        return 0

# -----------------------------------------------------------------------------
# Step 1: Find compatible Python + Create Virtual Env
# -----------------------------------------------------------------------------
print("="*60)
print("SWE-bench Setup Script")
print("="*60)

python_executable = find_compatible_python()
venv_dir = create_venv(python_executable=python_executable)
# Don't resolve symlinks - use the venv Python directly
python_path = venv_dir / "bin" / "python"

# Verify venv Python version
result = subprocess.run(
    [str(python_path), "--version"],
    capture_output=True,
    text=True
)
print(f"✅ Virtual environment created with: {result.stdout.strip()}")
print(f"📍 Python path: {python_path}")

# Verify swebench is installed in this Python
check_result = subprocess.run(
    [str(python_path), "-c", "import swebench; print(f'✅ swebench {swebench.__version__} installed')"],
    capture_output=True,
    text=True
)
if check_result.returncode == 0:
    print(check_result.stdout.strip())
else:
    print("ℹ️ swebench not yet installed (will be installed in Step 5)")

# -----------------------------------------------------------------------------
# Step 2: Install base dependencies
# -----------------------------------------------------------------------------
print("\n📦 Installing base dependencies...")

# Determine compatible torch version
torch_version = get_torch_version_for_python(python_path)

base_packages = [
    "numpy<2.0",  # Fix NumPy compatibility with older torch
    torch_version,
    "transformers>=4.36.0",
    "accelerate",
    "datasets",
    "docker",
    "gitpython",
    "requests",
    "tqdm",
    "beautifulsoup4",  # Required by SWE-bench (bs4)
    "chardet",         # Required by SWE-bench
    "tiktoken",        # Required by run_api.py
    "openai",          # Required by run_api.py
    "anthropic",       # Required by run_api.py for Claude models
    "python-dotenv",   # Required by run_api.py
    "tenacity",        # Required by run_api.py for retries
]

print("📦 Installing:")
for pkg in base_packages:
    print(f"  - {pkg}")

pip_install(venv_dir, *base_packages)

# -----------------------------------------------------------------------------
# Step 3: Optional FlashAttention setup
# -----------------------------------------------------------------------------
if platform.system() != "Darwin":  # Not macOS
    try:
        import torch
        if torch.cuda.is_available():
            print("🖥️ CUDA detected. Installing FlashAttention...")
            pip_install(venv_dir, "flash-attn==2.3.6")
        else:
            print("ℹ️ CUDA not available. Skipping FlashAttention.")
    except Exception as e:
        print(f"⚠️ FlashAttention skipped due to error: {e}")
else:
    print("🍎 CUDA and FlashAttention not supported. Running on CPU.")

# -----------------------------------------------------------------------------
# Step 4: Clone SWE-bench
# -----------------------------------------------------------------------------
repo_url = "https://github.com/princeton-nlp/SWE-bench.git"
repo_dir = Path("SWE-bench")
if not repo_dir.exists():
    print("📥 Cloning SWE-bench repository...")
    run(["git", "clone", repo_url])
else:
    print("✅ SWE-bench repo already exists. Skipping clone.")

# -----------------------------------------------------------------------------
# Step 5: Install SWE-bench
# -----------------------------------------------------------------------------
print("\n📦 Installing SWE-bench (editable mode)...")
pip_install(venv_dir, "-e", str(repo_dir))

# -----------------------------------------------------------------------------
# Step 5.5: Create .env configuration file
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("=== API Configuration Setup ===")
print("="*60)
print("Creating .env file for API credentials...")
print("This will be used for API-based inference (you can skip if using local model)")
print("="*60)

env_file_path = Path(".env")

# Check if .env already exists
if env_file_path.exists():
    print(f"\n⚠️  .env file already exists!")
    print(f"📄 Current file: {env_file_path.resolve()}")
    overwrite = input("\nDo you want to OVERWRITE the existing .env file? (y/n, default: n): ").strip().lower()
    
    if overwrite != "y":
        print("✅ Keeping existing .env file. Your configuration is safe!")
        print("ℹ️  The script will use settings from your existing .env file")
        configure_api = "n"
    else:
        print("⚠️  Proceeding to overwrite existing .env file...")
        configure_api = "y"
else:
    # Ask if user wants to configure API now
    configure_api = input("\nDo you want to configure API settings now? (y/n, default: y): ").strip().lower() or "y"

if configure_api == "y":
    print("\n⚙️ Please enter your API configuration:")
    print("(Press Enter to skip any optional field)")
    
    api_base_url = input("\nAPI Base URL (e.g., https://api.openai.com/v1 or your custom endpoint): ").strip()
    api_key = input("API Key: ").strip()
    model_name = input("Model Name (e.g., gpt-4, claude-3-sonnet, or your-custom-model): ").strip()
    
    print("\n⚙️ Optional parameters (press Enter to use defaults):")
    temperature = input("Temperature (default: 0.2): ").strip() or "0.2"
    max_tokens = input("Max Tokens (default: 1000): ").strip() or "1000"
    num_instances = input("Number of Instances (leave empty to be asked each time): ").strip()
    
    # Create .env file
    with open(env_file_path, 'w') as f:
        f.write("# SWE-bench API Configuration\n")
        f.write("# Generated automatically during setup\n\n")
        
        if api_base_url:
            f.write(f"API_BASE_URL={api_base_url}\n")
        if api_key:
            f.write(f"API_KEY={api_key}\n")
        if model_name:
            f.write(f"MODEL_NAME={model_name}\n")
        
        f.write(f"\n# Optional Parameters\n")
        f.write(f"TEMPERATURE={temperature}\n")
        f.write(f"MAX_TOKENS={max_tokens}\n")
        
        if num_instances:
            f.write(f"NUM_INSTANCES={num_instances}\n")
        else:
            f.write(f"# NUM_INSTANCES=10\n")
    
    print(f"\n✅ Configuration saved to .env file")
    print(f"ℹ️ You can edit .env file anytime to update your settings")
else:
    print("\nℹ️ Skipping API configuration. You'll be prompted during inference.")
    print("ℹ️ You can create .env file later using .env.example as template")

# -----------------------------------------------------------------------------
# Step 6: Choose dataset
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("=== Choose SWE-bench Dataset ===")
print("="*60)
print("1. SWE-bench Lite (300 instances - recommended for testing)")
print("2. SWE-bench Verified (500 instances - verified test cases)")
print("3. SWE-bench Full (2,294 instances - complete benchmark)")
print("="*60)
dataset_choice = input("Enter choice (1/2/3, default 1): ").strip() or "1"

if dataset_choice == "1":
    dataset_name = "princeton-nlp/SWE-bench_Lite"
    max_instances_available = 300
    print(f"✅ Selected: SWE-bench Lite ({max_instances_available} instances)")
elif dataset_choice == "2":
    dataset_name = "princeton-nlp/SWE-bench_Verified"
    max_instances_available = 500
    print(f"✅ Selected: SWE-bench Verified ({max_instances_available} instances)")
elif dataset_choice == "3":
    dataset_name = "princeton-nlp/SWE-bench"
    max_instances_available = 2294
    print(f"✅ Selected: SWE-bench Full ({max_instances_available} instances)")
    print("⚠️ This will take significant time and resources!")
else:
    print("❌ Invalid choice, defaulting to SWE-bench Lite")
    dataset_name = "princeton-nlp/SWE-bench_Lite"
    max_instances_available = 300

print(f"📊 Dataset: {dataset_name}")

# -----------------------------------------------------------------------------
# Step 7: Choose evaluation mode and generate predictions
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("=== Choose Evaluation Mode ===")
print("="*60)
print("1. Use a local model (run inference locally)")
print("2. Use a model through API (remote inference)")
print("3. Use an already existing predictions file")
print("="*60)
mode = input("Enter choice (1/2/3): ").strip()

if mode == "1":
    # Local model inference
    print("\n🖥️  Local Model Inference")
    print("="*60)
    model_path = input("Enter path to local model Enter path to local model (e.g., ./models/llama-7b): ").strip()
    num_instances = int(input(f"Number of instances to evaluate Number of instances to evaluate (default 10, max {max_instances_available}): ").strip() or "10")
    
    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        sys.exit(1)
    
    model_name = Path(model_path).name
    predictions_path = f"predictions_{model_name.replace('/', '_').replace(' ', '_')}.json"
    
    print(f"\n🤖 Running local inference with: {model_name}")
    print(f"Number of instances to evaluate Processing {num_instances} instances from {dataset_name}")
    print("⚠️  Note: Local inference requires significant compute resources")
    
    # Create local inference script
    gen_script = Path("run_local.py")
    gen_script.write_text(f'''
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

model_path = "{model_path}"
dataset_name = "{dataset_name}"
num_instances = {num_instances}
output_file = "{predictions_path}"

print(f"Loading model from {{model_path}}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Loading dataset...")
dataset = load_dataset(dataset_name, split="test")
if num_instances < len(dataset):
    dataset = dataset.select(range(num_instances))

predictions = []

for idx, instance in enumerate(tqdm(dataset, desc="Processing instances")):
    prompt = f"""You are an expert software engineer. Given the following issue from a GitHub repository, provide a code patch to fix the issue.

Issue:
{{instance['problem_statement']}}

Repository: {{instance['repo']}}
Base commit: {{instance['base_commit']}}

Please provide only the code patch in unified diff format."""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=4000,
            temperature=0.2,
            do_sample=True
        )
        model_patch = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        model_patch = model_patch[len(prompt):].strip()
        
        predictions.append({{
            "instance_id": instance["instance_id"],
            "model_patch": model_patch,
            "model_name_or_path": "{model_name}"
        }})
        
        print(f"✅ Completed instance {{idx + 1}}/{{len(dataset)}}")
        
    except Exception as e:
        print(f"⚠️ Error processing instance {{instance['instance_id']}}: {{e}}")
        predictions.append({{
            "instance_id": instance["instance_id"],
            "model_patch": "",
            "model_name_or_path": "{model_name}"
        }})

with open(output_file, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\\n✅ Predictions saved to: {{output_file}}")
print(f"Number of instances to evaluate Total predictions: {{len(predictions)}}")
''')
    
    # Run local inference
    print("\n🔹 Running local model inference...")
    abs_python_path = Path(python_path).resolve()
    abs_gen_script = gen_script.resolve()
    abs_repo_dir = repo_dir.resolve()
    
    run([str(abs_python_path), str(abs_gen_script)], cwd=str(abs_repo_dir))
    gen_script.unlink()
    
    print(f"\n✅ Predictions generated: {repo_dir / predictions_path}")
    
elif mode == "2":
    # API model inference using SWE-bench's run_api.py (patched for custom models)
    print("\n🌐 API Model Inference")
    print("="*60)
    
    # Load environment variables from .env file
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("✅ Loaded configuration from .env file")
    else:
        print("ℹ️ No .env file found. You can create one from .env.example")
        print("ℹ️ For now, please enter configuration manually")
    
    # Read from environment or prompt user
    api_base = os.getenv("API_BASE_URL")
    if not api_base:
        api_base = input("Enter API Base URL (e.g., https://api.openai.com/v1 or your custom endpoint): ").strip()
    else:
        print(f"⚙️ Using API Base URL from .env: {api_base}")
    
    # Remove /chat/completions if user included it
    if api_base.endswith('/chat/completions'):
        api_base = api_base[:-len('/chat/completions')]
    if api_base.endswith('/completions'):
        api_base = api_base[:-len('/completions')]
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        api_key = input("Enter API Key: ").strip()
    else:
        print(f"⚙️ Using API Key from .env: {'*' * (len(api_key) - 5) + api_key[-5:]}")
    
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        model_name = input("Enter Model Name: ").strip()
    else:
        print(f"⚙️ Using Model from .env: {model_name}")
    
    # Number of instances
    num_instances_env = os.getenv("NUM_INSTANCES")
    if num_instances_env:
        num_instances = int(num_instances_env)
        print(f"⚙️ Using NUM_INSTANCES from .env: {num_instances}")
    else:
        num_instances = int(input(f"Number of instances to evaluate (default 10, max {max_instances_available}): ").strip() or "10")
    
    # Temperature setting
    temperature_env = os.getenv("TEMPERATURE")
    if temperature_env:
        temperature = float(temperature_env)
        print(f"⚙️ Using TEMPERATURE from .env: {temperature}")
    else:
        temperature_input = input("Temperature (default 0.2, press Enter to skip): ").strip()
        temperature = float(temperature_input) if temperature_input else 0.2
    
    # Max tokens
    max_tokens_env = os.getenv("MAX_TOKENS")
    if max_tokens_env:
        max_tokens = int(max_tokens_env)
        print(f"⚙️ Using MAX_TOKENS from .env: {max_tokens}")
    else:
        max_tokens_input = input("Max tokens (default 1000, recommended 500-2000, press Enter to skip): ").strip()
        max_tokens = int(max_tokens_input) if max_tokens_input else 1000
    
    predictions_path = f"predictions_{model_name.replace('/', '_')}.jsonl"
    
    print(f"\n🤖 Using SWE-bench's run_api.py (patched for custom models)")
    print(f"⚙️ {num_instances} instances from {dataset_name}")
    print(f"🌐 API Base: {api_base}")
    print(f"🤖 Model: {model_name}")
    print(f"🌡️ Temperature: {temperature}")
    print(f"📝 Max Tokens: {max_tokens}")
    
    # Patch run_api.py to accept custom models using sed commands
    run_api_script = repo_dir / "swebench" / "inference" / "run_api.py"
    
    if not run_api_script.exists():
        print(f"❌ run_api.py not found at: {run_api_script}")
        print("   Please ensure SWE-bench is properly cloned.")
        sys.exit(1)
    
    print("🔧 Patching run_api.py to add custom model...")
    
    # Read the file
    with open(run_api_script, 'r') as f:
        content = f.read()
    
    # Create backup
    with open(f"{run_api_script}.backup", 'w') as f:
        f.write(content)
    
    # Apply patches to add custom model properly
    
    # 1. Add custom model to MODEL_LIMITS dictionary
    import re
    model_limits_pattern = r'(MODEL_LIMITS = \{[^}]+)(}\s*MODEL_COST_PER_INPUT)'
    model_limits_match = re.search(model_limits_pattern, content, re.DOTALL)
    if model_limits_match:
        # Add the custom model before the closing brace
        new_limits = model_limits_match.group(1) + f'\n    "{model_name}": 100_000,  # Custom model added\n' + model_limits_match.group(2)
        content = content[:model_limits_match.start()] + new_limits + content[model_limits_match.end():]
        print(f"  ✅ Added {model_name} to MODEL_LIMITS")
    
    # 2. Add custom model to MODEL_COST_PER_INPUT
    cost_input_pattern = r'(MODEL_COST_PER_INPUT = \{[^}]+)(}\s*MODEL_COST_PER_OUTPUT)'
    cost_input_match = re.search(cost_input_pattern, content, re.DOTALL)
    if cost_input_match:
        new_cost_input = cost_input_match.group(1) + f'\n    "{model_name}": 0,  # Custom model\n' + cost_input_match.group(2)
        content = content[:cost_input_match.start()] + new_cost_input + content[cost_input_match.end():]
        print(f"  ✅ Added {model_name} to MODEL_COST_PER_INPUT")
    
    # 3. Add custom model to MODEL_COST_PER_OUTPUT
    cost_output_pattern = r'(MODEL_COST_PER_OUTPUT = \{[^}]+)(})'
    cost_output_match = re.search(cost_output_pattern, content, re.DOTALL)
    if cost_output_match:
        new_cost_output = cost_output_match.group(1) + f'\n    "{model_name}": 0,  # Custom model\n' + cost_output_match.group(2)
        content = content[:cost_output_match.start()] + new_cost_output + content[cost_output_match.end():]
        print(f"  ✅ Added {model_name} to MODEL_COST_PER_OUTPUT")
    
    # 3b. Patch calc_cost function to handle ANY unknown model gracefully
    # Simply replace the dictionary lookups with .get() method
    content = content.replace(
        'MODEL_COST_PER_INPUT[model_name] * input_tokens',
        'MODEL_COST_PER_INPUT.get(model_name, 0) * input_tokens'
    )
    content = content.replace(
        'MODEL_COST_PER_OUTPUT[model_name] * output_tokens',
        'MODEL_COST_PER_OUTPUT.get(model_name, 0) * output_tokens'
    )
    print(f"  ✅ Patched calc_cost() to use .get() for unknown models (graceful handling)")
    
    # 4. Remove choices constraint from argparse
    content = content.replace(
        'choices=sorted(list(MODEL_LIMITS.keys())),',
        '# choices removed to support custom models'
    )
    
    # 5. Import OpenAI client at the top
    content = content.replace(
        'import openai',
        '''import openai
from openai import OpenAI as OpenAIClient'''
    )
    
    # 6. Fix main() text column issue
    content = content.replace(
        '    lens = np.array(list(map(len, dataset["text"])))',
        '''    # Handle both 'text' and 'problem_statement' columns
    text_column = "text" if "text" in dataset.column_names else "problem_statement"
    lens = np.array(list(map(len, dataset[text_column])))'''
    )
    
    # 6b. CRITICAL: Limit the test_dataset right after filtering
    # This happens AFTER filtering out already-completed instances
    # We need to store the original count before limiting
    content = content.replace(
        '    print(f"Filtered to {len(test_dataset)} instances")',
        f'''    # Store original count before limiting
    original_count = len(test_dataset)
    # Limit to user-requested number of instances
    if original_count > {num_instances}:
        test_dataset = test_dataset.select(range({num_instances}))
        print(f"🔹 Limited to {num_instances} instances (from {{original_count}} available)")
    print(f"Filtered to {{len(test_dataset)}} instances")'''
    )
    
    # 6c. Fix model routing in main() to handle custom models
    content = content.replace(
        '''        raise ValueError(f"Invalid model name or path {model_name_or_path}")''',
        f'''        # Handle custom model {model_name}
        if model_name_or_path == "{model_name}":
            openai_inference(**inference_args)
        else:
            raise ValueError(f"Invalid model name or path {{model_name_or_path}}")'''
    )
    
    # 7. Replace the tiktoken and OpenAI setup in openai_inference function
    old_openai_setup = '''    encoding = tiktoken.encoding_for_model(model_name_or_path)
    test_dataset = test_dataset.filter(
        lambda x: gpt_tokenize(x["text"], encoding) <= MODEL_LIMITS[model_name_or_path],
        desc="Filtering",
        load_from_cache_file=False,
    )
    openai_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_key is None:
        raise ValueError(
            "Must provide an api key. Expected in OPENAI_API_KEY environment variable."
        )
    openai.api_key = openai_key
    print(f"Using OpenAI key {{'*' * max(0, len(openai_key) - 5) + openai_key[-5:]}}")
    use_azure = model_args.pop("use_azure", False)
    if use_azure:
        openai.api_type = "azure"
        openai.api_base = "https://pnlpopenai3.openai.azure.com/"
        openai.api_version = "2023-05-15"'''
    
    new_openai_setup = f'''    # Handle custom model with special API endpoint
    if model_name_or_path == "{model_name}":
        filtered_dataset = test_dataset
        print(f"Skipping tokenization for custom model {model_name}")
        # Use custom API credentials
        client = OpenAIClient(
            api_key="{api_key}",
            base_url="{api_base}"
        )
        print(f"Using custom API at {api_base}")
        use_azure = False
    else:
        # Standard OpenAI setup
        encoding = tiktoken.encoding_for_model(model_name_or_path)
        text_column = "text" if "text" in test_dataset.column_names else "problem_statement"
        filtered_dataset = test_dataset.filter(
            lambda x: gpt_tokenize(x[text_column], encoding) <= MODEL_LIMITS[model_name_or_path],
            desc="Filtering",
            load_from_cache_file=False,
        )
        openai_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_key is None:
            raise ValueError(
                "Must provide an api key. Expected in OPENAI_API_KEY environment variable."
            )
        client = OpenAIClient(api_key=openai_key)
        print(f"Using OpenAI key {{'*' * max(0, len(openai_key) - 5) + openai_key[-5:]}}")
        use_azure = model_args.pop("use_azure", False)
        if use_azure:
            openai.api_type = "azure"
            openai.api_base = "https://pnlpopenai3.openai.azure.com/"
            openai.api_version = "2023-05-15"'''
    
    if old_openai_setup in content:
        content = content.replace(old_openai_setup, new_openai_setup)
        print(f"  ✅ Added custom client initialization for {model_name}")
    else:
        print(f"  ⚠️  Could not find exact match for OpenAI setup - trying line-by-line approach")
        # Try a simpler replacement at just the encoding line
        content = content.replace(
            '    encoding = tiktoken.encoding_for_model(model_name_or_path)\n    test_dataset = test_dataset.filter(',
            f'''    # Handle custom model
    if model_name_or_path == "{model_name}":
        filtered_dataset = test_dataset
        print(f"Skipping tokenization for custom model")
        client = OpenAIClient(api_key="{api_key}", base_url="{api_base}")
        print(f"Using custom API at {api_base}")
        use_azure = False
    else:
        encoding = tiktoken.encoding_for_model(model_name_or_path)
        filtered_dataset = test_dataset.filter('''
        )
        # Also need to update the openai.api_key line to use client - handle both cases
        if '    openai.api_key = openai_key\n    print(f"Using OpenAI key' in content:
            content = content.replace(
                '    openai.api_key = openai_key\n    print(f"Using OpenAI key',
                f'''        client = OpenAIClient(api_key=openai_key)
        print(f"Using OpenAI key'''
            )
        # Also handle if use_azure block needs fixing - with all lines indented
        if '    use_azure = model_args.pop("use_azure", False)\n    if use_azure:\n        openai.api_type = "azure"\n        openai.api_base = "https://pnlpopenai3.openai.azure.com/"\n        openai.api_version = "2023-05-15"' in content:
            content = content.replace(
                '    use_azure = model_args.pop("use_azure", False)\n    if use_azure:\n        openai.api_type = "azure"\n        openai.api_base = "https://pnlpopenai3.openai.azure.com/"\n        openai.api_version = "2023-05-15"',
                '''        use_azure = model_args.pop("use_azure", False)
        if use_azure:
            openai.api_type = "azure"
            openai.api_base = "https://pnlpopenai3.openai.azure.com/"
            openai.api_version = "2023-05-15"'''
            )
        print(f"  ✅ Applied alternative patching approach")
    
    # 8. The loop should already use test_dataset which we limited in step 6b
    # No need to change variable names since test_dataset is already limited
    
    # 9. Fix text access in inference loop
    content = content.replace(
        '''            output_dict["text"] = f"{datum['text']}\\n\\n"''',
        '''            # Handle both text and problem_statement
            text_content = datum.get('text') or datum.get('problem_statement', '')
            output_dict["text"] = f"{text_content}\\n\\n"'''
    )
    
    # 10. Update call_chat to use the client variable and explicitly handle max_tokens
    old_call_chat = '''def call_chat(model_name_or_path, inputs, use_azure, temperature, top_p, **model_args):
    """
    Calls the openai API to generate completions for the given inputs.

    Args:
    model_name_or_path (str): The name or path of the model to use.
    inputs (str): The inputs to generate completions for.
    use_azure (bool): Whether to use the azure API.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    **model_args (dict): A dictionary of model arguments.
    """
    system_messages = inputs.split("\\n", 1)[0]
    user_message = inputs.split("\\n", 1)[1]
    try:
        if use_azure:
            response = openai.chat.completions.create('''
    
    new_call_chat = '''def call_chat(model_name_or_path, inputs, use_azure, temperature, top_p, client=None, **model_args):
    """
    Calls the openai API to generate completions for the given inputs.

    Args:
    model_name_or_path (str): The name or path of the model to use.
    inputs (str): The inputs to generate completions for.
    use_azure (bool): Whether to use the azure API.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    client: OpenAI client instance (optional).
    **model_args (dict): A dictionary of model arguments.
    """
    system_messages = inputs.split("\\n", 1)[0]
    user_message = inputs.split("\\n", 1)[1]
    
    # Extract max_tokens explicitly from model_args
    max_tokens = model_args.pop("max_tokens", 500)  # Default to 500 if not specified
    print(f"🐛 Using max_tokens={max_tokens} for model {model_name_or_path}")
    
    # Add format instruction to ensure proper git diff output
    format_instruction = "\\n\\nIMPORTANT: Your response MUST be a valid git unified diff patch starting with 'diff --git a/path/to/file b/path/to/file'. Do not include explanations, comments, or any text outside the diff format."
    system_messages = system_messages + format_instruction
    
    if client is None:
        client = openai
    try:
        if use_azure:
            response = client.chat.completions.create('''
    
    content = content.replace(old_call_chat, new_call_chat)
    
    # 11. Replace all openai.chat.completions.create with client.chat.completions.create
    # AND add max_tokens parameter explicitly to both Azure and regular API calls
    content = content.replace(
        'response = openai.chat.completions.create(',
        'response = client.chat.completions.create('
    )
    
    # Add max_tokens to Azure API call
    content = content.replace(
        '''            response = client.chat.completions.create(
                engine=ENGINES[model_name_or_path] if use_azure else None,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                **model_args,
            )''',
        '''            response = client.chat.completions.create(
                engine=ENGINES[model_name_or_path] if use_azure else None,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **model_args,
            )'''
    )
    
    # Add max_tokens to regular API call
    content = content.replace(
        '''            response = client.chat.completions.create(
                model=model_name_or_path,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                **model_args,
            )''',
        '''            response = client.chat.completions.create(
                model=model_name_or_path,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **model_args,
            )'''
    )
    
    # 12. Update the call_chat invocation to pass the client
    content = content.replace(
        '''            response, cost = call_chat(
                output_dict["model_name_or_path"],
                output_dict["text"],
                use_azure,
                temperature,
                top_p,
            )''',
        '''            response, cost = call_chat(
                output_dict["model_name_or_path"],
                output_dict["text"],
                use_azure,
                temperature,
                top_p,
                client=client,
            )'''
    )
    
    # 13. Simplify output to only include the 3 required fields (no validation, just field selection)
    old_output_section = '''            output_dict["full_output"] = completion
            output_dict["model_patch"] = extract_diff(completion)
            print(json.dumps(output_dict), file=f, flush=True)'''
    
    new_output_section = '''            # Write only the 3 required fields for SWE-bench format
            minimal_output = {
                "instance_id": output_dict["instance_id"],
                "model_name_or_path": output_dict["model_name_or_path"],
                "model_patch": extract_diff(completion)
            }
            print(json.dumps(minimal_output), file=f, flush=True)'''
    
    content = content.replace(old_output_section, new_output_section)
    
    # Write patched content
    with open(run_api_script, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Successfully patched run_api.py with custom model {model_name}")
    print(f"   Backup saved at: {run_api_script}.backup")
    
    # Set up environment variables for API
    api_env = os.environ.copy()
    api_env["OPENAI_API_KEY"] = api_key
    api_env["OPENAI_API_BASE"] = api_base
    
    # Prepare output directory
    output_dir = repo_dir.resolve()
    
    # Build command for run_api.py - use venv Python directly without resolving symlinks
    abs_run_api_script = run_api_script.resolve()
    
    api_cmd = [
        str(python_path),
        str(abs_run_api_script),
        "--dataset_name_or_path", dataset_name,
        "--model_name_or_path", model_name,
        "--output_dir", str(output_dir),
        "--model_args", f"temperature={temperature},max_tokens={max_tokens}",
    ]
    
    # Note: We'll let run_api.py process all available instances
    # The dataset was already filtered during loading based on num_instances
    print(f"ℹ️ Processing {num_instances} instances from {dataset_name}")
    
    # Check for existing predictions file and handle it
    expected_output_file = output_dir / f"{model_name.replace('/', '_')}__SWE-bench_Lite__test.jsonl"
    if expected_output_file.exists():
        print(f"\n⚠️  Existing predictions file found: {expected_output_file.name}")
        print("   This may cause issues if the file has incomplete entries.")
        
        # Check if file has empty lines (common issue)
        try:
            with open(expected_output_file, 'r') as f:
                lines = f.readlines()
                non_empty_lines = [l for l in lines if l.strip()]
                if len(lines) != len(non_empty_lines):
                    print(f"   ⚠️  File has {len(lines) - len(non_empty_lines)} empty line(s) - this will cause JSON parsing errors!")
        except:
            pass
        
        response = input("   Options: [r]emove and start fresh, [k]eep and resume, [q]uit (default: r): ").strip().lower() or 'r'
        if response == 'r':
            backup_file = Path(str(expected_output_file) + ".backup")
            import shutil
            shutil.copy(expected_output_file, backup_file)
            expected_output_file.unlink()
            print(f"   ✅ Backed up to {backup_file.name} and removed")
        elif response == 'q':
            print("   Exiting...")
            sys.exit(0)
        else:
            print("   ⚠️  Keeping existing file - run_api.py will attempt to resume")
    
    print("\n🔹 Running SWE-bench's run_api.py...")
    print(f"   Command: {' '.join(api_cmd)}")
    print("\n" + "="*60)
    print("⏳ This may take several minutes...")
    print("💡 The script will process each instance and call the API")
    print(f"📊 Processing {num_instances} instances")
    print("="*60 + "\n")
    
    # Run API inference with REAL-TIME output (don't capture)
    try:
        # Use run() helper which prints the command and shows real-time output
        result = subprocess.run(
            api_cmd, 
            env=api_env, 
            cwd=str(repo_dir),
            check=True  # Will raise exception on error
        )
        print(f"\n✅ API inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ run_api.py failed with exit code {e.returncode}")
        print("\n💡 Common issues:")
        print("   • Check if API endpoint is accessible")
        print("   • Verify API key is correct")
        print("   • Ensure model name matches API requirements")
        print("   • Check if dataset can be loaded")
        print("   • Review the error output above for details")
        print(f"\n💡 Tip: Check the patched run_api.py at: {run_api_script}")
        raise
    finally:
        # Restore original run_api.py after inference (whether it succeeds or fails)
        print("\n🔹 Restoring original run_api.py...")
        if Path(f"{run_api_script}.backup").exists():
            import shutil
            shutil.copy(f"{run_api_script}.backup", run_api_script)
            print("✅ Original run_api.py restored")
        else:
            print("⚠️  Backup not found, skipping restore")
    
    # Find the generated predictions file
    expected_files = [
        output_dir / f"all_preds.jsonl",
        output_dir / f"{model_name.replace('/', '_')}_preds.jsonl",
        output_dir / predictions_path,
    ]
    
    predictions_file = None
    for f in expected_files:
        if f.exists():
            predictions_file = f
            break
    
    if predictions_file:
        print(f"\n✅ Predictions file found: {predictions_file}")
        # Update predictions_path to absolute path for evaluation
        predictions_path = str(predictions_file.resolve())
    else:
        print(f"⚠️ Predictions file not found. Checking output directory...")
        # Look for any .jsonl files that might be predictions
        pred_files = list(output_dir.glob("*.jsonl"))
        # Filter for files that contain the model name
        pred_files = [f for f in pred_files if model_name.replace('-', '_') in f.name or 'SWE-bench' in f.name]
        if pred_files:
            predictions_file = pred_files[0]
            print(f"✅ Found predictions file: {predictions_file}")
            predictions_path = str(predictions_file.resolve())
        else:
            # Try any jsonl file as last resort
            all_jsonl = list(output_dir.glob("*.jsonl"))
            if all_jsonl:
                predictions_file = all_jsonl[0]
                print(f"✅ Found predictions file: {predictions_file}")
                predictions_path = str(predictions_file.resolve())
            else:
                print(f"❌ No predictions file found in {output_dir}")
                sys.exit(1)
    
elif mode == "3":
    predictions_path = input("Enter path to local model Enter path to existing predictions file: ").strip()
    if not Path(predictions_path).exists():
        print(f"❌ File not found: {predictions_path}")
        sys.exit(1)
    print(f"✅ Using existing predictions: {predictions_path}")
    
else:
    print("❌ Invalid choice")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Step 8: Check Docker setup
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("=== Docker Setup Check ===")
print("="*60)
try:
    run(["docker", "info"], check=True)
    print("🐳 Docker is active.")
except Exception:
    print("❌ Docker not detected or not running. Please start Docker Desktop.")
    sys.exit(1)

# Log Docker images before building
print("\n📸 Logging SWE-bench Docker images (before building)...")
images_before = log_docker_images("docker_images.log", mode="write")

# -----------------------------------------------------------------------------
# Step 8.5: Detect hardware and check existing images
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("=== Hardware & Image Detection ===")
print("="*60)

# Detect hardware
is_arm = platform.processor() == "arm" or platform.machine() == "arm64"
is_mac = platform.system() == "Darwin"

# Check for GPU
has_gpu = False
gpu_info = "CPU only"
try:
    import torch
    if torch.cuda.is_available():
        has_gpu = True
        gpu_info = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
except:
    pass

print(f"🖥️  Platform: {platform.system()} {platform.machine()}")
print(f"🔧 Architecture: {'ARM (Apple Silicon)' if is_arm else 'x86_64'}")
print(f"🎮 GPU: {gpu_info}")

# Check for existing Docker images
print("\n� Checking for existing Docker images...")
try:
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}"],
        capture_output=True,
        text=True,
        check=True
    )
    existing_repos = result.stdout.strip().split('\n')
    has_aorwall = any('aorwall' in repo for repo in existing_repos)
    has_sweb_eval = any('sweb.eval' in repo for repo in existing_repos)
    
    if has_aorwall:
        print("✅ Found existing aorwall/* images (ARM-compatible)")
    if has_sweb_eval and not has_aorwall:
        print("✅ Found existing sweb.eval.* images (standard)")
    if not has_aorwall and not has_sweb_eval:
        print("ℹ️  No existing SWE-bench images found")
except:
    has_aorwall = False
    has_sweb_eval = False
    print("⚠️  Could not check existing images")

# -----------------------------------------------------------------------------
# Step 9: Choose Docker namespace and build strategy
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("=== Docker Image Build Configuration ===")
print("="*60)

# Determine recommendation based on existing images
print("\n💡 Docker Image Strategy:")
if has_sweb_eval:
    print("📦 You have standard local images - can skip build and use them")
elif has_aorwall and not has_sweb_eval:
    print("⚠️  You have 'aorwall/*' images that may cause Docker pull issues")
    print("   Recommend: Rebuild as standard local images")
elif has_gpu:
    print("� GPU detected - will build standard local images")
else:
    print("🖥️  Will build standard local images")

print("\n" + "="*60)
print("Choose Docker build option:")
print("="*60)
print("1. Build new images - Standard local build (sweb.eval.*)")
print("2. Skip build - Use existing images (if already built)")
print("="*60)
print("\n✅ All images are built as PURELY LOCAL:")
print("   sweb.eval.x86_64.pytest-dev_1776_pytest-5221:swebench-instance")
print("   No remote namespaces, no pulling from DockerHub!")
print("   Works perfectly on macOS, Linux, x86, ARM, with or without GPU!")
print("="*60)

# Set smart default
if has_sweb_eval or has_aorwall:
    default_choice = "2"  # Use existing images
else:
    default_choice = "1"  # Build new images

build_choice = input(f"\nEnter choice (1/2, default: {default_choice}): ").strip() or default_choice

# Always use no namespace (pure local images)
namespace = None
skip_build = False

if build_choice == "1":
    skip_build = False
    print("\n✅ Building standard local images (no namespace)")
    print("   Images will be: sweb.eval.x86_64.*")
    print("   100% local builds - no remote dependencies")
elif build_choice == "2":
    skip_build = True
    print("\n✅ Skipping build - will use existing local images")
    if has_sweb_eval:
        print("   Using: sweb.eval.x86_64.* images")
    elif has_aorwall:
        print("   Using: aorwall/* images (may cause pull issues during eval)")
    else:
        print("   Using: any existing local images")
else:
    skip_build = False
    print("\n⚠️  Invalid choice, building standard local images")

# Set up Docker environment for OrbStack
docker_env = os.environ.copy()
orbstack_socket = Path.home() / ".orbstack" / "run" / "docker.sock"
if orbstack_socket.exists():
    docker_env["DOCKER_HOST"] = f"unix://{orbstack_socket}"
    print(f"🐳 Using OrbStack Docker socket: {orbstack_socket}")

if not skip_build:
    # -----------------------------------------------------------------------------
    # Step 9: Build Docker images for SWE-bench instances
    # -----------------------------------------------------------------------------
    print("\n" + "="*60)
    print("🔨 Building Docker Images for SWE-bench")
    print("="*60)
    print("📦 This will create three types of images:")
    print("   1. Base image: Common dependencies for all evaluations")
    print("   2. Environment images: Python environments for different configurations (~60 images)")
    print("   3. Instance images: Specific dependencies for each evaluation task")
    print("⏱️  This process may take 30-60 minutes depending on the number of instances")
    print(f"🐳 Current Docker images: {images_before}")

    # Prepare image build command
    image_cmd = [
        str(python_path),
        "-m",
        "swebench.harness.prepare_images",
        "--dataset_name", dataset_name,
        "--split", "test",
        "--max_workers", "2",  # Reduced from 8 to prevent OOM errors
        "--env_image_tag", "swebench-base",
        "--tag", "swebench-instance",
    ]

    # Extract instance IDs from predictions file to only build images for those instances
    if mode in ["1", "2"]:  # Only if we generated predictions
        print(f"\n📊 Extracting instance IDs from predictions file...")
        try:
            import json
            instance_ids = []
            with open(predictions_path, 'r') as f:
                # Handle both JSON and JSONL formats
                if predictions_path.endswith('.jsonl'):
                    for line in f:
                        if line.strip():
                            pred = json.loads(line)
                            instance_ids.append(pred['instance_id'])
                else:
                    predictions = json.load(f)
                    instance_ids = [pred['instance_id'] for pred in predictions]
            
            if instance_ids:
                print(f"✅ Found {len(instance_ids)} instances: {', '.join(instance_ids[:5])}{'...' if len(instance_ids) > 5 else ''}")
                image_cmd.extend(["--instance_ids"] + instance_ids)
            else:
                print("⚠️  No instance IDs found in predictions, building all images")
        except Exception as e:
            print(f"⚠️  Could not extract instance IDs: {e}")
            print("   Building images for all instances in dataset")

    # No namespace - always build pure local images
    print(f"📦 Building WITHOUT namespace (pure local images)")
    print(f"   Images will be: sweb.eval.x86_64.*")
    
    print("\n🔹 Building Docker images...")
    print(f"   Command: {' '.join(image_cmd)}")

    # Run image building
    run(image_cmd, cwd=str(repo_dir), env=docker_env)
    
    # Log Docker images after building
    print("\n📸 Logging SWE-bench Docker images (after building)...")
    images_after_build = log_docker_images("docker_images.log", mode="append")
    new_images = images_after_build - images_before

    print(f"\n✅ Docker images built successfully!")
    print(f"   • Images before: {images_before}")
    print(f"   • Images after: {images_after_build}")
    print(f"   • New images created: {new_images}")
else:
    print("\n⏭️  Skipping Docker image build - using existing images")
    images_after_build = images_before
    new_images = 0  # No new images when skipping build

# Verify required images exist (namespace is always None now)
if mode in ["1", "2"]:
    print(f"\n🔍 Verifying required Docker images exist locally...")
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True
        )
        local_images = result.stdout.strip().split('\n')
        
        # Extract instance IDs from predictions
        import json
        required_instances = []
        with open(predictions_path, 'r') as f:
            if predictions_path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        required_instances.append(pred['instance_id'])
            else:
                predictions = json.load(f)
                required_instances = [pred['instance_id'] for pred in predictions]
        
        missing_images = []
        for inst_id in required_instances:
            # Convert instance_id to image name format
            # Format: pytest-dev__pytest-5221 → pytest-dev_1776_pytest-5221
            formatted_id = inst_id.replace('__', '_1776_').replace('-', '_')
            # Always check for standard local format (no namespace)
            image_pattern = f"sweb.eval.x86_64.{formatted_id}"
            
            if not any(image_pattern in img for img in local_images):
                missing_images.append(inst_id)
        
        if missing_images:
            print(f"\n⚠️  WARNING: {len(missing_images)} required images not found locally:")
            for inst_id in missing_images[:5]:
                formatted_id = inst_id.replace('__', '_1776_').replace('-', '_')
                expected = f"sweb.eval.x86_64.{formatted_id}:swebench-instance"
                print(f"   - {inst_id}")
                print(f"     Expected: {expected}")
            if len(missing_images) > 5:
                print(f"   ... and {len(missing_images) - 5} more")
            
            print(f"\n💡 These images need to be built")
            print(f"   Re-run the script and choose option 1 to build them")
        else:
            print(f"✅ All {len(required_instances)} required images found locally!")
            print(f"   Using: sweb.eval.x86_64.* (pure local images)")
    except Exception as e:
        print(f"⚠️  Could not verify images: {e}")

# -----------------------------------------------------------------------------
# Step 10: Run SWE-bench evaluation with Docker
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print(" Starting SWE-bench Evaluation")
print("="*60)
print("Number of instances to evaluate Running tests on built Docker images")
print("⏱️  This process may take 15-30 minutes depending on the number of instances")

# Prepare run_id
run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Set up evaluation command
eval_cmd = [
    str(python_path),
    "-m",
    "swebench.harness.run_evaluation",
    "--dataset_name",
    dataset_name,
    "--predictions_path",
    str(predictions_path) if not isinstance(predictions_path, str) or not predictions_path == "gold" else predictions_path,
    "--max_workers",
    "2",  # Reduced from 4 to prevent OOM errors
    "--run_id",
    run_id,
    "--cache_level",
    "instance",  # Use all local images including instance-specific ones
]

# No namespace - always use local images
print("📦 Using standard local images (no namespace)")
print("   Looking for: sweb.eval.x86_64.*")
print(f"🔧 Cache level: instance (will use all local images, no pulling)")

print(f"\n🆔 Run ID: {run_id}")
print(f"📊 Predictions: {predictions_path}")
print(f"📦 Dataset: {dataset_name}")
print(f" Command: {' '.join(eval_cmd)}")

# Run evaluation
run(eval_cmd, cwd=str(repo_dir), env=docker_env)

# Log Docker images after evaluation
print("\n📸 Logging SWE-bench Docker images (after evaluation)...")
images_after = log_docker_images("docker_images.log", mode="append")
new_eval_images = images_after - images_after_build

print("\n" + "="*60)
print("✅ SWE-bench Evaluation Complete!")
print("="*60)
# Show absolute paths where results actually are
results_file = repo_dir.resolve() / f"{model_name.replace('/', '_')}.{run_id}.json"
logs_dir = repo_dir.resolve() / "logs" / "run_evaluation" / run_id
predictions_file = repo_dir.resolve() / f"{model_name.replace('/', '_')}__SWE-bench_Lite__test.jsonl"
print(f"� Results file: {results_file}")
print(f"� Logs directory: {logs_dir}")
print(f"📄 Predictions file: {predictions_file}")
print(f"\n🔍 View results: cat {results_file}")
print(f"🔍 View logs: ls -la {logs_dir}")
print("\n🐳 Docker Images Summary:")
print(f"   • Images before: {images_before}")
print(f"   • Images after building: {images_after_build} (added {new_images})")
print(f"   • Images after evaluation: {images_after} (added {new_eval_images} during eval)")
print(f"   • Total new images: {images_after - images_before}")
print(f"   • Detailed log: docker_images.log (SWE-bench images only)")

# Final cleanup: Remove backup file if it exists
if mode == "2":  # Only for API mode where we patched the file
    backup_file = repo_dir / "swebench" / "inference" / "run_api.py.backup"
    if backup_file.exists():
        print("\n🧹 Cleaning up...")
        backup_file.unlink()
        print("✅ Backup file removed")

print("="*60)
