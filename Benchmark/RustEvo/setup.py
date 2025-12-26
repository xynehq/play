#!/usr/bin/env python3
"""
RustEvo Benchmark Setup and Execution Script
This script automates the setup and execution of RustEvo benchmarks
Clones the RustEvo repository and runs RQ1 and RQ3 evaluations

Repository: https://github.com/SYSUSELab/RustEvo
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
import json
import yaml
# Configuration
RUSTEVO_REPO = "https://github.com/SYSUSELab/RustEvo.git"
RUSTEVO_DIR = "RustEvo"

def print_info(message):
    """Print info message"""
    print(f"[INFO] {message}")

def print_success(message):
    """Print success message"""
    print(f"[SUCCESS] {message}")

def print_warning(message):
    """Print warning message"""
    print(f"[WARNING] {message}")

def print_error(message):
    """Print error message"""
    print(f"[ERROR] {message}")

def print_section(title):
    """Print section header"""
    print()
    print("========================================")
    print(title)
    print("========================================")

def command_exists(command):
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None

def run_command(cmd, shell=False, check=True, capture_output=False):
    """Run a shell command"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=check, 
                                   capture_output=capture_output, text=True)
        else:
            result = subprocess.run(cmd, check=check, 
                                   capture_output=capture_output, text=True)
        return result
    except subprocess.CalledProcessError as e:
        return None

def clone_rustevo_repo():
    """Clone RustEvo repository"""
    print_section("Cloning RustEvo Repository")
    
    if os.path.exists(RUSTEVO_DIR):
        print_success("Using existing RustEvo directory")
        return True
    
    print_info(f"Cloning from {RUSTEVO_REPO}...")
    
    result = run_command(["git", "clone", RUSTEVO_REPO, RUSTEVO_DIR], check=False)
    
    if result and result.returncode == 0:
        print_success(f"Successfully cloned RustEvo to {RUSTEVO_DIR}")
        return True
    else:
        print_error("Failed to clone RustEvo repository")
        print_info("Please clone manually: git clone https://github.com/SYSUSELab/RustEvo.git")
        return False

def install_rust():
    """Install Rust toolchain"""
    print_section("Installing Rust")
    
    if command_exists("rustc"):
        result = run_command("rustc --version", shell=True, capture_output=True)
        if result:
            print_info(f"Rust is already installed: {result.stdout.strip()}")
            return True
    
    print_info("Rust not found. Installing Rust...")
    
    # Download and run rustup installer
    install_cmd = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    result = run_command(install_cmd, shell=True, check=False)
    
    if result is None or result.returncode != 0:
        print_error("Rust installation failed. Please install manually from https://rustup.rs/")
        return False
    
    # Source cargo environment
    cargo_env = os.path.expanduser("~/.cargo/env")
    if os.path.exists(cargo_env):
        # Update PATH for current session
        cargo_bin = os.path.expanduser("~/.cargo/bin")
        if cargo_bin not in os.environ['PATH']:
            os.environ['PATH'] = f"{cargo_bin}:{os.environ['PATH']}"
    
    if command_exists("rustc"):
        result = run_command("rustc --version", shell=True, capture_output=True)
        if result:
            print_success(f"Rust installed successfully: {result.stdout.strip()}")
            return True
    
    print_error("Rust installation failed. Please install manually from https://rustup.rs/")
    return False

def install_python_deps():
    """Install Python dependencies"""
    print_section("Installing Python Dependencies")
    
    if not command_exists("python3"):
        print_error("Python 3 is not installed. Please install Python 3.8 or higher.")
        return False
    
    result = run_command("python3 --version", shell=True, capture_output=True)
    if result:
        print_info(f"Python version: {result.stdout.strip()}")
    
    if not command_exists("pip3"):
        print_error("pip3 is not installed. Please install pip3.")
        return False
    
    print_info("Installing required Python packages...")
    
    # Upgrade pip
    run_command("pip3 install --upgrade pip", shell=True, check=False)
    
    # Install required packages
    result = run_command("pip3 install openai tqdm", shell=True, check=False)
    
    if result is None or result.returncode != 0:
        print_error("Failed to install Python dependencies")
        return False
    
    print_success("Python dependencies installed successfully")
    return True

def get_api_credentials():
    """Get API credentials from Config file"""
    print_section("API Configuration")
    
    # Check if already in environment
    config_path = Path(Path(__file__).parent.parent / "model_config.yaml")
    config_file = Path(config_path)
    with open(config_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
        api_key = yaml_data.get('api_key', '')
        base_url = yaml_data.get('api_base', '')
    return api_key, base_url

def verify_datasets():
    """Verify that required dataset files exist"""
    print_section("Verifying Dataset Files")
    
    dataset_files = [
        f"{RUSTEVO_DIR}/Dataset/RustEvo^2.json",
        f"{RUSTEVO_DIR}/Dataset/APIDocs.json"
    ]
    
    all_exist = True
    for file_path in dataset_files:
        if not os.path.exists(file_path):
            print_error(f"Dataset file not found: {file_path}")
            all_exist = False
        else:
            print_success(f"Found: {file_path}")
    
    return all_exist

def create_results_dir():
    """Create results directory if it doesn't exist"""
    print_section("Creating Results Directory")
    
    results_dir = f"{RUSTEVO_DIR}/Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print_success("Created Results directory")
    else:
        print_info("Results directory already exists")
    return True

def run_rq1(model="kat-dev-hs-72b", max_workers=8):
    """Run RQ1 evaluation"""
    print_section("Running RQ1 Evaluation (Full Documentation)")
    
    print_info(f"Model: {model}")
    print_info(f"Max Workers: {max_workers}")
    print_info("This may take a while depending on the number of tasks...")
    
    api_key = os.environ.get('API_KEY', '')
    base_url = os.environ.get('BASE_URL', '')
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to RustEvo directory
        # os.chdir(RUSTEVO_DIR)
        
        cmd = [
            "python3", "script /eval_models_rq1.py",
            "--file_a", "RustEvo/Dataset/RustEvo^2.json",
            "--file_b", "RustEvo/Dataset/APIDocs.json",
            "--output", "RustEvo/Results/rq1_results.json",
            "--models", model,
            "--max_workers", str(max_workers),
            "--api_key", api_key,
            "--base_url", base_url
        ]
        
        result = run_command(cmd, check=False)
        
        if result and result.returncode == 0:
            print_success("RQ1 evaluation completed successfully!")
            print_info(f"Results saved to: RustEvo/Results/rq1_results.json")
            print_info(f"Metrics saved to: RustEvo/Results/rq1_results_metrics.json")
            return True
        else:
            print_error("RQ1 evaluation failed")
            return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def run_rq3(model="kat-dev-hs-72b", max_workers=8):
    """Run RQ3 evaluation"""
    print_section("Running RQ3 Evaluation (Minimal Documentation)")
    
    print_info(f"Model: {model}")
    print_info(f"Max Workers: {max_workers}")
    print_info("This may take a while depending on the number of tasks...")
    
    api_key = os.environ.get('API_KEY', '')
    base_url = os.environ.get('BASE_URL', '')
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to RustEvo directory
        # os.chdir(RUSTEVO_DIR)
        
        cmd = [
            "python3", "script /eval_models_rq3.py",
            "--file_a", "RustEvo/Dataset/RustEvo^2.json",
            "--file_b", "RustEvo/Dataset/APIDocs.json",
            "--output", "RustEvo/Results/rq3_results.json",
            "--models", model,
            "--max_workers", str(max_workers),
            "--api_key", api_key,
            "--base_url", base_url
        ]
        
        result = run_command(cmd, check=False)
        
        if result and result.returncode == 0:
            print_success("RQ3 evaluation completed successfully!")
            print_info(f"Results saved to: RustEvo/Results/rq3_results.json")
            print_info(f"Metrics saved to: RustEvo/Results/rq3_results_metrics.json")
            return True
        else:
            print_error("RQ3 evaluation failed")
            return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def display_results():
    """Display results summary"""
    print_section("Results Summary")

    rq1_metrics = "RustEvo/Results/rq1_results_metrics.json"
    rq3_metrics = "RustEvo/Results/rq3_results_metrics.json"
    
    if os.path.exists(rq1_metrics):
        print_info(f"RQ1 Metrics available at: {rq1_metrics}")
    
    if os.path.exists(rq3_metrics):
        print_info(f"RQ3 Metrics available at: {rq3_metrics}")
    
    print()
    print_info("To view detailed metrics, use:")
    print(f"  cat {rq1_metrics} | python3 -m json.tool")
    print(f"  cat {rq3_metrics} | python3 -m json.tool")

def show_menu():
    """Display interactive menu"""
    print()
    print("╔════════════════════════════════════════╗")
    print("║    RustEvo Benchmark Setup & Run      ║")
    print("╚════════════════════════════════════════╝")
    print()
    print("1. Full Setup (Install dependencies + Run both RQ1 & RQ3)")
    print("2. Install Dependencies Only")
    print("3. Run RQ1 Only (Full Documentation)")
    print("4. Run RQ3 Only (Minimal Documentation)")
    print("5. Run Both RQ1 & RQ3")
    print("6. Exit")
    print()

def interactive_mode():
    """Run in interactive menu mode"""
    # Clone repository first
    if not clone_rustevo_repo():
        print_error("Failed to clone RustEvo repository. Exiting...")
        sys.exit(1)
    
    # Get API credentials
    get_api_credentials()    
    config_path = Path(Path(__file__).parent.parent / "model_config.yaml")
    config_file = Path(config_path)    
    choice=1
    model=""
    workers_rq1=8
    workers_rq3=8
    with open(config_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
        choice = yaml_data.get('rust_evo_setup', 1)
        model = yaml_data.get('model_name')
        workers_rq1 = yaml_data.get('workers', 8)
        workers_rq3 = yaml_data.get('workers', 8)
    if choice == 1:
        install_rust()
        install_python_deps()
        verify_datasets()
        create_results_dir()

        run_rq1(model, workers_rq1)
        run_rq3(model, workers_rq3)
        display_results()
        
    elif choice == 2:
        install_rust()
        install_python_deps()
        verify_datasets()
        create_results_dir()
        print_success("Dependencies installed successfully!")
        
    elif choice == 3:
        verify_datasets()
        create_results_dir()
        
        run_rq1(model, workers_rq1)
        display_results()
        
    elif choice == 4:
        verify_datasets()
        create_results_dir()
        
        run_rq3(model, workers_rq3)
        display_results()
        
    elif choice == 5:
        verify_datasets()
        create_results_dir()
        
        run_rq1(model, workers_rq1)
        run_rq3(model, workers_rq3)
        display_results()
        
    elif choice == 6:
        print_info("Exiting...")
        sys.exit(0)
        
    else:
        print_error("Invalid option. Please select 1-6.")
    
    print()
    os.system('clear' if os.name == 'posix' else 'cls')

def main():
    """Main entry point"""
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Print banner
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║                                                    ║")
    print("║         RustEvo Benchmark Automation Script        ║")
    print("║                                                    ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RustEvo Benchmark Setup and Execution Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                                  # Full setup with default model
  %(prog)s --all kat-dev-hs-72b 8                # Full setup with custom settings
  %(prog)s --rq1 kat-dev-hs-72b 4                # Run RQ1 only
  %(prog)s --install                              # Install dependencies only
        """
    )
    
    parser.add_argument('--install', action='store_true',
                       help='Install dependencies only')
    parser.add_argument('--rq1', nargs='*',
                       help='Run RQ1 evaluation only [MODEL] [MAX_WORKERS]')
    parser.add_argument('--rq3', nargs='*',
                       help='Run RQ3 evaluation only [MODEL] [MAX_WORKERS]')
    parser.add_argument('--all', nargs='*',
                       help='Full setup and run both evaluations [MODEL] [MAX_WORKERS]')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.install:
        if not clone_rustevo_repo():
            print_error("Failed to clone RustEvo repository")
            sys.exit(1)
        install_rust()
        install_python_deps()
        get_api_credentials()
        verify_datasets()
        create_results_dir()
        print_success("Setup completed successfully!")
        sys.exit(0)
    
    elif args.rq1 is not None:
        if not clone_rustevo_repo():
            print_error("Failed to clone RustEvo repository")
            sys.exit(1)
        # Prompt for model if not provided
        if not args.rq1 or len(args.rq1) == 0:
            model = input("Enter model name: ").strip()
            workers_input = input("Enter max workers (default: 8): ").strip()
            workers = int(workers_input) if workers_input else 8
        else:
            model = args.rq1[0]
            workers = int(args.rq1[1]) if len(args.rq1) > 1 else 8
        get_api_credentials()
        verify_datasets()
        create_results_dir()
        run_rq1(model, workers)
        display_results()
        sys.exit(0)
    
    elif args.rq3 is not None:
        if not clone_rustevo_repo():
            print_error("Failed to clone RustEvo repository")
            sys.exit(1)
        # Prompt for model if not provided
        if not args.rq3 or len(args.rq3) == 0:
            model = input("Enter model name: ").strip()
            workers_input = input("Enter max workers (default: 8): ").strip()
            workers = int(workers_input) if workers_input else 8
        else:
            model = args.rq3[0]
            workers = int(args.rq3[1]) if len(args.rq3) > 1 else 8
        get_api_credentials()
        verify_datasets()
        create_results_dir()
        run_rq3(model, workers)
        display_results()
        sys.exit(0)
    
    elif args.all is not None:
        if not clone_rustevo_repo():
            print_error("Failed to clone RustEvo repository")
            sys.exit(1)
        # Prompt for model if not provided
        if not args.all or len(args.all) == 0:
            model = input("Enter model name: ").strip()
            workers_input = input("Enter max workers (default: 8): ").strip()
            workers = int(workers_input) if workers_input else 8
        else:
            model = args.all[0]
            workers = int(args.all[1]) if len(args.all) > 1 else 8
        install_rust()
        install_python_deps()
        get_api_credentials()
        verify_datasets()
        create_results_dir()
        run_rq1(model, workers)
        run_rq3(model, workers)
        display_results()
        sys.exit(0)
    
    # No arguments provided, run interactive mode
    interactive_mode()

if __name__ == "__main__":
    main()
