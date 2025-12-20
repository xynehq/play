#!/usr/bin/env python3
"""
RustEvo VM Execution Wrapper
This script provides a convenient interface to run RustEvo benchmarks in the VM environment.
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path

def load_config():
    """Load VM configuration for RustEvo."""
    config_path = Path("Benchmark/RustEvo/vm_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def check_rust_evo_installation():
    """Check if RustEvo is properly installed."""
    rust_evo_dir = Path("Benchmark/RustEvo")
    
    if not rust_evo_dir.exists():
        print("❌ RustEvo not found. Please run: ./scripts/setup_rust_evo.sh")
        return False
    
    required_dirs = ["Dataset", "Evaluate", "Scripts"]
    for dir_name in required_dirs:
        if not (rust_evo_dir / dir_name).exists():
            print(f"❌ Missing required directory: {dir_name}")
            return False
    
    print("✅ RustEvo installation verified")
    return True

def run_evaluation_script(script_name, args):
    """Run a specific RustEvo evaluation script."""
    config = load_config()
    rust_evo_dir = Path("Benchmark/RustEvo")
    
    script_path = rust_evo_dir / "Evaluate" / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return False
    
    # Set environment variables for VM configuration
    env = os.environ.copy()
    env.update({
        'RUSTEVO_DATASET_PATH': str(rust_evo_dir / "Dataset"),
        'RUSTEVO_RESULTS_PATH': str(rust_evo_dir / "Results"),
        'RUSTEVO_SCRIPTS_PATH': str(rust_evo_dir / "Scripts"),
        'RUSTEVO_MAX_MEMORY_GB': str(config.get('max_memory_gb', 8)),
        'RUSTEVO_MAX_CPU_CORES': str(config.get('max_cpu_cores', 4)),
        'RUSTEVO_TIMEOUT_SECONDS': str(config.get('timeout_seconds', 300)),
    })
    
    try:
        cmd = [sys.executable, str(script_path)] + args
        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=rust_evo_dir, env=env, check=True)
        print("✅ Script completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Script failed with exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Python interpreter not found: {sys.executable}")
        return False

def run_analysis_script(script_name, args):
    """Run a specific RustEvo analysis script."""
    config = load_config()
    rust_evo_dir = Path("Benchmark/RustEvo")
    
    script_path = rust_evo_dir / "Scripts" / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return False
    
    # Set environment variables for VM configuration
    env = os.environ.copy()
    env.update({
        'RUSTEVO_DATASET_PATH': str(rust_evo_dir / "Dataset"),
        'RUSTEVO_RESULTS_PATH': str(rust_evo_dir / "Results"),
        'RUSTEVO_SCRIPTS_PATH': str(rust_evo_dir / "Scripts"),
    })
    
    try:
        cmd = [sys.executable, str(script_path)] + args
        print(f"🔍 Running analysis: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=rust_evo_dir, env=env, check=True)
        print("✅ Analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with exit code: {e.returncode}")
        return False

def list_available_scripts():
    """List all available RustEvo scripts."""
    rust_evo_dir = Path("Benchmark/RustEvo")
    
    print("📋 Available Evaluation Scripts:")
    eval_dir = rust_evo_dir / "Evaluate"
    if eval_dir.exists():
        for script in sorted(eval_dir.glob("*.py")):
            if script.name != "__init__.py":
                print(f"  • {script.name}")
    
    print("\n🔍 Available Analysis Scripts:")
    scripts_dir = rust_evo_dir / "Scripts"
    if scripts_dir.exists():
        for script in sorted(scripts_dir.glob("*.py")):
            if script.name != "__init__.py":
                print(f"  • {script.name}")

def main():
    parser = argparse.ArgumentParser(
        description="RustEvo VM Execution Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run setup first
  ./scripts/setup_rust_evo.sh
  
  # List available scripts
  python scripts/run_rust_evo.py --list
  
  # Run evaluation
  python scripts/run_rust_evo.py --eval eval_models.py
  
  # Run analysis
  python scripts/run_rust_evo.py --analysis generate_code.py
        """
    )
    
    parser.add_argument("--list", action="store_true", 
                       help="List all available RustEvo scripts")
    parser.add_argument("--eval", metavar="SCRIPT", 
                       help="Run evaluation script from Evaluate/ directory")
    parser.add_argument("--analysis", metavar="SCRIPT", 
                       help="Run analysis script from Scripts/ directory")
    parser.add_argument("--args", nargs=argparse.REMAINDER, 
                       help="Arguments to pass to the script")
    parser.add_argument("--check", action="store_true", 
                       help="Check RustEvo installation")
    
    args = parser.parse_args()
    
    if args.check:
        check_rust_evo_installation()
        return
    
    if args.list:
        list_available_scripts()
        return
    
    if not check_rust_evo_installation():
        sys.exit(1)
    
    script_args = args.args or []
    
    if args.eval:
        success = run_evaluation_script(args.eval, script_args)
    elif args.analysis:
        success = run_analysis_script(args.analysis, script_args)
    else:
        parser.print_help()
        print("\n💡 Use --list to see available scripts")
        return
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
