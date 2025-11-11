#!/usr/bin/env python3
"""
Rust Benchmarking Complete Setup and Run Script
===============================================
This script performs complete setup and runs the benchmark in one command:
- Installs Rust toolchain and Cargo
- Installs Python dependencies
- Installs system utilities
- Verifies installation
- Runs the benchmark
"""

import os
import sys
import subprocess
import platform
import tempfile
import shutil
from pathlib import Path

class BenchmarkSetup:
    def __init__(self):
        self.os_type = platform.system()
        self.home = str(Path.home())
        self.script_dir = Path(__file__).parent.absolute()
        
    def print_header(self, message):
        """Print formatted header"""
        print("\n" + "=" * 70)
        print(f"🚀 {message}")
        print("=" * 70 + "\n")
        
    def print_step(self, step_num, message):
        """Print step information"""
        print(f"\nℹ️  Step {step_num}: {message}...\n")
        
    def print_success(self, message):
        """Print success message"""
        print(f"✅ {message}")
        
    def print_warning(self, message):
        """Print warning message"""
        print(f"⚠️  {message}")
        
    def print_error(self, message):
        """Print error message"""
        print(f"❌ {message}")
        
    def command_exists(self, command):
        """Check if a command exists"""
        return shutil.which(command) is not None
    
    def run_command(self, cmd, shell=False, check=True, capture_output=False):
        """Run a shell command"""
        try:
            if capture_output:
                result = subprocess.run(
                    cmd if isinstance(cmd, list) else cmd,
                    shell=shell,
                    check=check,
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()
            else:
                subprocess.run(
                    cmd if isinstance(cmd, list) else cmd,
                    shell=shell,
                    check=check
                )
                return True
        except subprocess.CalledProcessError as e:
            if check:
                self.print_error(f"Command failed: {e}")
                raise
            return False
    
    def check_system_requirements(self):
        """Step 1: Check system requirements"""
        self.print_step(1, "Checking system requirements")
        
        if self.os_type == "Darwin":
            machine = "Mac"
        elif self.os_type == "Linux":
            machine = "Linux"
        else:
            machine = f"UNKNOWN: {self.os_type}"
            
        print(f"ℹ️  Detected OS: {machine}")
        return machine
    
    def install_rust(self):
        """Step 2: Install Rust toolchain"""
        self.print_step(2, "Installing Rust toolchain and Cargo")
        
        if self.command_exists("rustc") and self.command_exists("cargo"):
            rust_version = self.run_command("rustc --version", shell=True, capture_output=True)
            cargo_version = self.run_command("cargo --version", shell=True, capture_output=True)
            self.print_success(f"Rust is already installed: {rust_version}")
            self.print_success(f"Cargo is already installed: {cargo_version}")
            
            # Ask to update
            response = input("Do you want to update Rust? (y/n): ")
            if response.lower() == 'y':
                print("ℹ️  Updating Rust...")
                self.run_command("rustup update", shell=True)
                self.print_success("Rust updated successfully")
        else:
            print("ℹ️  Installing Rust and Cargo via rustup...")
            
            # Download and install rustup
            install_cmd = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
            self.run_command(install_cmd, shell=True)
            
            # Add cargo to PATH
            cargo_env = os.path.join(self.home, ".cargo", "env")
            if os.path.exists(cargo_env):
                # Source the environment (for current process)
                os.environ["PATH"] = f"{os.path.join(self.home, '.cargo', 'bin')}:{os.environ['PATH']}"
            
            self.print_success("Rust and Cargo installed successfully")
            
            # Verify installation
            rust_version = self.run_command("rustc --version", shell=True, capture_output=True)
            cargo_version = self.run_command("cargo --version", shell=True, capture_output=True)
            print(f"  {rust_version}")
            print(f"  {cargo_version}")
    
    def install_python_deps(self):
        """Step 3: Install Python dependencies"""
        self.print_step(3, "Installing Python dependencies")
        
        # Check Python version
        if not self.command_exists("python3"):
            self.print_error("Python 3 is not installed. Please install Python 3.7+ first.")
            sys.exit(1)
        
        python_version = self.run_command("python3 --version", shell=True, capture_output=True)
        self.print_success(f"Python is installed: {python_version}")
        
        # Upgrade pip
        print("ℹ️  Upgrading pip...")
        self.run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install the human_eval package
        print("ℹ️  Installing human_eval package...")
        self.run_command([sys.executable, "-m", "pip", "install", "-e", "."])
        
        # Install additional dependencies
        print("ℹ️  Installing additional Python dependencies...")
        requirements_file = self.script_dir / "requirements.txt"
        
        if requirements_file.exists():
            self.run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            self.print_success("Requirements installed from requirements.txt")
        else:
            # Install essential packages manually
            packages = ["numpy", "openai", "tqdm", "anthropic", "python-dotenv"]
            self.run_command([sys.executable, "-m", "pip", "install"] + packages)
            self.print_success("Essential Python packages installed")
    
    def install_system_utilities(self, machine):
        """Step 4: Install system utilities"""
        self.print_step(4, "Installing system utilities")
        
        if machine == "Mac":
            # macOS using Homebrew
            if not self.command_exists("brew"):
                self.print_warning("Homebrew not found. Installing Homebrew...")
                install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                self.run_command(install_cmd, shell=True)
                self.print_success("Homebrew installed successfully")
            else:
                self.print_success("Homebrew is already installed")
            
            # Install utilities
            print("ℹ️  Installing additional utilities via Homebrew...")
            self.run_command("brew install coreutils jq", shell=True, check=False)
            
        elif machine == "Linux":
            # Linux using apt
            print("ℹ️  Installing additional utilities via apt...")
            self.run_command("sudo apt-get update", shell=True, check=False)
            self.run_command("sudo apt-get install -y build-essential curl git jq", shell=True, check=False)
        
        self.print_success("System utilities installed")
    
    def verify_rust(self):
        """Step 5: Verify Rust installation"""
        self.print_step(5, "Verifying Rust installation")
        
        # Test cargo by creating a simple test project
        with tempfile.TemporaryDirectory() as test_dir:
            print("ℹ️  Creating test Rust project...")
            self.run_command(["cargo", "new", "test_project", "--bin"], cwd=test_dir, check=False)
            
            project_dir = os.path.join(test_dir, "test_project")
            print("ℹ️  Building test project...")
            
            try:
                self.run_command(["cargo", "build"], cwd=project_dir, capture_output=True)
                self.print_success("Rust toolchain is working correctly")
            except subprocess.CalledProcessError:
                self.print_error("Rust build test failed")
                sys.exit(1)
    
    def check_dataset(self):
        """Step 6: Check for Rust dataset"""
        self.print_step(6, "Checking for Rust dataset")
        
        dataset_path = self.script_dir / "data" / "humaneval-rust.jsonl.gz"
        
        if dataset_path.exists():
            self.print_success(f"Rust dataset already exists at {dataset_path}")
        else:
            self.print_warning(f"Rust dataset not found at {dataset_path}")
            print("ℹ️  Please ensure you have the Rust dataset file in the data/ directory")
            print("ℹ️  You can download it from the MultiPL-E repository")
    
    def setup_environment(self):
        """Step 7: Set up environment variables"""
        self.print_step(7, "Setting up environment variables")
        
        # Determine shell profile
        shell_profiles = [
            os.path.join(self.home, ".zshrc"),
            os.path.join(self.home, ".bashrc"),
            os.path.join(self.home, ".bash_profile")
        ]
        
        shell_profile = None
        for profile in shell_profiles:
            if os.path.exists(profile):
                shell_profile = profile
                break
        
        if shell_profile:
            # Check if cargo is already in PATH
            with open(shell_profile, 'r') as f:
                content = f.read()
            
            if 'cargo/env' not in content:
                with open(shell_profile, 'a') as f:
                    f.write('\n# Rust environment\n')
                    f.write('source "$HOME/.cargo/env"\n')
                self.print_success(f"Added Rust environment to {shell_profile}")
            else:
                self.print_success(f"Rust environment already configured in {shell_profile}")
    
    def create_env_example(self):
        """Step 8: Create .env.example file"""
        self.print_step(8, "Creating .env.example file")
        
        env_example_content = """# API Configuration
# Copy this file to .env and fill in your API keys

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI API Key (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_NAME=claude-3-7-sonnet-20250219
BASE_MODEL_NAME=claude-3-7-sonnet-20250219

# Benchmark Settings
NUM_SAMPLES=10
TEMPERATURE=0.8
MAX_TOKENS=2048
TIMEOUT=60
"""
        
        env_example_path = self.script_dir / ".env.example"
        with open(env_example_path, 'w') as f:
            f.write(env_example_content)
        
        self.print_success("Created .env.example file")
        
        env_path = self.script_dir / ".env"
        if not env_path.exists():
            self.print_warning("Please create a .env file with your API keys:")
            print("ℹ️    cp .env.example .env")
            print("ℹ️    # Then edit .env with your actual API keys")
    
    def run_benchmark(self):
        """Step 9: Run the benchmark"""
        self.print_step(9, "Running Rust benchmark")
        
        # Check if .env file exists
        env_path = self.script_dir / ".env"
        if not env_path.exists():
            self.print_error(".env file not found!")
            print("\nTo run the benchmark, you need to:")
            print("  1. Create .env file: cp .env.example .env")
            print("  2. Edit .env with your actual API keys")
            print("  3. Run this script again, OR run: python3 run_humaneval_api.py")
            return False
        
        # Check if benchmark script exists
        benchmark_script = self.script_dir / "run_humaneval_api.py"
        if not benchmark_script.exists():
            self.print_error("run_humaneval_api.py not found!")
            return False
        
        print("ℹ️  Starting benchmark execution...")
        print("ℹ️  This may take several minutes depending on the number of samples...\n")
        
        try:
            # Run the benchmark script
            self.run_command([sys.executable, str(benchmark_script)])
            self.print_success("Benchmark completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Benchmark failed: {e}")
            return False
    
    def print_summary(self, benchmark_ran):
        """Print final summary"""
        self.print_header("🎉 Setup Complete!")
        
        print("ℹ️  Summary of installed components:")
        print("  ✅ Rust toolchain (rustc, cargo)")
        print("  ✅ Python 3 and pip")
        print("  ✅ human_eval package")
        print("  ✅ Required Python dependencies")
        print("  ✅ System utilities")
        
        if benchmark_ran:
            print("\n✅ Benchmark execution completed!")
            print("\nℹ️  Results saved in result/ directory:")
            print("  - api_finetuned_rust.jsonl: Fine-tuned model completions")
            print("  - api_base_rust.jsonl: Base model completions")
            print("  - api_*_results.jsonl: Evaluation results")
            print("  - api_rust_benchmark_results.json: Summary metrics")
        else:
            print("\n⚠️  Benchmark was not run.")
            print("\nℹ️  To run the benchmark:")
            print("  1. Create .env file with your API keys:")
            print("     cp .env.example .env")
            print("     # Then edit .env with your actual API keys")
            print("")
            print("  2. Run the benchmark:")
            print("     python3 run_humaneval_api.py")
            print("     # OR run this script again:")
            print("     python3 setup_and_run.py")
        
        print("\nℹ️  Verification commands:")
        print("  cargo --version")
        print('  python3 -c "from human_eval.evaluation import evaluate_rust_correctness; print(\'✅ OK\')"')
        print("\n✅ Happy benchmarking! 🚀")
        print("=" * 70 + "\n")
    
    def run(self):
        """Execute the complete setup and benchmark"""
        try:
            self.print_header("Rust Benchmarking Environment Setup and Run")
            
            # Step 1: Check system requirements
            machine = self.check_system_requirements()
            
            # Step 2: Install Rust
            self.install_rust()
            
            # Step 3: Install Python dependencies
            self.install_python_deps()
            
            # Step 4: Install system utilities
            self.install_system_utilities(machine)
            
            # Step 5: Verify Rust installation
            self.verify_rust()
            
            # Step 6: Check for dataset
            self.check_dataset()
            
            # Step 7: Setup environment variables
            self.setup_environment()
            
            # Step 8: Create .env.example
            self.create_env_example()
            
            # Step 9: Run benchmark (if .env exists)
            benchmark_ran = self.run_benchmark()
            
            # Print summary
            self.print_summary(benchmark_ran)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    setup = BenchmarkSetup()
    setup.run()


if __name__ == "__main__":
    main()
