#!/usr/bin/env python3
"""
OpenHands SWE-bench Full Pipeline Runner
=========================================

This script runs the complete SWE-bench evaluation pipeline:
1. Inference: Runs OpenHands agent to generate patches
2. Evaluation: Tests the generated patches against the test suite

This combines the functionality of script.py (inference) and evaluate_results.py (evaluation)
into a single unified pipeline.

Usage:
    python3 run_full_pipeline.py

Author: Full pipeline script for OpenHands SWE-bench
Date: December 9, 2025
"""

import os
import sys
import subprocess
import json
import shutil
import urllib.request
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import yaml

# Import the classes from existing scripts
# This allows us to reuse the functionality
sys.path.insert(0, str(Path(__file__).parent))

# We'll define the Colors class here since both scripts use it
class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class FullPipelineRunner:
    """Runs both inference and evaluation in sequence."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the pipeline runner.
        
        Args:
            base_dir: Base directory for installation. Defaults to /mnt/storage/openhands_workspace
        """
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            storage_mount = Path("/mnt/storage")
            if storage_mount.exists():
                self.base_dir = storage_mount / "openhands_workspace"
            else:
                self.base_dir = Path.cwd()
        
        self.venv_dir = self.base_dir / "venv_openhands"
        self.openhands_dir = self.base_dir / "OpenHands"
        self.eval_output_base = self.openhands_dir / "evaluation" / "evaluation_outputs" / "outputs"
        
    def print_header(self, text: str, color: str = Colors.BLUE):
        """Print a formatted header."""
        print(f"\n{color}{'=' * 70}{Colors.NC}")
        print(f"{color}{text.center(70)}{Colors.NC}")
        print(f"{color}{'=' * 70}{Colors.NC}\n")
        
    def print_step(self, step: str, substep: str = ""):
        """Print a step indicator."""
        if substep:
            print(f"{Colors.CYAN}  → {substep}{Colors.NC}")
        else:
            print(f"\n{Colors.BOLD}{Colors.GREEN}[STEP] {step}{Colors.NC}")
            
    def print_success(self, message: str):
        """Print a success message."""
        print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
        
    def print_error(self, message: str):
        """Print an error message."""
        print(f"{Colors.RED}❌ {message}{Colors.NC}")
        
    def print_warning(self, message: str):
        """Print a warning message."""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")
    
    def run_command(self, cmd: list, cwd: Optional[Path] = None, 
                   check: bool = True, capture: bool = False,
                   env: Optional[Dict] = None, input_text: Optional[str] = None) -> Tuple[int, str, str]:
        """Run a shell command."""
        try:
            if capture:
                result = subprocess.run(
                    cmd, cwd=cwd, check=check, 
                    capture_output=True, text=True, env=env,
                    input=input_text
                )
                return result.returncode, result.stdout, result.stderr
            else:
                result = subprocess.run(cmd, cwd=cwd, check=check, env=env, 
                                      input=input_text, text=True if input_text else False)
                return result.returncode, "", ""
        except subprocess.CalledProcessError as e:
            if capture:
                return e.returncode, e.stdout or "", e.stderr or ""
            return e.returncode, "", ""
    
    def check_docker(self) -> bool:
        """Check if Docker is running."""
        self.print_step("Checking Docker...")
        code, _, _ = self.run_command(["docker", "info"], capture=True, check=False)
        
        if code == 0:
            self.print_success("Docker is running")
            return True
        else:
            self.print_error("Docker is not running!")
            self.print_warning("Please start Docker and try again")
            return False
    
    def get_user_input_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration from user input."""
        self.print_header("Model Configuration", Colors.YELLOW)
        
        print(f"{Colors.CYAN}Please provide your LLM configuration:{Colors.NC}\n")
        
        # Get API base URL
        print(f"{Colors.BOLD}1. API Base URL{Colors.NC}")
        print(f"{Colors.CYAN}   Examples:{Colors.NC}")
        print(f"   - OpenAI: https://api.openai.com/v1")
        print(f"   - vLLM: http://your-vllm-server:8001/v1")
        print(f"   - Ollama: http://localhost:11434")

        config_path = Path(Path(__file__).parent.parent / "model_config.yaml")
        config_file = Path(config_path)
        with open(config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
            api_base = yaml_data.get('api_base', '').strip()
        
            if not api_base:
                self.print_error("API base URL is required")
                return None
        
            # Get API key
            api_key = yaml_data.get('api_key', '').strip()
        
            if not api_key:
                api_key = "EMPTY"
        
            # Get model name

            model_name = yaml_data.get('model_name', '').strip()
            if not model_name:
                self.print_error("Model name is required")
                return None
        
        # Auto-detect if this is a vLLM/custom endpoint and add openai/ prefix
            is_custom_endpoint = (
                'openai.com' not in api_base.lower() and
                'anthropic.com' not in api_base.lower() and
                'openrouter.ai' not in api_base.lower() and
                not model_name.startswith('openai/') and
                not model_name.startswith('anthropic/') and
                not model_name.startswith('gpt-') and
                not model_name.startswith('claude-')
            )
        
            if is_custom_endpoint:
                print(f"\n{Colors.CYAN}💡 Detected custom/vLLM endpoint.{Colors.NC}")
                # add_prefix = input(f"{Colors.YELLOW}Add 'openai/' prefix for OpenAI-compatible API? (y/n) [y]: {Colors.NC}").strip().lower()
                # if add_prefix != 'n':
                model_name = f"openai/{model_name}"
                self.print_success(f"Model name updated to: {model_name}")
        
            # Get max iterations
            max_iterations = yaml_data.get('max_iterations', 100)
            
 
            
            # Get temperature
            temperature = yaml_data.get('temperature', 0.0)
            
            
            # Get top_p
            print(f"\n{Colors.BOLD}6. Top P (optional){Colors.NC}")
            print(f"{Colors.CYAN}   Nucleus sampling threshold (0.0 to 1.0)")
            print(f"{Colors.CYAN}   Lower = more focused, higher = more diverse (default: 0.95)")
            top_p = yaml_data.get('top_p', 0.95)
            
            # Summary
            print(f"\n{Colors.BOLD}Configuration Summary:{Colors.NC}")
            print(f"  API Base URL: {api_base}")
            print(f"  API Key: {'*' * 8}")
            print(f"  Model Name: {model_name}")
            print(f"  Max Iterations: {max_iterations}")
            print(f"  Temperature: {temperature}")
            print(f"  Top P: {top_p}")
            
            return {
                'model': model_name,
                'api_base': api_base,
                'api_key': api_key,
                'max_iterations': max_iterations,
                'temperature': temperature,
                'top_p': top_p
            }
    
    def get_config_from_file(self) -> Optional[Dict[str, Any]]:
        """Read configuration from existing config.toml file."""
        config_file = self.openhands_dir / "config.toml"
        
        if not config_file.exists():
            self.print_warning(f"Config file not found: {config_file}")
            return None
        
        self.print_step("Reading configuration from config.toml...")
        
        # Try using tomli first
        tomli_available = False
        try:
            import tomli
            tomli_available = True
        except ImportError:
            # Try to install tomli for reading TOML
            self.print_step("tomli not found, attempting to install...", "")
            pip_exe = self.venv_dir / "bin" / "pip"
            if pip_exe.exists():
                code, _, _ = self.run_command([str(pip_exe), "install", "tomli"], capture=True, check=False)
                if code == 0:
                    try:
                        import tomli
                        tomli_available = True
                        self.print_success("tomli installed successfully")
                    except ImportError:
                        pass
        
        # Try using tomli if available
        if tomli_available:
            try:
                with open(config_file, 'rb') as f:
                    config_data = tomli.load(f)
                
                llm_config = config_data.get('llm', {}).get('my_model', {})
                agent_config = config_data.get('agent', {})
                
                # Extract values
                model = llm_config.get('model', 'unknown')
                api_base = llm_config.get('base_url', '')
                max_iterations = agent_config.get('max_iterations', 100)
                
                self.print_success(f"Loaded config: model={model}")
                
                return {
                    'model': model,
                    'api_base': api_base,
                    'max_iterations': max_iterations,
                    'config_file': config_file
                }
                
            except Exception as e:
                self.print_warning(f"tomli parsing failed: {e}")
                self.print_step("Falling back to manual parsing...", "")
        
        # Fallback: parse manually
        self.print_warning("Using manual TOML parsing")
        return self._parse_toml_manually(config_file)
    
    def _parse_toml_manually(self, config_file: Path) -> Optional[Dict[str, Any]]:
        """Manual TOML parsing fallback."""
        try:
            config = {
                'model': 'unknown',
                'api_base': '',
                'api_key': 'EMPTY',
                'max_iterations': 100
            }
            current_section = None
            
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1].strip()
                    elif '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        
                        if current_section == 'llm.my_model':
                            if key == 'model':
                                config['model'] = value
                            elif key == 'base_url':
                                config['api_base'] = value
                            elif key == 'api_key':
                                config['api_key'] = value
                        elif current_section == 'agent':
                            if key == 'max_iterations':
                                try:
                                    config['max_iterations'] = int(value)
                                except ValueError:
                                    pass
            
            config['config_file'] = config_file
            self.print_success(f"Manually parsed config: model={config['model']}")
            return config
            
        except Exception as e:
            self.print_error(f"Error parsing TOML manually: {e}")
            return None
    
    def create_config_file(self, config: Dict[str, Any]) -> bool:
        """Create config.toml file with user-provided configuration."""
        config_file = self.openhands_dir / "config.toml"
        
        try:
            self.print_step(f"Creating config.toml at {config_file}...")
            
            config_content = f"""# OpenHands Configuration File
# Generated by run_full_pipeline.py

[llm.my_model]
model = "{config['model']}"
api_key = "{config['api_key']}"
base_url = "{config['api_base']}"
temperature = {config.get('temperature', 0.0)}
top_p = {config.get('top_p', 0.95)}

[agent]
name = "CodeActAgent"
max_iterations = {config['max_iterations']}

[core]
workspace_base = "./workspace"


"""
            
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            self.print_success(f"Config file created: {config_file}")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to create config file: {e}")
            return False
    
    def run_inference(self, eval_limit: int = 500, config: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Run the inference phase.
        
        Args:
            eval_limit: Number of instances to evaluate
            config: Configuration dictionary (optional, will be read from file if not provided)
            
        Returns:
            Path to the output directory, or None if failed
        """
        self.print_header("Phase 1: Running Inference", Colors.CYAN)
        
        # Get or use existing config
        if not config:
            config = self.get_config_from_file()
            if not config:
                self.print_error("Cannot read configuration. Please run script.py first to set up.")
                return None
        
        # Prepare environment
        env = os.environ.copy()
        
        # Add venv bin to PATH
        venv_bin = self.venv_dir / "bin"
        if venv_bin.exists():
            current_path = env.get("PATH", "")
            env["PATH"] = f"{venv_bin}:{current_path}"
            self.print_success(f"Added {venv_bin} to PATH")
        
        # Docker socket
        orbstack_socket = Path.home() / ".orbstack" / "run" / "docker.sock"
        if orbstack_socket.exists():
            env["DOCKER_HOST"] = f"unix://{orbstack_socket}"
        
        # Run inference script
        eval_script = self.openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "scripts" / "run_infer.sh"
        
        if not eval_script.exists():
            self.print_error(f"Inference script not found: {eval_script}")
            return None
        
        cmd = [
            "bash",
            str(eval_script),
            "llm.my_model",
            "HEAD",
            "CodeActAgent",
            str(eval_limit),
            str(config.get('max_iterations', 100)),
            "3",  # 3 parallel workers
            "princeton-nlp/SWE-bench_Verified",
            "test"
        ]
        
        self.print_step("Starting inference...")
        print(f"{Colors.CYAN}Command: {' '.join(cmd)}{Colors.NC}")
        print(f"{Colors.CYAN}Working directory: {self.openhands_dir}{Colors.NC}")
        print(f"{Colors.YELLOW}Estimated time: ~{eval_limit * 5 // 60} hours for {eval_limit} instances{Colors.NC}\n")
        
        # Run command and capture output to parse the output file path
        output_dir = self._run_inference_and_capture_output(cmd, env)
        
        if not output_dir:
            self.print_error("Inference failed or could not determine output directory!")
            return None
        
        self.print_success("Inference completed!")
        self.print_success(f"Output directory: {output_dir}")
        
        return output_dir
    
    def _run_inference_and_capture_output(self, cmd: list, env: Dict) -> Optional[Path]:
        """Run inference command and capture the output directory from logs.
        
        Args:
            cmd: Command to run
            env: Environment variables
            
        Returns:
            Path to the output directory extracted from logs
        """
        import re
        
        output_dir_path = None
        
        try:
            # Run the process and stream output in real-time
            process = subprocess.Popen(
                cmd,
                cwd=self.openhands_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Pattern to match the output file path
            # Example: "Writing evaluation output to evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/glm-latest_maxiter_1_N_v0.62.0-no-hint-run_1/output.jsonl"
            output_pattern = re.compile(r'Writing evaluation output to (.+/output\.jsonl)')
            
            # Stream and display output while capturing the path
            for line in process.stdout:
                # Print the line to terminal
                print(line, end='')
                
                # Check if this line contains the output path
                match = output_pattern.search(line)
                if match:
                    output_file_path = match.group(1)
                    # Convert to absolute path and get parent directory
                    if not output_file_path.startswith('/'):
                        output_file_path = self.openhands_dir / output_file_path
                    else:
                        output_file_path = Path(output_file_path)
                    
                    output_dir_path = output_file_path.parent
                    self.print_success(f"Detected output directory: {output_dir_path}")
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                self.print_error(f"Inference command failed with exit code {return_code}")
                return None
            
            # If we didn't capture the path from logs, fall back to finding latest
            if not output_dir_path:
                self.print_warning("Could not detect output path from logs, using fallback method...")
                output_dir_path = self._find_latest_output_dir()
            
            return output_dir_path
            
        except Exception as e:
            self.print_error(f"Error running inference: {e}")
            return None
    
    def _find_latest_output_dir(self) -> Optional[Path]:
        """Find the most recent output directory."""
        if not self.eval_output_base.exists():
            return None
        
        output_files = list(self.eval_output_base.rglob("output.jsonl"))
        if not output_files:
            return None
        
        # Get the most recently modified
        latest = max(output_files, key=lambda p: p.stat().st_mtime)
        return latest.parent
    
    def analyze_output_file(self, output_file: Path) -> Dict[str, Any]:
        """Analyze output.jsonl to see what needs evaluation."""
        stats = {
            "total": 0,
            "with_patches": 0,
            "with_test_results": 0,
            "needs_evaluation": 0
        }
        
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            stats["total"] += 1
                            
                            has_history = data.get("history") and len(data.get("history", [])) > 0
                            if has_history:
                                stats["with_patches"] += 1
                            
                            test_result = data.get("test_result")
                            has_test_results = False
                            
                            if test_result and test_result != "":
                                if isinstance(test_result, dict):
                                    # Check if actual evaluation or just inference output
                                    if "resolved" in test_result or "FAIL_TO_PASS" in test_result:
                                        has_test_results = True
                            
                            if has_test_results:
                                stats["with_test_results"] += 1
                            elif has_history:
                                stats["needs_evaluation"] += 1
                                
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.print_warning(f"Error analyzing output: {e}")
        
        return stats
    
    def run_evaluation(self, output_dir: Path) -> bool:
        """Run the evaluation phase.
        
        Args:
            output_dir: Directory containing output.jsonl from inference
            
        Returns:
            True if evaluation succeeded
        """
        self.print_header("Phase 2: Running Evaluation", Colors.GREEN)
        
        output_file = output_dir / "output.jsonl"
        if not output_file.exists():
            self.print_error(f"output.jsonl not found in {output_dir}")
            return False
        
        self.print_success(f"Using output file: {output_file}")
        
        # Check what needs evaluation
        stats = self.analyze_output_file(output_file)
        
        print(f"\n{Colors.CYAN}Inference Results:{Colors.NC}")
        print(f"  Total instances: {stats['total']}")
        print(f"  With patches: {stats['with_patches']}")
        print(f"  Already evaluated: {stats['with_test_results']}")
        print(f"  Need evaluation: {stats['needs_evaluation']}")
        
        if stats['needs_evaluation'] == 0:
            self.print_success("All instances already evaluated!")
            return True
        
        print(f"\n{Colors.YELLOW}Will evaluate {stats['needs_evaluation']} instances{Colors.NC}")
        print(f"{Colors.YELLOW}Estimated time: ~{stats['needs_evaluation'] * 3}-{stats['needs_evaluation'] * 5} minutes{Colors.NC}\n")
        
        # Prepare environment
        env = os.environ.copy()
        
        # Add venv bin to PATH for poetry
        venv_bin = self.venv_dir / "bin"
        if venv_bin.exists():
            current_path = env.get("PATH", "")
            env["PATH"] = f"{venv_bin}:{current_path}"
            self.print_success(f"Added {venv_bin} to PATH for poetry")
        
        # Docker socket
        orbstack_socket = Path.home() / ".orbstack" / "run" / "docker.sock"
        if orbstack_socket.exists():
            env["DOCKER_HOST"] = f"unix://{orbstack_socket}"
        
        # Find evaluation script
        eval_script = self.openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "scripts" / "eval_infer.sh"
        
        if not eval_script.exists():
            self.print_error(f"Evaluation script not found: {eval_script}")
            return False
        
        # Extract dataset info from directory path
        dataset_part = output_dir.parent.parent.name
        if "__" in dataset_part:
            dataset_name = dataset_part.replace("__", "/")
            if dataset_name.endswith("-test"):
                dataset_name = dataset_name[:-5]
                split = "test"
            else:
                split = "test"
        else:
            dataset_name = "princeton-nlp/SWE-bench_Verified"
            split = "test"
        
        cmd = [
            "bash",
            str(eval_script),
            str(output_file),
            "",  # Empty instance_id means eval all
            dataset_name,
            split
        ]
        
        self.print_step("Starting evaluation...")
        print(f"{Colors.CYAN}Command: {' '.join(cmd)}{Colors.NC}")
        print(f"{Colors.CYAN}Working directory: {self.openhands_dir}{Colors.NC}\n")
        
        code, _, _ = self.run_command(cmd, cwd=self.openhands_dir, check=False, env=env)
        
        if code != 0:
            self.print_error("Evaluation failed!")
            return False
        
        self.print_success("Evaluation completed!")
        
        # Show final results
        self.print_step("Analyzing final results...")
        final_stats = self._parse_final_results(output_file)
        
        print(f"\n{Colors.BOLD}Final Results:{Colors.NC}")
        print(f"  Total: {final_stats['total']}")
        print(f"  Resolved: {final_stats['resolved']}")
        print(f"  Failed: {final_stats['failed']}")
        
        if final_stats['total'] > 0:
            success_rate = final_stats['resolved'] / final_stats['total'] * 100
            print(f"\n{Colors.GREEN}{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.NC}")
        
        return True
    
    def _parse_final_results(self, output_file: Path) -> Dict[str, Any]:
        """Parse final evaluation results."""
        results = {
            "total": 0,
            "resolved": 0,
            "failed": 0
        }
        
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            test_result = data.get("test_result")
                            
                            if test_result and isinstance(test_result, dict):
                                if "resolved" in test_result or "FAIL_TO_PASS" in test_result:
                                    results["total"] += 1
                                    
                                    is_resolved = (
                                        test_result.get("resolved", False) or
                                        test_result.get("FAIL_TO_PASS", 0) > 0
                                    )
                                    
                                    if is_resolved:
                                        results["resolved"] += 1
                                    else:
                                        results["failed"] += 1
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.print_warning(f"Error parsing results: {e}")
        
        return results
    
    def run(self, eval_limit: int = 500):
        """Run the full pipeline.
        
        Args:
            eval_limit: Number of instances to evaluate
        """
        self.print_header("OpenHands SWE-bench Full Pipeline", Colors.MAGENTA)
        
        print(f"{Colors.CYAN}This script runs the complete pipeline:{Colors.NC}")
        print(f"  1. Inference - Generate patches with OpenHands agent")
        print(f"  2. Evaluation - Test patches against the test suite\n")
        
        # Check prerequisites
        if not self.check_docker():
            return False
        
        if not self.openhands_dir.exists():
            self.print_error(f"OpenHands directory not found: {self.openhands_dir}")
            self.print_warning("Please run script.py first to set up OpenHands")
            return False
        
        # Always ask user for configuration
        config = self.get_user_input_config()
        if not config:
            self.print_error("Configuration required to proceed")
            return False
        
        # Create config file with user input
        if not self.create_config_file(config):
            self.print_error("Failed to create configuration file")
            return False
        
        # Ask user for confirmation
        print(f"\n{Colors.BOLD}Pipeline Configuration:{Colors.NC}")
        print(f"  Instances to evaluate: {eval_limit}")
        print(f"  Model: {config.get('model')}")
        print(f"  API Base: {config.get('api_base')}")
        print(f"  Max Iterations: {config.get('max_iterations')}")
        print(f"  Temperature: {config.get('temperature', 0.0)}")
        print(f"  Top P: {config.get('top_p', 0.95)}")
        print(f"  Estimated total time: ~{eval_limit * 5 // 60} hours (inference) + ~{eval_limit * 3 // 60} hours (evaluation)")
        print(f"\n{Colors.CYAN}How it works:{Colors.NC}")
        print(f"  1. Runs inference and monitors output for the result file path")
        print(f"  2. Automatically uses the detected path for evaluation")
        print(f"  3. No manual file selection needed!")
        
        # response = input(f"\n{Colors.YELLOW}Proceed with full pipeline? (y/n) [y]: {Colors.NC}").strip().lower()
        # if response == 'n':
        #     self.print_warning("Pipeline cancelled")
        #     return False
        
        # Phase 1: Run inference
        output_dir = self.run_inference(eval_limit, config)
        if not output_dir:
            self.print_error("Inference phase failed!")
            return False
        
        # Brief pause between phases
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.CYAN}Inference complete. Starting evaluation in 3 seconds...{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
        import time
        time.sleep(3)
        
        # Phase 2: Run evaluation
        success = self.run_evaluation(output_dir)
        
        if success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.NC}")
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 Full pipeline completed successfully!{Colors.NC}")
            print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.NC}\n")
            print(f"{Colors.CYAN}Results saved to:{Colors.NC}")
            print(f"  {output_dir / 'output.jsonl'}\n")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Evaluation phase failed{Colors.NC}\n")
        
        return success


def main():
    """Main entry point."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}OpenHands SWE-bench Full Pipeline Runner{Colors.NC}\n")
    
    # Parse command-line arguments
    eval_limit = 500  # Default
    base_dir = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python3 run_full_pipeline.py [eval_limit] [base_dir]")
            print("\nArguments:")
            print("  eval_limit: Number of instances to evaluate (default: 500)")
            print("  base_dir: Base directory for OpenHands (default: /mnt/storage/openhands_workspace)")
            print("\nExample:")
            print("  python3 run_full_pipeline.py 10")
            print("  python3 run_full_pipeline.py 50 /path/to/workspace")
            return 0
        else:
            try:
                eval_limit = int(sys.argv[1])
            except ValueError:
                print(f"{Colors.RED}Invalid eval_limit: {sys.argv[1]}{Colors.NC}")
                return 1
    
    if len(sys.argv) > 2:
        base_dir = sys.argv[2]
    
    # Create and run pipeline
    runner = FullPipelineRunner(base_dir)
    
    try:
        success = runner.run(eval_limit)
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Pipeline interrupted by user{Colors.NC}")
        return 130
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.NC}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
