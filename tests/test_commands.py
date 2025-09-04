"""
Unit tests for all SFT-Play Makefile commands.

This module tests the core functionality of each command available in the Makefile,
ensuring they work correctly and handle edge cases appropriately.
"""

import pytest
import subprocess
import os
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestMakefileCommands:
    """Test suite for Makefile commands."""
    
    def test_help_command(self):
        """Test that help command displays available commands."""
        result = subprocess.run(
            ["make", "help"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        assert "SFT-Play Makefile Commands:" in result.stdout
        assert "train" in result.stdout
        assert "eval" in result.stdout
        assert "infer" in result.stdout
    
    def test_setup_dirs_command(self):
        """Test that setup-dirs creates necessary directories."""
        # Clean up first
        subprocess.run(["make", "clean"], cwd=".", capture_output=True)
        
        result = subprocess.run(
            ["make", "setup-dirs"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        
        # Check that directories were created
        expected_dirs = [
            "data/raw", "data/processed", "data/processed_with_style", 
            "data/rendered", "outputs", "adapters"
        ]
        for dir_path in expected_dirs:
            assert Path(dir_path).exists(), f"Directory {dir_path} was not created"
    
    def test_check_command_with_missing_data(self):
        """Test check command when training data is missing."""
        # Clean up first
        subprocess.run(["make", "clean"], cwd=".", capture_output=True)
        
        result = subprocess.run(
            ["make", "check"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        # Should fail when training data is missing
        assert result.returncode != 0
        assert "Training data not found" in result.stdout
    
    def test_check_command_with_data(self, test_data_files):
        """Test check command when data is present."""
        # Create minimal data structure
        os.makedirs("data/processed", exist_ok=True)
        
        # Create minimal training data
        with open("data/processed/train.jsonl", "w") as f:
            f.write('{"messages": [{"role": "user", "content": "test"}]}\n')
        
        result = subprocess.run(
            ["make", "check"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        assert "Training data found" in result.stdout
    
    def test_clean_command(self):
        """Test that clean command removes generated files."""
        # Create some files to clean
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        test_file = Path("data/processed/test.jsonl")
        test_file.write_text("test data")
        
        result = subprocess.run(
            ["make", "clean"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        assert not test_file.exists()
    
    def test_train_command_dry_run(self):
        """Test train command structure (dry run)."""
        # Test that the command would be called correctly
        result = subprocess.run(
            ["make", "-n", "train"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        assert "scripts/train.py" in result.stdout
        assert "--config" in result.stdout
    
    def test_train_bnb_command_dry_run(self):
        """Test train-bnb command structure (dry run)."""
        result = subprocess.run(
            ["make", "-n", "train-bnb"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        assert "scripts/train.py" in result.stdout
        assert "configs/run_bnb.yaml" in result.stdout
    
    def test_train_unsloth_command_dry_run(self):
        """Test train-unsloth command structure (dry run)."""
        result = subprocess.run(
            ["make", "-n", "train-unsloth"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        assert "scripts/train.py" in result.stdout
        assert "configs/run_unsloth.yaml" in result.stdout
        assert "XFORMERS_DISABLED=1" in result.stdout
    
    def test_tensorboard_commands_dry_run(self):
        """Test TensorBoard-related commands (dry run)."""
        commands = ["tensorboard", "tb-stop", "tb-clean", "tb-open"]
        
        for cmd in commands:
            result = subprocess.run(
                ["make", "-n", cmd], 
                cwd=".", 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0
    
    def test_eval_commands_dry_run(self):
        """Test evaluation commands (dry run)."""
        eval_commands = ["eval", "eval-test", "eval-val", "eval-quick", "eval-full"]
        
        for cmd in eval_commands:
            result = subprocess.run(
                ["make", "-n", cmd], 
                cwd=".", 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0
            assert "scripts/eval.py" in result.stdout
    
    def test_infer_commands_dry_run(self):
        """Test inference commands (dry run)."""
        infer_commands = ["infer", "infer-batch", "infer-interactive"]
        
        for cmd in infer_commands:
            result = subprocess.run(
                ["make", "-n", cmd], 
                cwd=".", 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0
            assert "scripts/infer.py" in result.stdout
    
    def test_merge_commands_dry_run(self):
        """Test model merging commands (dry run)."""
        merge_commands = ["merge", "merge-bf16", "merge-test"]
        
        for cmd in merge_commands:
            result = subprocess.run(
                ["make", "-n", cmd], 
                cwd=".", 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0
            # merge-test has different structure
            if cmd == "merge-test":
                assert "python -c" in result.stdout
            else:
                assert "scripts/merge_lora.py" in result.stdout
    
    def test_data_pipeline_commands_dry_run(self):
        """Test data processing pipeline commands (dry run)."""
        data_commands = ["process", "style", "render"]
        
        for cmd in data_commands:
            result = subprocess.run(
                ["make", "-n", cmd], 
                cwd=".", 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0
            if cmd == "process":
                assert "scripts/process_data.py" in result.stdout
            elif cmd == "style":
                assert "scripts/style_prompt.py" in result.stdout
            elif cmd == "render":
                assert "scripts/render_template.py" in result.stdout
    
    def test_dapt_commands_dry_run(self):
        """Test DAPT-related commands (dry run)."""
        dapt_commands = ["dapt-docx", "dapt-train"]
        
        for cmd in dapt_commands:
            result = subprocess.run(
                ["make", "-n", cmd], 
                cwd=".", 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0
            if cmd == "dapt-docx":
                assert "scripts/ingest_docx.py" in result.stdout
            elif cmd == "dapt-train":
                assert "scripts/train.py" in result.stdout
                assert "configs/run_dapt.yaml" in result.stdout
    
    def test_full_pipeline_command_dry_run(self):
        """Test full-pipeline command (dry run)."""
        result = subprocess.run(
            ["make", "-n", "full-pipeline"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        # Should include the actual pipeline steps
        expected_content = [
            "Creating necessary directories",
            "scripts/process_data.py",
            "scripts/style_prompt.py", 
            "scripts/render_template.py",
            "Full data processing pipeline completed"
        ]
        for content in expected_content:
            assert content in result.stdout


class TestCommandValidation:
    """Test command validation and error handling."""
    
    def test_config_file_validation(self):
        """Test that commands validate configuration files properly."""
        # Test with non-existent config
        result = subprocess.run(
            ["make", "check", "CONFIG=nonexistent.yaml"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode != 0
        assert "Config file not found" in result.stdout
    
    def test_print_python_command(self):
        """Test print-python command shows Python information."""
        result = subprocess.run(
            ["make", "print-python"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        assert "which python:" in result.stdout
        assert "sys.executable:" in result.stdout


class TestDataValidation:
    """Test data validation and processing."""
    
    def test_data_file_formats(self):
        """Test that data files are in correct format."""
        # Create test data with correct format
        os.makedirs("data/processed", exist_ok=True)
        
        valid_data = [
            {"messages": [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]}
        ]
        
        with open("data/processed/train.jsonl", "w") as f:
            for item in valid_data:
                f.write(json.dumps(item) + '\n')
        
        # Test that check command passes
        result = subprocess.run(
            ["make", "check"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
    
    def test_config_file_formats(self):
        """Test that configuration files are valid YAML."""
        config_files = [
            "configs/config_base.yaml",
            "configs/run_bnb.yaml",
            "configs/run_dapt.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    try:
                        yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        pytest.fail(f"Invalid YAML in {config_file}: {e}")


class TestIntegration:
    """Integration tests for command workflows."""
    
    def test_setup_to_check_workflow(self):
        """Test the workflow from setup to check."""
        # Clean first
        subprocess.run(["make", "clean"], cwd=".", capture_output=True)
        
        # Setup directories
        result = subprocess.run(
            ["make", "setup-dirs"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        
        # Create minimal data
        with open("data/processed/train.jsonl", "w") as f:
            f.write('{"messages": [{"role": "user", "content": "test"}]}\n')
        
        # Check should now pass
        result = subprocess.run(
            ["make", "check"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
    
    def test_clean_workflow(self):
        """Test that clean properly resets the environment."""
        # Create some test files
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        test_files = [
            "data/processed/train.jsonl",
            "outputs/test.txt"
        ]
        
        for file_path in test_files:
            Path(file_path).write_text("test content")
        
        # Clean
        result = subprocess.run(
            ["make", "clean"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        assert result.returncode == 0
        
        # Files should be gone
        for file_path in test_files:
            assert not Path(file_path).exists()
