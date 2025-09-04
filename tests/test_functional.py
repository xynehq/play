"""
Functional tests that actually validate command behavior and catch real issues.

These tests go beyond just checking return codes and actually verify that:
1. Commands produce expected outputs
2. Data processing works correctly
3. File formats are valid
4. Pipelines produce usable results
"""

import pytest
import subprocess
import os
import json
import yaml
from pathlib import Path
import tempfile
import shutil


class TestDataProcessingFunctional:
    """Test actual data processing functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clean up before each test
        subprocess.run(["make", "clean"], cwd=".", capture_output=True)
        
    def test_process_data_with_instruction_format(self):
        """Test that process_data correctly handles instruction format."""
        # Create test raw data in instruction format
        test_data = [
            {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
            {"instruction": "Explain ML", "input": "briefly", "output": "ML is machine learning."}
        ]
        
        os.makedirs("data/raw", exist_ok=True)
        with open("data/raw/raw.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Run process command
        result = subprocess.run(
            ["make", "process"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        # Should succeed
        assert result.returncode == 0, f"Process failed: {result.stderr}"
        
        # Check output files exist
        assert Path("data/processed/train.jsonl").exists()
        assert Path("data/processed/val.jsonl").exists()
        assert Path("data/processed/test.jsonl").exists()
        
        # Validate processed data format
        with open("data/processed/train.jsonl") as f:
            processed_data = [json.loads(line) for line in f if line.strip()]
        
        assert len(processed_data) > 0, "No processed data found"
        
        # Check required fields
        for item in processed_data:
            assert "user" in item, "Missing 'user' field in processed data"
            assert "assistant" in item, "Missing 'assistant' field in processed data"
            assert "system" in item, "Missing 'system' field in processed data"
            assert len(item["user"]) > 0, "Empty user field"
            assert len(item["assistant"]) > 0, "Empty assistant field"
    
    def test_full_pipeline_produces_valid_training_data(self):
        """Test that full pipeline produces data suitable for training."""
        # Create test raw data (need enough samples for proper splitting)
        test_data = []
        for i in range(10):  # Create 10 samples to ensure proper splitting
            test_data.append({
                "instruction": f"Test question {i+1}", 
                "input": "context" if i % 2 == 0 else "", 
                "output": f"Test answer {i+1}"
            })
        
        os.makedirs("data/raw", exist_ok=True)
        with open("data/raw/raw.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Run full pipeline
        result = subprocess.run(
            ["make", "full-pipeline"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        # Should succeed
        assert result.returncode == 0, f"Full pipeline failed: {result.stderr}"
        
        # Check all expected output files exist
        expected_files = [
            "data/processed/train.jsonl",
            "data/processed/val.jsonl", 
            "data/processed/test.jsonl",
            "data/processed_with_style/train.jsonl",
            "data/processed_with_style/val.jsonl",
            "data/processed_with_style/test.jsonl",
            "data/rendered/train.jsonl",
            "data/rendered/val.jsonl",
            "data/rendered/test.jsonl"
        ]
        
        for file_path in expected_files:
            assert Path(file_path).exists(), f"Expected file not created: {file_path}"
            
            # Check file is not empty (except for validation files which might be empty with small datasets)
            with open(file_path) as f:
                content = f.read().strip()
                if "val.jsonl" not in file_path:  # Allow validation files to be empty for small test datasets
                    assert len(content) > 0, f"File is empty: {file_path}"
        
        # Validate rendered data format (should be ready for training)
        with open("data/rendered/train.jsonl") as f:
            rendered_data = [json.loads(line) for line in f if line.strip()]
        
        assert len(rendered_data) > 0, "No rendered training data found"
        
        for item in rendered_data:
            assert "input" in item, "Missing 'input' field in rendered data"
            assert "target" in item, "Missing 'target' field in rendered data"
            assert len(item["input"]) > 0, "Empty input field in rendered data"
            assert len(item["target"]) > 0, "Empty target field in rendered data"
    
    def test_dapt_pipeline_functionality(self):
        """Test DAPT pipeline actually processes DOCX files correctly."""
        # Check if DOCX files exist
        docx_files = list(Path("data/raw").glob("*.docx"))
        if not docx_files:
            pytest.skip("No DOCX files found for DAPT testing")
        
        # Run DAPT DOCX processing
        result = subprocess.run(
            ["make", "dapt-docx"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        # Should succeed
        assert result.returncode == 0, f"DAPT DOCX processing failed: {result.stderr}"
        
        # Check CPT data was created
        cpt_file = Path("data/processed/dpip_cpt.jsonl")
        assert cpt_file.exists(), "CPT data file not created"
        
        # Validate CPT data format
        with open(cpt_file) as f:
            cpt_data = [json.loads(line) for line in f if line.strip()]
        
        assert len(cpt_data) > 0, "No CPT data found"
        
        for item in cpt_data:
            assert "text" in item, "Missing 'text' field in CPT data"
            assert len(item["text"]) > 100, "CPT text chunks too short (should be substantial)"
            assert "dpip_doc" in item["text"], "CPT data missing expected format markers"


class TestConfigValidation:
    """Test configuration file validation."""
    
    def test_config_files_have_required_fields(self):
        """Test that config files contain all required fields."""
        config_files = {
            "configs/config_base.yaml": {
                "required_fields": ["seed", "mode", "model", "tuning", "train", "data"],
                "model_fields": ["name", "type"],
                "tuning_fields": ["mode", "backend"],
                "train_fields": ["epochs", "learning_rate", "output_dir"]
            },
            "configs/run_bnb.yaml": {
                "required_fields": ["include", "model", "tuning"],
                "model_fields": ["name", "type"],
                "tuning_fields": ["mode", "backend"]
            }
        }
        
        for config_file, requirements in config_files.items():
            if not Path(config_file).exists():
                continue
                
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            # Check top-level required fields
            for field in requirements["required_fields"]:
                assert field in config, f"Missing required field '{field}' in {config_file}"
            
            # Check model section
            if "model" in config and "model_fields" in requirements:
                for field in requirements["model_fields"]:
                    assert field in config["model"], f"Missing model field '{field}' in {config_file}"
            
            # Check tuning section
            if "tuning" in config and "tuning_fields" in requirements:
                for field in requirements["tuning_fields"]:
                    assert field in config["tuning"], f"Missing tuning field '{field}' in {config_file}"
    
    def test_config_values_are_valid(self):
        """Test that config values are within valid ranges."""
        with open("configs/config_base.yaml") as f:
            config = yaml.safe_load(f)
        
        # Check learning rate is reasonable
        lr = config["train"]["learning_rate"]
        if isinstance(lr, str):
            lr = float(lr)
        assert 0 < lr < 1, f"Learning rate {lr} is out of reasonable range"
        
        # Check epochs is positive
        epochs = config["train"]["epochs"]
        assert epochs > 0, f"Epochs {epochs} must be positive"
        
        # Check mode is valid
        mode = config["mode"]
        assert mode in ["sft", "cpt", "cpt_mixed"], f"Invalid mode: {mode}"
        
        # Check tuning mode is valid
        tuning_mode = config["tuning"]["mode"]
        assert tuning_mode in ["qlora", "lora", "full"], f"Invalid tuning mode: {tuning_mode}"


class TestErrorHandling:
    """Test that commands properly handle error conditions."""
    
    def test_process_fails_with_invalid_data(self):
        """Test that process command fails gracefully with invalid data."""
        # Create invalid raw data
        os.makedirs("data/raw", exist_ok=True)
        with open("data/raw/raw.jsonl", "w") as f:
            f.write('{"invalid": "data"}\n')  # Missing required fields
        
        result = subprocess.run(
            ["make", "process"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        # Should fail
        assert result.returncode != 0, "Process should fail with invalid data"
        # Error message might be in stderr
        error_output = (result.stdout + result.stderr).lower()
        assert "no valid rows" in error_output, "Should report no valid rows"
    
    def test_check_fails_with_missing_files(self):
        """Test that check command fails when required files are missing."""
        # Clean everything
        subprocess.run(["make", "clean"], cwd=".", capture_output=True)
        
        result = subprocess.run(
            ["make", "check"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        # Should fail
        assert result.returncode != 0, "Check should fail with missing files"
        assert "Training data not found" in result.stdout, "Should report missing training data"
    
    def test_train_fails_with_missing_config(self):
        """Test that train command fails with non-existent config."""
        result = subprocess.run(
            ["make", "train", "CONFIG=nonexistent.yaml"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        # Should fail
        assert result.returncode != 0, "Train should fail with missing config"


class TestOutputQuality:
    """Test the quality and correctness of command outputs."""
    
    def test_style_prompt_actually_modifies_data(self):
        """Test that style command actually modifies the data."""
        # Create test processed data
        os.makedirs("data/processed", exist_ok=True)
        original_data = [
            {"system": "", "user": "Test question", "assistant": "Test answer"}
        ]
        
        with open("data/processed/train.jsonl", "w") as f:
            for item in original_data:
                f.write(json.dumps(item) + '\n')
        
        # Run style command
        result = subprocess.run(
            ["make", "style"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0, f"Style command failed: {result.stderr}"
        
        # Check styled data
        with open("data/processed_with_style/train.jsonl") as f:
            styled_data = [json.loads(line) for line in f if line.strip()]
        
        assert len(styled_data) > 0, "No styled data found"
        
        # System prompt should be modified
        for item in styled_data:
            assert len(item["system"]) > 0, "System prompt should not be empty after styling"
            assert "concisely" in item["system"].lower(), "Style prompt not applied correctly"
    
    def test_render_produces_valid_chat_format(self):
        """Test that render command produces valid chat format."""
        # Create test processed data
        os.makedirs("data/processed", exist_ok=True)
        test_data = [
            {"system": "You are helpful", "user": "Hello", "assistant": "Hi there!"}
        ]
        
        with open("data/processed/train.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Run render command
        result = subprocess.run(
            ["make", "render"], 
            cwd=".", 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0, f"Render command failed: {result.stderr}"
        
        # Check rendered data
        with open("data/rendered/train.jsonl") as f:
            rendered_data = [json.loads(line) for line in f if line.strip()]
        
        assert len(rendered_data) > 0, "No rendered data found"
        
        for item in rendered_data:
            assert "input" in item, "Missing input field in rendered data"
            assert "target" in item, "Missing target field in rendered data"
            
            # Should contain chat format markers
            input_text = item["input"]
            assert "<|im_start|>" in input_text, "Missing chat format markers in input"
            assert "<|im_end|>" in input_text, "Missing chat format end markers in input"
            assert "system" in input_text, "Missing system section in rendered input"
            assert "user" in input_text, "Missing user section in rendered input"
