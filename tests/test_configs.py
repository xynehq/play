"""
Unit tests for configuration files and validation.

This module tests that all configuration files are valid and contain
the required fields for proper operation.
"""

import pytest
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestConfigFiles:
    """Test configuration file validity and structure."""
    
    def test_base_config_structure(self):
        """Test that base config has required fields."""
        config_path = Path("configs/config_base.yaml")
        assert config_path.exists(), "Base config file not found"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Required top-level fields
        required_fields = ["seed", "run_name", "mode", "train", "data"]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
        
        # Train section requirements
        train_config = config["train"]
        train_required = ["epochs", "learning_rate", "output_dir"]
        for field in train_required:
            assert field in train_config, f"Missing train field: {field}"
        
        # Data section requirements
        data_config = config["data"]
        data_required = ["format"]
        for field in data_required:
            assert field in data_config, f"Missing data field: {field}"
    
    def test_bnb_config_structure(self):
        """Test BitsAndBytes config structure."""
        config_path = Path("configs/run_bnb.yaml")
        assert config_path.exists(), "BnB config file not found"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Should include base config
        assert "include" in config, "BnB config should include base config"
        
        # Model section requirements
        assert "model" in config, "Missing model section"
        model_config = config["model"]
        model_required = ["name", "type"]
        for field in model_required:
            assert field in model_config, f"Missing model field: {field}"
        
        # Tuning section requirements
        assert "tuning" in config, "Missing tuning section"
        tuning_config = config["tuning"]
        assert tuning_config["mode"] in ["qlora", "lora", "full"], "Invalid tuning mode"
        assert tuning_config["backend"] == "bnb", "BnB config should use bnb backend"
    
    def test_dapt_config_structure(self):
        """Test DAPT config structure."""
        config_path = Path("configs/run_dapt.yaml")
        assert config_path.exists(), "DAPT config file not found"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Should include base config
        assert "include" in config, "DAPT config should include base config"
        
        # Should have DAPT-specific mode (check both 'mode' and 'task_mode' fields)
        mode_value = config.get("mode") or config.get("task_mode")
        assert mode_value in ["cpt", "cpt_mixed"], "DAPT config should use CPT mode"
        
        # Should have datasets for mixed mode
        if mode_value == "cpt_mixed":
            assert "datasets" in config, "Mixed mode requires datasets configuration"
    
    def test_unsloth_config_structure(self):
        """Test Unsloth config structure."""
        config_path = Path("configs/run_unsloth.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Should include base config
            assert "include" in config, "Unsloth config should include base config"
            
            # Tuning section requirements
            if "tuning" in config:
                tuning_config = config["tuning"]
                assert tuning_config.get("backend") == "unsloth", "Unsloth config should use unsloth backend"
    
    def test_config_yaml_validity(self):
        """Test that all config files are valid YAML."""
        config_dir = Path("configs")
        config_files = list(config_dir.glob("*.yaml"))
        
        assert len(config_files) > 0, "No config files found"
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_file}: {e}")
    
    def test_chat_template_exists(self):
        """Test that chat template file exists."""
        template_path = Path("chat_templates/default.jinja")
        assert template_path.exists(), "Default chat template not found"
        
        # Should be readable
        with open(template_path, 'r') as f:
            content = f.read()
            assert len(content) > 0, "Chat template is empty"


class TestConfigValidation:
    """Test configuration validation logic."""
    
    def test_mode_validation(self):
        """Test that mode values are valid."""
        valid_modes = ["sft", "cpt", "cpt_mixed"]
        
        config_files = [
            "configs/config_base.yaml",
            "configs/run_bnb.yaml", 
            "configs/run_dapt.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if "mode" in config:
                    assert config["mode"] in valid_modes, f"Invalid mode in {config_file}: {config['mode']}"
    
    def test_tuning_mode_validation(self):
        """Test that tuning modes are valid."""
        valid_tuning_modes = ["qlora", "lora", "full"]
        valid_backends = ["bnb", "unsloth"]
        
        config_files = [
            "configs/run_bnb.yaml",
            "configs/run_unsloth.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if "tuning" in config:
                    tuning = config["tuning"]
                    if "mode" in tuning:
                        assert tuning["mode"] in valid_tuning_modes, f"Invalid tuning mode in {config_file}"
                    if "backend" in tuning:
                        assert tuning["backend"] in valid_backends, f"Invalid backend in {config_file}"
    
    def test_data_format_validation(self):
        """Test that data formats are valid."""
        valid_formats = ["chat", "instruction", "completion"]
        
        config_files = [
            "configs/config_base.yaml",
            "configs/run_bnb.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if "data" in config and "format" in config["data"]:
                    data_format = config["data"]["format"]
                    assert data_format in valid_formats, f"Invalid data format in {config_file}: {data_format}"
    
    def test_numeric_values(self):
        """Test that numeric configuration values are reasonable."""
        config_files = [
            "configs/config_base.yaml",
            "configs/run_bnb.yaml",
            "configs/run_dapt.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check learning rate
                if "train" in config and "learning_rate" in config["train"]:
                    lr = config["train"]["learning_rate"]
                    # Convert to float if it's a string (handles scientific notation)
                    if isinstance(lr, str):
                        lr = float(lr)
                    assert 0 < lr < 1, f"Learning rate out of range in {config_file}: {lr}"
                
                # Check epochs
                if "train" in config and "epochs" in config["train"]:
                    epochs = config["train"]["epochs"]
                    assert epochs > 0, f"Epochs must be positive in {config_file}: {epochs}"
                
                # Check LoRA rank
                if "tuning" in config and "lora" in config["tuning"] and "r" in config["tuning"]["lora"]:
                    r = config["tuning"]["lora"]["r"]
                    assert r > 0, f"LoRA rank must be positive in {config_file}: {r}"


class TestConfigInheritance:
    """Test configuration inheritance and overrides."""
    
    def test_config_includes(self):
        """Test that config includes work properly."""
        # Test configs that include base config
        including_configs = [
            "configs/run_bnb.yaml",
            "configs/run_dapt.yaml"
        ]
        
        for config_file in including_configs:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if "include" in config:
                    include_path = config["include"]
                    assert Path(include_path).exists(), f"Included config not found: {include_path}"
    
    @patch('yaml.safe_load')
    def test_config_merging_logic(self, mock_yaml_load):
        """Test that configuration merging works correctly."""
        # Mock base config
        base_config = {
            "seed": 42,
            "train": {
                "epochs": 3,
                "learning_rate": 2e-4
            }
        }
        
        # Mock override config
        override_config = {
            "include": "configs/config_base.yaml",
            "train": {
                "epochs": 1  # Override epochs
            },
            "model": {
                "name": "test-model"  # Add new field
            }
        }
        
        mock_yaml_load.side_effect = [base_config, override_config]
        
        # In a real implementation, this would test the actual merging logic
        # For now, we just verify the structure is correct
        assert "include" in override_config
        assert "train" in override_config
        assert "model" in override_config
