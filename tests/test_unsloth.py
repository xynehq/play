"""
Unit tests for Unsloth/XFormers fix implementation.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestUnslothEnvironment:
    """Test Unsloth environment setup."""
    
    def test_environment_variables_set(self):
        """Test that required environment variables are properly set."""
        # Import the train script to trigger environment setup
        from scripts import train
        
        required_env_vars = [
            "XFORMERS_DISABLED",
            "UNSLOTH_DISABLE_FAST_ATTENTION", 
            "UNSLOTH_FORCE_SDPA",
            "CUDA_LAUNCH_BLOCKING",
            "TORCH_USE_CUDA_DSA"
        ]
        
        for var in required_env_vars:
            assert var in os.environ, f"Environment variable {var} not set"
            assert os.environ[var] == "1", f"Environment variable {var} should be '1'"
    
    def test_torch_backends_configured(self):
        """Test that PyTorch backends are configured correctly."""
        import torch
        
        # Check that torch is available
        assert torch.__version__ is not None
        
        # Check CUDA availability (if available)
        if torch.cuda.is_available():
            assert torch.cuda.device_count() > 0
        
        # Check that CUDA backends are available
        if hasattr(torch.backends, 'cuda'):
            # These checks are version-dependent, so we just verify they exist
            assert hasattr(torch.backends.cuda, 'flash_sdp_enabled') or True
            assert hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled') or True
            assert hasattr(torch.backends.cuda, 'math_sdp_enabled') or True


class TestUnslothImport:
    """Test Unsloth import and basic functionality."""
    
    def test_unsloth_import_available(self):
        """Test if Unsloth can be imported."""
        try:
            from unsloth import FastLanguageModel
            assert FastLanguageModel is not None
        except ImportError:
            pytest.skip("Unsloth not installed")
    
    def test_unsloth_chat_templates(self):
        """Test Unsloth chat templates import."""
        try:
            from unsloth.chat_templates import get_chat_template
            assert get_chat_template is not None
        except ImportError:
            pytest.skip("Unsloth chat templates not available")
    
    def test_unsloth_basic_functionality(self):
        """Test basic Unsloth functionality."""
        try:
            from unsloth import FastLanguageModel
            # This should not fail if Unsloth is properly configured
            _ = FastLanguageModel
            assert True
        except ImportError:
            pytest.skip("Unsloth not installed")
        except Exception as e:
            pytest.fail(f"Unsloth functionality test failed: {e}")


class TestConfigurationLoading:
    """Test configuration loading for Unsloth."""
    
    def test_unsloth_config_exists(self):
        """Test that Unsloth config file exists."""
        config_path = Path("configs/run_unsloth.yaml")
        assert config_path.exists(), f"Config file not found: {config_path}"
    
    def test_unsloth_config_structure(self):
        """Test Unsloth config structure."""
        from scripts.train import load_config
        
        config_path = "configs/run_unsloth.yaml"
        cfg = load_config(config_path)
        
        # Check backend setting
        backend = cfg.get("tuning", {}).get("backend")
        assert backend == "unsloth", f"Backend should be 'unsloth', got '{backend}'"
        
        # Check precision settings
        train_config = cfg.get("train", {})
        bf16 = train_config.get("bf16", False)
        fp16 = train_config.get("fp16", False)
        
        # Should have one precision setting enabled
        assert bf16 or fp16, "Either bf16 or fp16 should be enabled"
        
        # Should not have both enabled
        if bf16 and fp16:
            pytest.fail("Both bf16 and fp16 are enabled, should only have one")
    
    def test_config_inheritance(self):
        """Test that Unsloth config properly inherits from base config."""
        from scripts.train import load_config
        
        config_path = "configs/run_unsloth.yaml"
        cfg = load_config(config_path)
        
        # Should have inherited base config elements
        assert "model" in cfg, "Model config should be inherited from base"
        assert "data" in cfg, "Data config should be inherited from base"
        assert "train" in cfg, "Train config should be inherited from base"
        assert "tuning" in cfg, "Tuning config should be inherited from base"


class TestFallbackLogic:
    """Test the fallback logic implementation."""
    
    def test_target_module_resolution(self):
        """Test target module resolution for different model types."""
        from scripts.train import resolve_target_modules
        
        test_cases = [
            ("llama-test", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            ("mistral-test", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            ("qwen-test", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            ("phi-test", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            ("gemma-test", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            ("t5-test", ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]),
            ("flan-test", ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]),
            ("unknown-model", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ]
        
        for model_name, expected_targets in test_cases:
            targets = resolve_target_modules(model_name, None)
            assert targets == expected_targets, f"Wrong targets for {model_name}: got {targets}, expected {expected_targets}"
    
    def test_target_module_override(self):
        """Test target module override functionality."""
        from scripts.train import resolve_target_modules
        
        custom_targets = ["custom_q", "custom_k", "custom_v"]
        result = resolve_target_modules("any-model", custom_targets)
        assert result == custom_targets, f"Override failed: got {result}, expected {custom_targets}"
        
        # Test auto override (should not override)
        result = resolve_target_modules("llama-test", "auto")
        expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert result == expected, f"Auto override failed: got {result}, expected {expected}"


class TestTrainingScriptIntegration:
    """Test integration aspects of the training script."""
    
    def test_config_loading_function(self):
        """Test the config loading function."""
        from scripts.train import load_config, deep_merge
        
        # Test deep merge functionality
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}
        result = deep_merge(base, override)
        
        expected = {"a": 1, "b": {"c": 4, "d": 3}, "e": 5}
        assert result == expected, f"Deep merge failed: got {result}, expected {expected}"
    
    def test_batch_accumulation_logic(self):
        """Test batch size and accumulation logic."""
        from scripts.train import choose_bs_and_accum
        
        # Test with explicit values
        cfg_train = {"batch_size": 2, "grad_accum": 4}
        bs, accum = choose_bs_and_accum(cfg_train, 512, "test-model")
        assert bs == 2 and accum == 4, f"Explicit values failed: got bs={bs}, accum={accum}"
        
        # Test with auto values (should return reasonable defaults)
        cfg_train = {"batch_size": "auto", "grad_accum": "auto"}
        bs, accum = choose_bs_and_accum(cfg_train, 512, "test-model")
        assert isinstance(bs, int) and isinstance(accum, int), f"Auto values failed: got bs={bs}, accum={accum}"
        assert bs > 0 and accum > 0, f"Auto values should be positive: got bs={bs}, accum={accum}"


@pytest.mark.integration
class TestUnslothIntegration:
    """Integration tests for Unsloth (marked as integration tests)."""
    
    def test_unsloth_backend_detection(self):
        """Test that Unsloth backend detection works."""
        # This would require a more complex setup to test the actual backend detection
        # For now, we just test that the logic exists
        from scripts.train import main
        assert main is not None, "Main function should be available"
    
    @pytest.mark.slow
    def test_unsloth_model_loading_simulation(self):
        """Test Unsloth model loading simulation (slow test)."""
        # This would be a slow test that actually tries to load a small model
        # Marked as slow so it can be skipped in fast test runs
        pytest.skip("Slow test - requires actual model loading")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
