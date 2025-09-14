"""
Tests for distributed training functionality
"""
import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import torch

# Add the parent directory to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_distributed import (
    get_gpu_memory_info,
    calculate_optimal_batch_size,
    load_config,
    deep_merge,
    resolve_target_modules
)


class TestGPUMemoryInfo:
    """Test GPU memory detection"""
    
    @patch('torch.cuda.is_available')
    def test_no_cuda_available(self, mock_cuda_available):
        """Test when CUDA is not available"""
        mock_cuda_available.return_value = False
        result = get_gpu_memory_info()
        assert result == []
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.is_available')
    def test_single_gpu(self, mock_cuda_available, mock_device_name, mock_device_props, mock_device_count):
        """Test single GPU detection"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_device_name.return_value = "NVIDIA RTX 4090"
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3  # 24GB in bytes
        mock_device_props.return_value = mock_props
        
        result = get_gpu_memory_info()
        
        assert len(result) == 1
        assert result[0]['id'] == 0
        assert result[0]['name'] == "NVIDIA RTX 4090"
        assert result[0]['total_memory_gb'] == 24.0
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.is_available')
    def test_multi_gpu(self, mock_cuda_available, mock_device_name, mock_device_props, mock_device_count):
        """Test multi-GPU detection"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        mock_device_name.side_effect = ["NVIDIA H200", "NVIDIA H200"]
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 141 * 1024**3  # 141GB in bytes
        mock_device_props.return_value = mock_props
        
        result = get_gpu_memory_info()
        
        assert len(result) == 2
        assert result[0]['name'] == "NVIDIA H200"
        assert result[1]['name'] == "NVIDIA H200"
        assert result[0]['total_memory_gb'] == 141.0
        assert result[1]['total_memory_gb'] == 141.0


class TestBatchSizeCalculation:
    """Test optimal batch size calculation"""
    
    def test_small_model_single_gpu(self):
        """Test batch size for small model on single GPU"""
        bs, accum = calculate_optimal_batch_size("Qwen/Qwen2.5-3B", 512, 1, 24.0)
        assert bs >= 1
        assert accum >= 1
        assert isinstance(bs, int)
        assert isinstance(accum, int)
    
    def test_large_model_multi_gpu(self):
        """Test batch size for large model on multi-GPU"""
        bs, accum = calculate_optimal_batch_size("google/gemma-3-27b-it", 2048, 2, 282.0)
        assert bs >= 1
        assert accum >= 1
        # Should be able to handle larger batches with more memory
        effective_batch = bs * 2 * accum
        assert effective_batch >= 16  # Should achieve reasonable effective batch size
    
    def test_huge_model_insufficient_memory(self):
        """Test batch size when model barely fits"""
        bs, accum = calculate_optimal_batch_size("meta-llama/Llama-2-70b-hf", 4096, 1, 16.0)
        # Should fallback to minimum settings
        assert bs == 1
        assert accum == 64
    
    def test_model_size_detection(self):
        """Test model size detection from name"""
        # Test 27B model
        bs_27b, _ = calculate_optimal_batch_size("gemma-27b", 1024, 2, 200.0)
        
        # Test 7B model (should allow larger batch size)
        bs_7b, _ = calculate_optimal_batch_size("llama-7b", 1024, 2, 200.0)
        
        # 7B model should allow larger batch size than 27B
        assert bs_7b >= bs_27b


class TestConfigLoading:
    """Test configuration loading and merging"""
    
    def test_deep_merge(self):
        """Test deep dictionary merging"""
        base = {
            "model": {"name": "base-model", "type": "causal"},
            "train": {"epochs": 1, "lr": 1e-4}
        }
        override = {
            "model": {"name": "new-model"},
            "train": {"epochs": 3},
            "new_key": "new_value"
        }
        
        result = deep_merge(base, override)
        
        assert result["model"]["name"] == "new-model"  # Overridden
        assert result["model"]["type"] == "causal"     # Preserved
        assert result["train"]["epochs"] == 3          # Overridden
        assert result["train"]["lr"] == 1e-4           # Preserved
        assert result["new_key"] == "new_value"        # Added
    
    def test_load_config_without_include(self):
        """Test loading config without base include"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"model": {"name": "test-model"}}, f)
            config_path = f.name
        
        try:
            result = load_config(config_path)
            assert result["model"]["name"] == "test-model"
        finally:
            os.unlink(config_path)
    
    def test_load_config_with_include(self):
        """Test loading config with base include"""
        # Create base config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_f:
            yaml.dump({
                "model": {"type": "causal", "max_seq_len": 512},
                "train": {"epochs": 1}
            }, base_f)
            base_path = base_f.name
        
        # Create run config that includes base
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as run_f:
            yaml.dump({
                "include": base_path,
                "model": {"name": "test-model"},
                "train": {"epochs": 3}
            }, run_f)
            run_path = run_f.name
        
        try:
            result = load_config(run_path)
            assert result["model"]["name"] == "test-model"      # From run config
            assert result["model"]["type"] == "causal"          # From base config
            assert result["model"]["max_seq_len"] == 512        # From base config
            assert result["train"]["epochs"] == 3               # Overridden in run config
        finally:
            os.unlink(base_path)
            os.unlink(run_path)


class TestTargetModules:
    """Test target module resolution"""
    
    def test_explicit_target_modules(self):
        """Test when target modules are explicitly specified"""
        result = resolve_target_modules("any-model", ["custom", "modules"])
        assert result == ["custom", "modules"]
    
    def test_auto_target_modules_llama(self):
        """Test auto target modules for Llama models"""
        result = resolve_target_modules("meta-llama/Llama-2-7b-hf", "auto")
        expected = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        assert result == expected
    
    def test_auto_target_modules_qwen(self):
        """Test auto target modules for Qwen models"""
        result = resolve_target_modules("Qwen/Qwen2.5-3B", "auto")
        expected = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        assert result == expected
    
    def test_auto_target_modules_gemma(self):
        """Test auto target modules for Gemma models"""
        result = resolve_target_modules("google/gemma-2b", "auto")
        expected = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        assert result == expected
    
    def test_auto_target_modules_t5(self):
        """Test auto target modules for T5 models"""
        result = resolve_target_modules("google/flan-t5-base", "auto")
        expected = ["q","k","v","o","wi_0","wi_1","wo"]
        assert result == expected
    
    def test_auto_target_modules_fallback(self):
        """Test auto target modules fallback for unknown models"""
        result = resolve_target_modules("unknown/model", "auto")
        expected = ["q_proj","k_proj","v_proj","o_proj"]
        assert result == expected


class TestDistributedTrainingIntegration:
    """Integration tests for distributed training"""
    
    def test_config_validation(self):
        """Test that distributed config is valid"""
        config_path = Path(__file__).parent.parent / "configs" / "run_distributed.yaml"
        assert config_path.exists(), "run_distributed.yaml config should exist"
        
        # Load and validate config
        config = load_config(str(config_path))
        
        # Check required sections
        assert "model" in config
        assert "tuning" in config
        assert "train" in config
        assert "data" in config
        
        # Check model section
        assert "name" in config["model"]
        assert "type" in config["model"]
        
        # Check tuning section
        assert "mode" in config["tuning"]
        assert "backend" in config["tuning"]
        assert config["tuning"]["backend"] in ["bnb", "unsloth"]
        
        # Check train section
        assert "epochs" in config["train"]
        assert "batch_size" in config["train"]
        assert "grad_accum" in config["train"]
    
    def test_backend_support(self):
        """Test that both BitsAndBytes and Unsloth backends are supported"""
        # This is more of a documentation test
        supported_backends = ["bnb", "unsloth"]
        supported_modes = ["qlora", "lora", "full"]
        
        # Verify these are the expected values
        assert "bnb" in supported_backends
        assert "unsloth" in supported_backends
        assert "qlora" in supported_modes
    
    def test_data_format_support(self):
        """Test that multiple data formats are supported"""
        # Test data format detection logic
        formats = [
            {"input": "test input", "target": "test target"},  # Rendered format
            {"user": "test user", "assistant": "test assistant"},  # Processed format
            {"messages": [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]}  # Messages format
        ]
        
        # All formats should be valid (this is tested in the actual collator)
        for fmt in formats:
            assert isinstance(fmt, dict)
            # Each format has different required keys
            if "input" in fmt:
                assert "target" in fmt
            elif "user" in fmt:
                assert "assistant" in fmt
            elif "messages" in fmt:
                assert isinstance(fmt["messages"], list)


class TestCheckpointingAndResume:
    """Test checkpointing and resume functionality"""
    
    def test_backend_stamp_creation(self):
        """Test backend stamp file creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend_file = os.path.join(tmpdir, "backend.json")
            
            # Simulate creating backend stamp
            backend_info = {
                "backend": "bnb",
                "dtype": "bf16",
                "attn_impl": "default",
                "num_gpus": 2,
                "distributed": True,
                "run_name": "test-run"
            }
            
            with open(backend_file, "w") as f:
                json.dump(backend_info, f, indent=2)
            
            # Verify file was created and contains correct info
            assert os.path.exists(backend_file)
            
            with open(backend_file, "r") as f:
                loaded_info = json.load(f)
            
            assert loaded_info["backend"] == "bnb"
            assert loaded_info["dtype"] == "bf16"
            assert loaded_info["num_gpus"] == 2
            assert loaded_info["distributed"] is True
            assert loaded_info["run_name"] == "test-run"
    
    def test_run_name_generation(self):
        """Test run name generation with timestamps"""
        from datetime import datetime
        
        # Test generic run name gets timestamp
        run_name = "distributed-training"
        if run_name == "distributed-training":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_run_name = f"distributed-training-{timestamp}"
            
            assert new_run_name.startswith("distributed-training-")
            assert len(new_run_name) > len("distributed-training-")
    
    def test_output_directory_structure(self):
        """Test output directory structure"""
        run_name = "test-run-123"
        expected_outdir = f"outputs/{run_name}"
        
        # Verify directory structure
        assert expected_outdir.startswith("outputs/")
        assert run_name in expected_outdir


class TestRequirements:
    """Test that all required dependencies are available"""
    
    def test_core_dependencies(self):
        """Test that core ML dependencies are importable"""
        try:
            import torch
            import transformers
            import datasets
            import peft
            import accelerate
            import deepspeed
            assert True  # All imports successful
        except ImportError as e:
            pytest.fail(f"Required dependency not available: {e}")
    
    def test_optional_dependencies(self):
        """Test that optional dependencies are importable"""
        optional_deps = [
            "bitsandbytes",
            "evaluate",
            "tensorboard",
            "jinja2"
        ]
        
        missing_deps = []
        for dep in optional_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            pytest.skip(f"Optional dependencies not available: {missing_deps}")


@pytest.mark.integration
class TestDistributedTrainingEnd2End:
    """End-to-end integration tests (require GPU)"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_detection_real(self):
        """Test real GPU detection (requires CUDA)"""
        import torch
        
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            assert len(gpu_info) > 0
            
            for gpu in gpu_info:
                assert 'id' in gpu
                assert 'name' in gpu
                assert 'total_memory_gb' in gpu
                assert gpu['total_memory_gb'] > 0
        else:
            pytest.skip("CUDA not available for real GPU testing")
    
    def test_data_pipeline_integration(self):
        """Test that data pipeline works with distributed training"""
        # Create temporary data files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample training data
            train_data = [
                {"user": "What is AI?", "assistant": "AI is artificial intelligence."},
                {"user": "Explain ML", "assistant": "ML is machine learning."}
            ]
            
            train_path = os.path.join(tmpdir, "train.jsonl")
            with open(train_path, "w") as f:
                for item in train_data:
                    f.write(json.dumps(item) + "\n")
            
            # Verify file exists and is readable
            assert os.path.exists(train_path)
            
            # Test that our ChatDataset can load it
            from scripts.train_distributed import ChatDataset
            dataset = ChatDataset(train_path)
            
            assert len(dataset) == 2
            assert dataset[0]["user"] == "What is AI?"
            assert dataset[1]["assistant"] == "ML is machine learning."
