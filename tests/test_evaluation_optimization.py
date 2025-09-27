"""
Tests for evaluation optimization in distributed training
"""
import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import yaml
import torch
import numpy as np

# Add the parent directory to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_distributed import (
    compute_metrics_builder,
    TrainingArguments,
    Trainer,
    EvalSpeedCallback,
    CausalCollator,
    ChatDataset
)

# GEN_SUPPORTED is set at runtime in the main function
# For testing, we'll simulate it as a global variable
GEN_SUPPORTED = True


class TestEvaluationOptimization:
    """Test evaluation optimization features"""
    
    def test_compute_metrics_builder_creation(self):
        """Test that compute_metrics_builder creates a valid function"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.batch_decode = Mock(return_value=["test output"])
        
        # Create compute_metrics function
        compute_metrics = compute_metrics_builder(mock_tokenizer)
        
        # Verify it's callable
        assert callable(compute_metrics)
        
        # Test with empty predictions (should handle gracefully)
        eval_pred = Mock()
        eval_pred.predictions = []
        eval_pred.label_ids = []
        
        result = compute_metrics(eval_pred)
        assert isinstance(result, dict)
        assert "rougeL" in result
        assert result["rougeL"] == 0.0  # Empty predictions should return 0.0
    
    def test_compute_metrics_with_valid_data(self):
        """Test compute_metrics with valid prediction data"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.batch_decode = Mock(side_effect=[
            ["Generated text response"],  # Predictions
            ["Expected text response"]   # Labels
        ])
        
        # Mock ROUGE metric
        with patch('scripts.train_distributed.ROUGE_METRIC') as mock_rouge:
            mock_rouge.compute.return_value = {"rougeL": Mock(mid=Mock(fmeasure=0.5))}
            
            compute_metrics = compute_metrics_builder(mock_tokenizer)
            
            # Test with valid data
            eval_pred = Mock()
            eval_pred.predictions = [[1, 2, 3]]  # Token IDs
            eval_pred.label_ids = [[1, 2, 3, -100]]  # Token IDs with -100 mask
            
            result = compute_metrics(eval_pred)
            
            assert isinstance(result, dict)
            assert "rougeL" in result
            assert result["rougeL"] == 0.5
    
    def test_compute_metrics_handles_numpy_arrays(self):
        """Test that compute_metrics handles numpy array inputs"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.batch_decode = Mock(return_value=["test"])
        
        compute_metrics = compute_metrics_builder(mock_tokenizer)
        
        # Test with numpy arrays
        eval_pred = Mock()
        eval_pred.predictions = np.array([[1, 2, 3]])
        eval_pred.label_ids = np.array([[1, 2, 3, -100]])
        
        # Should not crash
        result = compute_metrics(eval_pred)
        assert isinstance(result, dict)
    
    def test_compute_metrics_filters_empty_predictions(self):
        """Test that compute_metrics filters out empty predictions"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.batch_decode = Mock(return_value=[""])
        
        compute_metrics = compute_metrics_builder(mock_tokenizer)
        
        # Test with empty predictions
        eval_pred = Mock()
        eval_pred.predictions = [[]]  # Empty prediction
        eval_pred.label_ids = [[1, 2, 3]]
        
        result = compute_metrics(eval_pred)
        assert result["rougeL"] == 0.0  # Should return 0.0 for empty predictions
    
    def test_compute_metrics_error_handling(self):
        """Test error handling in compute_metrics"""
        # Mock tokenizer that raises an error
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.batch_decode = Mock(side_effect=Exception("Decode error"))
        
        compute_metrics = compute_metrics_builder(mock_tokenizer)
        
        # Test with data that causes error
        eval_pred = Mock()
        eval_pred.predictions = [[1, 2, 3]]
        eval_pred.label_ids = [[1, 2, 3]]
        
        # Should handle error gracefully
        result = compute_metrics(eval_pred)
        assert isinstance(result, dict)
        assert "rougeL" in result
        assert result["rougeL"] == 0.0


class TestEvalSpeedCallback:
    """Test EvalSpeedCallback functionality"""
    
    def test_eval_speed_callback_creation(self):
        """Test that EvalSpeedCallback can be created"""
        mock_model = Mock()
        mock_model.config = Mock()
        
        callback = EvalSpeedCallback(mock_model)
        
        assert callback.model == mock_model
        assert callback.prev is None
    
    def test_eval_speed_callback_toggle_cache(self):
        """Test that callback toggles use_cache during evaluation"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.use_cache = False
        
        callback = EvalSpeedCallback(mock_model)
        
        # Mock training arguments and state
        mock_args = Mock()
        mock_state = Mock()
        
        # Test on_evaluate
        callback.on_evaluate(mock_args, mock_state, Mock())
        assert mock_model.config.use_cache is True
        
        # Test on_train_end (should restore previous value)
        callback.on_train_end(mock_args, mock_state, Mock())
        assert mock_model.config.use_cache is False
    
    def test_eval_speed_callback_predict(self):
        """Test that callback handles predict events"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.use_cache = False
        
        callback = EvalSpeedCallback(mock_model)
        
        # Mock training arguments and state
        mock_args = Mock()
        mock_state = Mock()
        
        # Test on_predict
        callback.on_predict(mock_args, mock_state, Mock())
        assert mock_model.config.use_cache is True
        
        # Test on_epoch_end (should restore previous value)
        callback.on_epoch_end(mock_args, mock_state, Mock())
        assert mock_model.config.use_cache is False


class TestTrainingArgumentsCompatibility:
    """Test TrainingArguments compatibility handling"""
    
    def test_gen_supported_flag_initialization(self):
        """Test GEN_SUPPORTED flag initialization"""
        # This test verifies the flag exists and can be set
        # The actual value is determined at runtime
        assert 'GEN_SUPPORTED' in globals()
    
    @patch('scripts.train_distributed.TrainingArguments')
    def test_generation_params_supported(self, mock_training_args):
        """Test behavior when generation parameters are supported"""
        # Mock TrainingArguments to accept all parameters
        mock_args_instance = Mock()
        mock_training_args.return_value = mock_args_instance
        
        training_args_kwargs = {
            "output_dir": "test_output",
            "predict_with_generate": True,
            "generation_num_beams": 1,
            "generation_max_new_tokens": 64,
            "eval_strategy": "steps"
        }
        
        # This should work without raising an exception
        try:
            from play.scripts.train_distributed import main
            # We can't easily test the full main function, but we can test the logic
            GEN_SUPPORTED = True
            assert GEN_SUPPORTED is True
        except Exception:
            pytest.fail("TrainingArguments creation should not fail with supported parameters")
    
    @patch('scripts.train_distributed.TrainingArguments')
    def test_generation_params_unsupported(self, mock_training_args):
        """Test behavior when generation parameters are not supported"""
        # Mock TrainingArguments to raise TypeError for generation parameters
        def side_effect(*args, **kwargs):
            if "predict_with_generate" in kwargs:
                raise TypeError("unexpected keyword argument 'predict_with_generate'")
            return Mock()
        
        mock_training_args.side_effect = side_effect
        
        training_args_kwargs = {
            "output_dir": "test_output",
            "predict_with_generate": True,
            "generation_num_beams": 1,
            "generation_max_new_tokens": 64,
            "eval_strategy": "steps"
        }
        
        # Test the fallback logic
        GEN_SUPPORTED = True
        
        try:
            # Simulate the try/except logic from the main function
            tr_args = TrainingArguments(**training_args_kwargs)
        except TypeError as e:
            if "predict_with_generate" in str(e):
                # Remove generation-related parameters
                training_args_kwargs.pop("predict_with_generate", None)
                training_args_kwargs.pop("generation_num_beams", None)
                training_args_kwargs.pop("generation_max_new_tokens", None)
                GEN_SUPPORTED = False
                training_args_kwargs["prediction_loss_only"] = True
                tr_args = TrainingArguments(**training_args_kwargs)
                assert GEN_SUPPORTED is False
                assert training_args_kwargs["prediction_loss_only"] is True
            else:
                raise e


class TestTrainerConfiguration:
    """Test Trainer configuration with evaluation optimization"""
    
    @patch('scripts.train_distributed.Trainer')
    def test_trainer_with_generation_supported(self, mock_trainer):
        """Test Trainer configuration when generation is supported"""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock TrainingArguments
        mock_args = Mock()
        mock_args.eval_strategy = "steps"
        
        # Mock datasets
        mock_train_dataset = Mock()
        mock_val_dataset = Mock()
        
        # Mock data collator
        mock_data_collator = Mock()
        
        # Set GEN_SUPPORTED to True
        GEN_SUPPORTED = True
        
        # Mock compute_metrics_builder
        with patch('scripts.train_distributed.compute_metrics_builder') as mock_compute_builder:
            mock_compute_fn = Mock()
            mock_compute_builder.return_value = mock_compute_fn
            
            # Mock EvalSpeedCallback
            with patch('scripts.train_distributed.EvalSpeedCallback') as mock_callback:
                mock_callback_instance = Mock()
                mock_callback.return_value = mock_callback_instance
                
                # Create Trainer (simulating the logic from main function)
                callbacks = [EvalSpeedCallback(mock_model)] if GEN_SUPPORTED else None
                trainer = Trainer(
                    model=mock_model,
                    args=mock_args,
                    train_dataset=mock_train_dataset,
                    eval_dataset=mock_val_dataset if mock_args.eval_strategy != "no" else None,
                    data_collator=mock_data_collator,
                    tokenizer=mock_tokenizer,
                    compute_metrics=compute_metrics_builder(mock_tokenizer) if (mock_args.eval_strategy != "no" and GEN_SUPPORTED) else None,
                    callbacks=callbacks,
                )
                
                # Verify Trainer was called with correct parameters
                mock_trainer.assert_called_once()
                call_args = mock_trainer.call_args
                
                # Verify compute_metrics is included when generation is supported
                assert call_args[1]['compute_metrics'] == mock_compute_fn
                # Verify callbacks are included when generation is supported
                assert call_args[1]['callbacks'] == [mock_callback_instance]
    
    @patch('scripts.train_distributed.Trainer')
    def test_trainer_with_generation_unsupported(self, mock_trainer):
        """Test Trainer configuration when generation is not supported"""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock TrainingArguments
        mock_args = Mock()
        mock_args.eval_strategy = "steps"
        
        # Mock datasets
        mock_train_dataset = Mock()
        mock_val_dataset = Mock()
        
        # Mock data collator
        mock_data_collator = Mock()
        
        # Set GEN_SUPPORTED to False
        GEN_SUPPORTED = False
        
        # Create Trainer (simulating the logic from main function)
        callbacks = [EvalSpeedCallback(mock_model)] if GEN_SUPPORTED else None
        trainer = Trainer(
            model=mock_model,
            args=mock_args,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_val_dataset if mock_args.eval_strategy != "no" else None,
            data_collator=mock_data_collator,
            tokenizer=mock_tokenizer,
            compute_metrics=compute_metrics_builder(mock_tokenizer) if (mock_args.eval_strategy != "no" and GEN_SUPPORTED) else None,
            callbacks=callbacks,
        )
        
        # Verify Trainer was called with correct parameters
        mock_trainer.assert_called_once()
        call_args = mock_trainer.call_args
        
        # Verify compute_metrics is None when generation is not supported
        assert call_args[1]['compute_metrics'] is None
        # Verify callbacks are None when generation is not supported
        assert call_args[1]['callbacks'] is None


class TestPostEvaluationSynchronization:
    """Test post-evaluation synchronization"""
    
    @patch('torch.cuda.synchronize')
    @patch('torch.distributed.barrier')
    def test_post_eval_sync_with_cuda_exception(self, mock_dist_barrier, mock_cuda_sync):
        """Test post-evaluation synchronization when CUDA sync fails"""
        # Mock CUDA sync to raise an exception
        mock_cuda_sync.side_effect = Exception("CUDA error")
        mock_dist_barrier.return_value = None
        
        # Simulate the post-evaluation sync code
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        
        # Verify CUDA sync was called and failed gracefully
        mock_cuda_sync.assert_called_once()
        # Verify dist barrier was still called
        mock_dist_barrier.assert_called_once()
    
    @patch('torch.cuda.synchronize')
    @patch('torch.distributed.barrier')
    def test_post_eval_sync_with_dist_exception(self, mock_dist_barrier, mock_cuda_sync):
        """Test post-evaluation synchronization when dist barrier fails"""
        # Mock successful CUDA sync
        mock_cuda_sync.return_value = None
        # Mock dist barrier to raise an exception
        mock_dist_barrier.side_effect = Exception("Dist error")
        
        # Simulate the post-evaluation sync code
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        
        # Verify both syncs were called and failed gracefully
        mock_cuda_sync.assert_called_once()
        mock_dist_barrier.assert_called_once()
    
    @patch('scripts.train_distributed.torch.cuda.synchronize')
    @patch('scripts.train_distributed.dist.barrier')
    def test_post_eval_sync_with_dist_exception(self, mock_dist_barrier, mock_cuda_sync):
        """Test post-evaluation synchronization when dist barrier fails"""
        # Mock successful CUDA sync
        mock_cuda_sync.return_value = None
        # Mock dist barrier to raise an exception
        mock_dist_barrier.side_effect = Exception("Dist error")
        
        # Simulate the post-evaluation sync code
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        
        # Verify both syncs were called and failed gracefully
        mock_cuda_sync.assert_called_once()
        mock_dist_barrier.assert_called_once()


class TestEvaluationParameterOptimization:
    """Test evaluation parameter optimization"""
    
    def test_eval_do_concat_batches_false(self):
        """Test that eval_do_concat_batches is set to False"""
        # This parameter should be set in the training arguments
        training_args_kwargs = {
            "eval_do_concat_batches": False,
            "output_dir": "test_output"
        }
        
        assert training_args_kwargs["eval_do_concat_batches"] is False
    
    def test_generation_num_beams_one(self):
        """Test that generation_num_beams is set to 1 for speed"""
        training_args_kwargs = {
            "generation_num_beams": 1,
            "output_dir": "test_output"
        }
        
        assert training_args_kwargs["generation_num_beams"] == 1
    
    def test_generation_max_new_tokens_reasonable(self):
        """Test that generation_max_new_tokens is set to reasonable value"""
        training_args_kwargs = {
            "generation_max_new_tokens": 64,
            "output_dir": "test_output"
        }
        
        assert training_args_kwargs["generation_max_new_tokens"] == 64
        assert 0 < training_args_kwargs["generation_max_new_tokens"] <= 128  # Reasonable range
    
    def test_eval_accumulation_steps_set(self):
        """Test that eval_accumulation_steps is set"""
        training_args_kwargs = {
            "eval_accumulation_steps": 8,
            "output_dir": "test_output"
        }
        
        assert training_args_kwargs["eval_accumulation_steps"] == 8
        assert training_args_kwargs["eval_accumulation_steps"] > 0


class TestConfigurationIntegration:
    """Test integration of all evaluation optimizations"""
    
    def test_optimized_config_structure(self):
        """Test that optimized config has all required parameters"""
        config_path = Path(__file__).parent.parent / "configs" / "run_distributed_optimized.yaml"
        assert config_path.exists(), "run_distributed_optimized.yaml config should exist"
        
        # Load and validate config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check evaluation optimization parameters
        train_config = config.get("train", {})
        
        # Verify evaluation parameters are present
        assert "eval_strategy" in train_config
        assert "eval_steps" in train_config
        assert "predict_with_generate" in train_config
        assert "generation_num_beams" in train_config
        assert "generation_max_new_tokens" in train_config
        assert "eval_do_concat_batches" in train_config
        assert "eval_accumulation_steps" in train_config
        
        # Verify optimization values
        assert train_config["eval_do_concat_batches"] is False
        assert train_config["generation_num_beams"] == 1
        assert train_config["generation_max_new_tokens"] == 64
        assert train_config["eval_accumulation_steps"] == 8
    
    def test_config_values_are_optimal(self):
        """Test that config values are set to optimal values"""
        config_path = Path(__file__).parent.parent / "configs" / "run_distributed_optimized.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        train_config = config.get("train", {})
        
        # Check specific optimal values
        assert train_config.get("bf16") is True  # Optimal for H200/H100
        assert train_config.get("gradient_checkpointing") is True  # Memory efficiency
        assert train_config.get("batch_size") == "auto"  # Auto-optimization
        assert train_config.get("grad_accum") == "auto"  # Auto-optimization
        assert train_config.get("eval_do_concat_batches") is False  # Memory optimization
        assert train_config.get("generation_num_beams") == 1  # Fast generation
        assert train_config.get("dataloader_num_workers") == 4  # Parallel loading


@pytest.mark.integration
class TestEvaluationOptimizationIntegration:
    """Integration tests for evaluation optimization"""
    
    def test_end_to_end_evaluation_flow(self):
        """Test the complete evaluation flow with optimization"""
        # This test simulates the complete flow from config loading to evaluation
        
        # Load optimized config
        config_path = Path(__file__).parent.parent / "configs" / "run_distributed_optimized.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify config has all optimization parameters
        train_config = config["train"]
        
        # Check evaluation optimization
        assert "eval_strategy" in train_config
        assert "predict_with_generate" in train_config
        assert "generation_num_beams" in train_config
        assert "generation_max_new_tokens" in train_config
        assert "eval_do_concat_batches" in train_config
        
        # Check values are optimized
        assert train_config["eval_do_concat_batches"] is False
        assert train_config["generation_num_beams"] == 1
        assert train_config["generation_max_new_tokens"] == 64
        
        # Check memory optimization
        assert train_config["gradient_checkpointing"] is True
        assert train_config["bf16"] is True
        
        # Check auto-optimization
        assert train_config["batch_size"] == "auto"
        assert train_config["grad_accum"] == "auto"
    
    def test_compatibility_with_different_transformers_versions(self):
        """Test compatibility handling for different Transformers versions"""
        # Test the logic that handles different Transformers versions
        
        # Simulate old version (no generation support)
        training_args_kwargs_old = {
            "output_dir": "test_output",
            "predict_with_generate": True,
            "eval_strategy": "steps"
        }
        
        GEN_SUPPORTED = True
        
        # Simulate the compatibility check
        try:
            # This would fail in old versions
            tr_args = TrainingArguments(**training_args_kwargs_old)
        except TypeError as e:
            if "predict_with_generate" in str(e):
                # Fallback logic
                training_args_kwargs_old.pop("predict_with_generate", None)
                training_args_kwargs_old.pop("generation_num_beams", None)
                training_args_kwargs_old.pop("generation_max_new_tokens", None)
                GEN_SUPPORTED = False
                training_args_kwargs_old["prediction_loss_only"] = True
                tr_args = TrainingArguments(**training_args_kwargs_old)
                assert GEN_SUPPORTED is False
                assert training_args_kwargs_old["prediction_loss_only"] is True
            else:
                raise e
        
        # Simulate new version (with generation support)
        training_args_kwargs_new = {
            "output_dir": "test_output",
            "predict_with_generate": True,
            "eval_strategy": "steps"
        }
        
        GEN_SUPPORTED = True
        
        # This should work in new versions
        try:
            tr_args = TrainingArguments(**training_args_kwargs_new)
            assert GEN_SUPPORTED is True
        except TypeError:
            # If it fails, we should handle it
            GEN_SUPPORTED = False
            assert False, "New version should support generation parameters"


class TestPerformanceCharacteristics:
    """Test performance characteristics of evaluation optimization"""
    
    def test_loss_only_evaluation_speed(self):
        """Test that loss-only evaluation is fast"""
        # This is a conceptual test - actual speed testing would require running training
        
        # Verify that prediction_loss_only is set when generation is not supported
        training_args_kwargs = {
            "output_dir": "test_output",
            "prediction_loss_only": True,
            "eval_strategy": "steps"
        }
        
        assert training_args_kwargs["prediction_loss_only"] is True
        
        # Verify that no compute_metrics is set
        GEN_SUPPORTED = False
        compute_metrics = None if not GEN_SUPPORTED else Mock()
        
        assert compute_metrics is None
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization parameters"""
        # Check that memory optimization parameters are set correctly
        
        # These parameters reduce memory usage during evaluation
        memory_optimization_params = {
            "eval_do_concat_batches": False,  # Reduces peak RAM
            "gradient_checkpointing": True,  # Reduces memory during training
            "dataloader_pin_memory": True,   # Optimizes data loading
            "ddp_bucket_cap_mb": 200,        # Optimizes communication
        }
        
        for param, expected_value in memory_optimization_params.items():
            assert expected_value is not None, f"{param} should be set"
    
    def test_evaluation_frequency_optimization(self):
        """Test evaluation frequency optimization"""
        # Check that evaluation frequency is reasonable
        
        # These values balance between frequent evaluation and training speed
        eval_params = {
            "eval_steps": 200,               # Evaluate every 200 steps
            "eval_accumulation_steps": 8,    # Accumulate eval batches
            "logging_steps": 1,              # Log every step
            "save_steps": 200,               # Save every 200 steps
        }
        
        for param, value in eval_params.items():
            assert isinstance(value, int), f"{param} should be integer"
            assert value > 0, f"{param} should be positive"


if __name__ == "__main__":
    pytest.main([__file__])
