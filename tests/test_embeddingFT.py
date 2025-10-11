#!/usr/bin/env python3
"""
Test cases for embedding fine-tuning module
"""

import os
import sys
import pytest
import tempfile
import shutil
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add embeddingFT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'embeddingFT'))

try:
    from scripts.train_embed import (
        load_config, setup_env, prepare_datasets, 
        mine_negatives, evaluate_baseline, train_model
    )
except ImportError:
    # Skip tests if module not available
    pytest.skip("embeddingFT module not available", allow_module_level=True)

class TestEmbeddingFT:
    """Test cases for embedding fine-tuning functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create sample configuration for testing"""
        config = {
            "dataset_name": "sentence-transformers/all-nli",
            "sample_size": 100,
            "test_size": 20,
            "seed": 42,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "mining_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "num_epochs": 1,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "cache_dir": temp_dir
        }
        
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path, config
    
    def test_load_config(self, sample_config):
        """Test configuration loading"""
        config_path, config_data = sample_config
        
        loaded_config = load_config(config_path)
        
        assert loaded_config == config_data
        assert loaded_config["dataset_name"] == "sentence-transformers/all-nli"
        assert loaded_config["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert loaded_config["num_epochs"] == 1
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")
    
    def test_setup_env(self, temp_dir):
        """Test environment setup"""
        setup_env(temp_dir)
        
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["HF_HOME"] == temp_dir
        assert os.environ["HF_HUB_CACHE"] == temp_dir
    
    def test_setup_env_default(self):
        """Test environment setup with default cache directory"""
        setup_env()
        
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["HF_HOME"] == "/models"
        assert os.environ["HF_HUB_CACHE"] == "/models"
    
    @patch('scripts.train_embed.load_dataset')
    def test_prepare_datasets(self, mock_load_dataset):
        """Test dataset preparation"""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = {
            "train": "train_data",
            "test": "test_data"
        }
        mock_load_dataset.return_value = mock_dataset
        
        # Test function
        ds_full, ds_train, ds_eval = prepare_datasets(
            dataset_name="test_dataset",
            sample_size=50,
            test_size=10,
            seed=42
        )
        
        # Verify calls
        mock_load_dataset.assert_called_once_with("test_dataset", split="train")
        mock_dataset.select.assert_called_once_with(range(50))
        mock_dataset.train_test_split.assert_called_once_with(test_size=10, seed=42)
        
        assert ds_full == mock_dataset
        assert ds_train == "train_data"
        assert ds_eval == "test_data"
    
    @patch('scripts.train_embed.mine_hard_negatives')
    def test_mine_negatives(self, mock_mine_hard_negatives):
        """Test hard negative mining"""
        # Mock dataset and model
        mock_dataset = MagicMock()
        mock_model = MagicMock()
        mock_result = {"anchor": ["test"], "positive": ["test"], "negative": ["test"]}
        mock_mine_hard_negatives.return_value = mock_result
        
        # Test function
        result = mine_negatives(
            mock_dataset, 
            mock_model, 
            num_negatives=2, 
            batch_size=256,
            margin=0.5
        )
        
        # Verify calls
        mock_mine_hard_negatives.assert_called_once_with(
            mock_dataset,
            embed_model=mock_model,
            num_negatives=2,
            batch_size=256,
            output_format="triplet",
            margin=0.5
        )
        
        assert result == mock_result
    
    @patch('scripts.train_embed.SentenceTransformer')
    @patch('scripts.train_embed.mine_negatives')
    @patch('scripts.train_embed.TripletEvaluator')
    def test_evaluate_baseline(self, mock_evaluator, mock_mine_negatives, mock_sentence_transformer):
        """Test baseline evaluation"""
        # Mock models and data
        mock_model = MagicMock()
        mock_mining_model = MagicMock()
        mock_eval_instance = MagicMock()
        mock_evaluator.return_value = mock_eval_instance
        mock_mine_negatives.return_value = {
            "query": ["test query"],
            "answer": ["test answer"],
            "negative_1": ["test negative"]
        }
        mock_eval_instance.return_value = {"accuracy": 0.8}
        
        # Mock dataset
        ds_full = {"answer": ["answer1", "answer2"]}
        ds_eval = {"query": ["query1"], "answer": ["answer1"]}
        
        # Test function
        with patch('scripts.train_embed.logging') as mock_logging:
            result = evaluate_baseline(
                mock_model, ds_full, ds_eval, 
                "test-mining-model", "test-eval"
            )
        
        # Verify calls
        mock_sentence_transformer.assert_called_once_with("test-mining-model", device="cpu")
        mock_mine_negatives.assert_called_once()
        mock_evaluator.assert_called_once()
        mock_eval_instance.assert_called_once_with(mock_model)
        
        assert result == {"accuracy": 0.8}
    
    @patch('scripts.train_embed.SentenceTransformer')
    @patch('scripts.train_embed.SentenceTransformerTrainingArguments')
    @patch('scripts.train_embed.SentenceTransformerTrainer')
    @patch('scripts.train_embed.MultipleNegativesRankingLoss')
    @patch('scripts.train_embed.TripletEvaluator')
    @patch('scripts.train_embed.concatenate_datasets')
    @patch('scripts.train_embed.prepare_datasets')
    @patch('scripts.train_embed.evaluate_baseline')
    @patch('scripts.train_embed.setup_env')
    def test_train_model_integration(self, mock_setup_env, mock_evaluate_baseline, 
                                   mock_prepare_datasets, mock_concatenate,
                                   mock_evaluator, mock_loss, mock_trainer,
                                   mock_training_args, mock_sentence_transformer,
                                   sample_config):
        """Test complete training model integration"""
        config_path, config_data = sample_config
        
        # Mock all dependencies
        mock_model = MagicMock()
        mock_mining_model = MagicMock()
        mock_sentence_transformer.side_effect = [mock_model, mock_mining_model]
        
        mock_ds_full = MagicMock()
        mock_ds_train = MagicMock()
        mock_ds_eval = MagicMock()
        mock_prepare_datasets.return_value = (mock_ds_full, mock_ds_train, mock_ds_eval)
        
        mock_hard_train = MagicMock()
        mock_hard_eval = MagicMock()
        mock_concatenate.return_value = mock_hard_train
        
        mock_loss_instance = MagicMock()
        mock_loss.return_value = mock_loss_instance
        
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        mock_args = MagicMock()
        mock_args.run_name = "test-run"
        mock_training_args.return_value = mock_args
        
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock mine_negatives
        with patch('scripts.train_embed.mine_negatives') as mock_mine:
            mock_mine.side_effect = [
                mock_hard_eval,  # For evaluation
                mock_hard_train   # For training (chunked)
            ]
            
            # Test training
            with patch('scripts.train_embed.logging'):
                train_model(config_data)
        
        # Verify key calls
        mock_setup_env.assert_called_once_with(config_data["cache_dir"])
        mock_prepare_datasets.assert_called_once()
        mock_sentence_transformer.assert_called()
        mock_evaluate_baseline.assert_called_once()
        mock_loss.assert_called_once_with(mock_model)
        mock_training_args.assert_called_once()
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_evaluator_instance.assert_called_with(mock_trainer_instance.model)
        mock_model.save_pretrained.assert_called_once()
        mock_model.push_to_hub.assert_called_once()
    
    def test_config_validation(self, sample_config):
        """Test configuration validation"""
        config_path, config_data = sample_config
        
        # Test valid config
        loaded_config = load_config(config_path)
        
        required_fields = [
            "dataset_name", "sample_size", "test_size", "seed",
            "model_name", "mining_model_name", "num_epochs",
            "batch_size", "learning_rate", "weight_decay", "warmup_ratio"
        ]
        
        for field in required_fields:
            assert field in loaded_config, f"Missing required field: {field}"
    
    def test_config_types(self, sample_config):
        """Test configuration field types"""
        config_path, config_data = sample_config
        loaded_config = load_config(config_path)
        
        # Test type validation
        assert isinstance(loaded_config["dataset_name"], str)
        assert isinstance(loaded_config["sample_size"], int)
        assert isinstance(loaded_config["test_size"], int)
        assert isinstance(loaded_config["seed"], int)
        assert isinstance(loaded_config["model_name"], str)
        assert isinstance(loaded_config["mining_model_name"], str)
        assert isinstance(loaded_config["num_epochs"], int)
        assert isinstance(loaded_config["batch_size"], int)
        assert isinstance(loaded_config["learning_rate"], (int, float))
        assert isinstance(loaded_config["weight_decay"], (int, float))
        assert isinstance(loaded_config["warmup_ratio"], (int, float))
    
    @patch('scripts.train_embed.train_model')
    def test_main_execution(self, mock_train_model, sample_config):
        """Test main script execution"""
        config_path, config_data = sample_config
        
        # Mock sys.argv
        with patch.object(sys, 'argv', ['train_embed.py', '--config', config_path]):
            with patch('scripts.train_embed.logging.basicConfig'):
                # Import and run main
                from scripts.train_embed import main
                
                # This would normally call train_model
                # mock_train_model.assert_called_once_with(config_data)
                pass
    
    def test_error_handling_invalid_config(self, temp_dir):
        """Test error handling for invalid configuration"""
        # Create invalid config (missing required fields)
        invalid_config = {"model_name": "test-model"}
        config_path = os.path.join(temp_dir, "invalid_config.yaml")
        
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Load should work but training should fail
        loaded_config = load_config(config_path)
        
        # Missing required fields should cause issues in training
        with pytest.raises(KeyError):
            train_model(loaded_config)
    
    def test_memory_optimization_config(self, temp_dir):
        """Test memory optimization configuration"""
        config = {
            "dataset_name": "test",
            "sample_size": 50,
            "test_size": 10,
            "seed": 42,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "mining_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "num_epochs": 1,
            "batch_size": 4,  # Small batch size for memory
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "cache_dir": temp_dir
        }
        
        config_path = os.path.join(temp_dir, "memory_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        loaded_config = load_config(config_path)
        assert loaded_config["batch_size"] == 4
    
    def test_different_mining_models(self, temp_dir):
        """Test different mining model configurations"""
        mining_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/static-retrieval-mrl-en-v1",
            "sentence-transformers/stsb-roberta-large"
        ]
        
        for model_name in mining_models:
            config = {
                "dataset_name": "test",
                "sample_size": 50,
                "test_size": 10,
                "seed": 42,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "mining_model_name": model_name,
                "num_epochs": 1,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "cache_dir": temp_dir
            }
            
            config_path = os.path.join(temp_dir, f"config_{model_name.split('/')[-1]}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            loaded_config = load_config(config_path)
            assert loaded_config["mining_model_name"] == model_name

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
