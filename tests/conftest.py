"""
Pytest configuration and shared fixtures for SFT-Play tests.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import yaml
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "seed": 42,
        "run_name": "test-run",
        "mode": "sft",
        "train": {
            "epochs": 1,
            "learning_rate": 2e-4,
            "output_dir": "outputs/test"
        },
        "model": {
            "name": "microsoft/DialoGPT-small",
            "type": "causal"
        },
        "data": {
            "format": "chat",
            "template_path": "chat_templates/default.jinja"
        }
    }


@pytest.fixture
def sample_chat_data():
    """Sample chat data for testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain neural networks."},
                {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks."}
            ]
        }
    ]


@pytest.fixture
def sample_cpt_data():
    """Sample CPT data for testing."""
    return [
        {"text": "This is a sample document for continued pretraining. It contains domain-specific knowledge."},
        {"text": "Another document with technical content that would be useful for domain adaptation."}
    ]


@pytest.fixture
def mock_model_dir(temp_dir):
    """Create a mock model directory structure."""
    model_dir = temp_dir / "models" / "test-model"
    model_dir.mkdir(parents=True)
    
    # Create mock model files
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "gpt2",
        "vocab_size": 50257,
        "n_positions": 1024
    }))
    
    (model_dir / "tokenizer.json").write_text(json.dumps({
        "version": "1.0",
        "truncation": None,
        "padding": None
    }))
    
    return model_dir


@pytest.fixture
def test_config_file(temp_dir, sample_config):
    """Create a test configuration file."""
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def test_data_files(temp_dir, sample_chat_data):
    """Create test data files."""
    data_dir = temp_dir / "data" / "processed"
    data_dir.mkdir(parents=True)
    
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    # Write training data
    with open(train_file, 'w') as f:
        for item in sample_chat_data:
            f.write(json.dumps(item) + '\n')
    
    # Write validation data (same as training for testing)
    with open(val_file, 'w') as f:
        for item in sample_chat_data[:1]:  # Just one sample for validation
            f.write(json.dumps(item) + '\n')
    
    return {"train": train_file, "val": val_file}
