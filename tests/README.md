# SFT-Play Test Suite

This directory contains comprehensive tests for the SFT-Play framework, ensuring all components work correctly and reliably.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_commands.py         # Tests for all Makefile commands
├── test_configs.py          # Tests for configuration files
├── test_dapt_integration.py # DAPT integration tests (moved from root)
├── test_mixed_training.py   # Mixed training tests (moved from root)
├── test_simple_mixed.py     # Simple mixed training tests (moved from root)
└── README.md               # This file
```

## Test Categories

### 1. Command Tests (`test_commands.py`)
Tests all Makefile commands to ensure they:
- Execute without errors
- Have correct command structure
- Handle missing dependencies gracefully
- Validate input parameters

**Covered Commands:**
- `help` - Display available commands
- `setup-dirs` - Create directory structure
- `check` - Validate project setup
- `clean` - Remove generated files
- `train`, `train-bnb`, `train-unsloth` - Training commands
- `eval`, `eval-test`, `eval-val`, `eval-quick`, `eval-full` - Evaluation commands
- `infer`, `infer-batch`, `infer-interactive` - Inference commands
- `merge`, `merge-bf16`, `merge-test` - Model merging commands
- `process`, `style`, `render` - Data processing commands
- `dapt-docx`, `dapt-train` - DAPT-specific commands
- `tensorboard`, `tb-stop`, `tb-clean`, `tb-open` - TensorBoard commands
- `full-pipeline` - Complete workflow

### 2. Configuration Tests (`test_configs.py`)
Tests configuration files for:
- Valid YAML syntax
- Required field presence
- Value validation (ranges, types)
- Configuration inheritance
- Backend-specific settings

**Covered Configs:**
- `config_base.yaml` - Base configuration
- `run_bnb.yaml` - BitsAndBytes configuration
- `run_dapt.yaml` - DAPT configuration
- `run_unsloth.yaml` - Unsloth configuration

### 3. Integration Tests
Tests moved from root directory:
- `test_dapt_integration.py` - DAPT functionality integration
- `test_mixed_training.py` - Mixed training workflows
- `test_simple_mixed.py` - Simple mixed training validation

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-cov

# Ensure you're in the play directory
cd play
```

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_commands.py

# Run specific test class
pytest tests/test_commands.py::TestMakefileCommands

# Run specific test method
pytest tests/test_commands.py::TestMakefileCommands::test_help_command
```

### Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU-dependent tests
pytest -m "not gpu"

# Skip network-dependent tests
pytest -m "not network"
```

### Coverage Reports
```bash
# Run tests with coverage
pytest --cov=scripts --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Fixtures

The `conftest.py` file provides shared fixtures:

- `temp_dir` - Temporary directory for test files
- `sample_config` - Sample configuration for testing
- `sample_chat_data` - Sample chat format data
- `sample_cpt_data` - Sample CPT format data
- `mock_model_dir` - Mock model directory structure
- `test_config_file` - Temporary config file
- `test_data_files` - Temporary data files

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure
```python
import pytest
from pathlib import Path

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality works."""
        # Arrange
        input_data = "test input"
        
        # Act
        result = process_input(input_data)
        
        # Assert
        assert result == "expected output"
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            process_input(None)
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance (marked as slow)."""
        # Performance test code
        pass
```

### Test Markers
Use markers to categorize tests:
```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test."""
    pass

@pytest.mark.integration
def test_integration_workflow():
    """Integration test."""
    pass

@pytest.mark.gpu
def test_gpu_functionality():
    """Test requiring GPU."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Slow test."""
    pass
```

## Continuous Integration

Tests are designed to run in CI environments:

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov=scripts
```

## Test Data

### Minimal Test Data
Tests use minimal data sets to ensure fast execution:
- 2-3 chat examples for SFT testing
- 2-3 text documents for CPT testing
- Small model configurations

### Mock Objects
Heavy dependencies are mocked:
- Model downloads
- GPU operations
- Network requests
- File I/O operations

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH includes current directory
   export PYTHONPATH=.
   pytest
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install pytest pytest-cov
   ```

3. **GPU Tests Failing**
   ```bash
   # Skip GPU tests if no GPU available
   pytest -m "not gpu"
   ```

4. **Slow Tests**
   ```bash
   # Skip slow tests for quick validation
   pytest -m "not slow"
   ```

### Debug Mode
```bash
# Run with debug output
pytest -s -vv

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Add appropriate markers** for categorization
3. **Update this README** if adding new test categories
4. **Ensure tests pass** in CI environment
5. **Add fixtures** to `conftest.py` for reusable test data

## Test Coverage Goals

- **Commands**: 100% of Makefile commands tested
- **Configurations**: All config files validated
- **Core Scripts**: Key functionality covered
- **Error Handling**: Edge cases and error conditions
- **Integration**: End-to-end workflows validated

Current coverage can be viewed by running:
```bash
pytest --cov=scripts --cov-report=term-missing
