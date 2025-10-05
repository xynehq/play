# Contributing to Xyne-LLM-Play

Thank you for your interest in contributing to Xyne-LLM-Play! This guide will help you get started with contributing to this universal LLM training suite.

## 🎯 Project Overview

Xyne-LLM-Play is a universal training suite for SFT, DAPT, multi-GPU, embeddings finetuning, and RL. We're building a comprehensive toolkit that makes LLM training accessible from hobbyist RTX 4060 setups to enterprise H200 clusters.

## 🤝 How to Contribute

### Reporting Issues

#### Bug Reports
- Use the [GitHub Issues](https://github.com/xynehq/xyne-llm-play/issues) page
- Provide detailed information:
  - Hardware setup (GPU, VRAM, OS)
  - Configuration used
  - Error messages and logs
  - Steps to reproduce
  - Expected vs actual behavior

#### Feature Requests
- Open an issue with the "enhancement" label
- Describe the use case and motivation
- Suggest implementation approach if you have ideas
- Consider if it fits the project scope

### Development Contributions

#### Areas Where We Need Help

1. **Core Training Features**
   - New training algorithms and optimizers
   - Additional model architectures support
   - Enhanced memory optimization techniques
   - Improved evaluation metrics

2. **Multi-GPU & Distributed Training**
   - Advanced DeepSpeed configurations
   - Better load balancing strategies
   - Network optimization for multi-node training
   - Fault tolerance and recovery

3. **Domain Adaptation (DAPT)**
   - New document format parsers
   - Mixed training strategies
   - Domain-specific evaluation metrics
   - Curriculum learning approaches

4. **User Experience**
   - CLI improvements and new commands
   - Configuration validation and helpers
   - Better error messages and debugging
   - Progress visualization

5. **Documentation & Examples**
   - New examples for different use cases
   - Tutorial content and guides
   - API documentation improvements
   - Translation to other languages

6. **Testing & Quality**
   - Unit tests for core functionality
   - Integration tests for workflows
   - Performance benchmarks
   - CI/CD improvements

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- CUDA-compatible GPU (recommended for testing)
- GitHub account

### Setup Steps

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/xyne-llm-play.git
   cd xyne-llm-play
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   make install
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   # Install pre-commit hooks
   pre-commit install
   ```

4. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   make check
   pytest
   ```

## 📝 Development Workflow

### 1. Create a Branch
```bash
# Create a feature branch from main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-description
```

### 2. Make Changes

#### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to new functions and classes
- Keep lines under 88 characters (Black default)

#### Code Organization
- Place new training scripts in `scripts/`
- Add new configurations in `configs/`
- Put examples in `examples/`
- Update documentation as needed

#### Testing
- Write tests for new functionality
- Ensure existing tests still pass
- Test on different hardware configurations if possible

### 3. Commit Changes
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new training algorithm for DAPT

- Implement curriculum learning for domain adaptation
- Add configuration options for learning rate scheduling
- Include tests and documentation
- Fixes #123"
```

#### Commit Message Format
Use conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

### 4. Test Your Changes
```bash
# Run full test suite
pytest

# Run specific tests
pytest tests/test_new_feature.py

# Test training functionality
make train CONFIG=configs/test_config.yaml
```

### 5. Submit Pull Request
```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## 🧪 Testing Guidelines

### Test Categories

#### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Fast execution (< 1 second per test)
- Located in `tests/test_*.py`

#### Integration Tests
- Test complete workflows
- Use real data when possible
- May be slower and require GPU
- Mark with `@pytest.mark.integration`

#### Functional Tests
- Test CLI commands and Makefile targets
- Verify end-to-end functionality
- Located in `tests/test_commands.py`

### Writing Tests

```python
import pytest
from pathlib import Path

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self, sample_config):
        """Test basic functionality works."""
        # Arrange
        config = sample_config
        
        # Act
        result = process_config(config)
        
        # Assert
        assert result is not None
        assert result.success is True
    
    @pytest.mark.gpu
    def test_gpu_functionality(self):
        """Test GPU-specific functionality."""
        # GPU test code
        pass
    
    @pytest.mark.slow
    def test_performance(self):
        """Performance test (marked as slow)."""
        # Performance test code
        pass
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scripts --cov-report=html

# Run specific categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## 📚 Documentation Guidelines

### Types of Documentation

#### Code Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include parameter types and return values
- Add examples for complex functions

```python
def train_model(config: Dict[str, Any]) -> TrainingResult:
    """Train a language model with the given configuration.
    
    Args:
        config: Training configuration containing model, data,
            and training parameters.
    
    Returns:
        TrainingResult: Object containing training metrics and
            model artifacts.
    
    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If training fails.
    
    Example:
        >>> config = load_config("configs/run_bnb.yaml")
        >>> result = train_model(config)
        >>> print(result.final_loss)
        0.85
    """
```

#### User Documentation
- Update README.md for user-facing changes
- Add examples to `examples/` directory
- Update relevant documentation in `docs/`
- Include troubleshooting information

#### API Documentation
- Document new configuration options
- Update configuration guides
- Add command-line help text
- Include migration guides for breaking changes

### Documentation Style
- Use clear, concise language
- Include code examples
- Add screenshots where helpful
- Maintain consistent formatting

## 🔧 Configuration Guidelines

### Adding New Configurations

1. **Create Configuration File**
   ```yaml
   # configs/my_new_config.yaml
   model:
     name: "Qwen/Qwen2.5-3B-Instruct"
     max_seq_len: 512
   
   tuning:
     mode: "qlora"
     backend: "bnb"
   
   train:
     epochs: 3
     learning_rate: 2e-4
   ```

2. **Add Validation**
   ```python
   # tests/test_configs.py
   def test_my_new_config(self):
       config = load_config("configs/my_new_config.yaml")
       validate_config(config)
   ```

3. **Update Documentation**
   - Add to configuration guide
   - Include in examples
   - Document new parameters

### Configuration Best Practices
- Use descriptive names
- Include comments for complex settings
- Provide reasonable defaults
- Test on different hardware configurations

## 🚀 Release Process

### Version Management
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `pyproject.toml`
- Update CHANGELOG.md
- Tag releases on GitHub

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Release notes prepared
- [ ] GitHub release created

## 🏷️ Labeling Issues and PRs

### Issue Labels
- `bug`: Bug reports
- `enhancement`: Feature requests
- `documentation`: Documentation issues
- `good first issue`: Good for newcomers
- `help wanted`: Community help needed
- `priority/high`: High priority issues

### PR Labels
- `ready for review`: Ready for maintainer review
- `work in progress`: Still being developed
- `needs changes`: Requires updates
- `merged`: Successfully merged

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain professional communication

### Getting Help
- Check existing documentation first
- Search issues for similar problems
- Ask questions in discussions
- Join community channels (if available)

### Recognition
- Contributors are recognized in README.md
- Significant contributions may be invited as maintainers
- Community highlights in release notes

## 🎯 Specific Contribution Areas

### High Priority Areas

1. **Multi-GPU Optimization**
   - Better memory management
   - Improved load balancing
   - Network optimization

2. **Model Support**
   - Additional model architectures
   - Better quantization support
   - Custom model integration

3. **User Experience**
   - Better error messages
   - Progress indicators
   - Configuration wizards

### Good First Issues
- Documentation improvements
- Test coverage additions
- Small feature enhancements
- Bug fixes with clear reproduction steps

### Advanced Contributions
- Core algorithm improvements
- Distributed training research
- Performance optimization
- Integration with external tools

## 📧 Getting in Touch

- **Issues**: [GitHub Issues](https://github.com/your-username/xyne-llm-play/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/xyne-llm-play/discussions)
- **Email**: [maintainer@example.com]

## 🙏 Acknowledgments

Thank you to all contributors who have helped make Xyne-LLM-Play better! Your contributions, whether code, documentation, bug reports, or feature suggestions, are valuable and appreciated.

---

**Ready to contribute? Start with [good first issues](https://github.com/your-username/xyne-llm-play/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or join our [discussions](https://github.com/your-username/xyne-llm-play/discussions)! 🚀**
