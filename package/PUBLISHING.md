# Publishing Guide for SFT-Play

## Package Built Successfully ✓

The package has been built and is ready for publishing to PyPI.

**Build artifacts:**
- `dist/sft_play-2.0.0-py3-none-any.whl` (wheel)
- `dist/sft_play-2.0.0.tar.gz` (source distribution)

---

## Pre-Publishing Checklist

Before publishing to PyPI, ensure:

- [ ] Version number is correct in `pyproject.toml` (currently: 2.0.0)
- [ ] README.md is complete and accurate
- [ ] LICENSE file is included
- [ ] All entry points work correctly
- [ ] Package builds without errors (`python3 -m build`)
- [ ] Test installation locally (see below)
- [ ] GitHub repository URLs are correct in pyproject.toml
- [ ] CHANGELOG.md is updated

---

## Testing Local Installation

### 1. Install Locally
```bash
# From the package/ directory
cd package/

# Install the built wheel
pip install dist/sft_play-2.0.0-py3-none-any.whl

# Or install in editable mode for development
pip install -e .
```

### 2. Test CLI Commands
```bash
# Check version
sft-play --version

# Test help
sft-play --help

# Test subcommands
sft-train --help
sft-eval --help
sft-infer --help
sft-process --help
```

### 3. Test Import
```bash
python3 -c "import sft_play; print(sft_play.__version__)"
```

---

## Publishing to PyPI

### Option 1: PyPI (Production)

#### First Time Setup
```bash
# Install twine if not already installed
pip install twine

# Create PyPI account at https://pypi.org/account/register/
# Generate API token at https://pypi.org/manage/account/token/
```

#### Upload to PyPI
```bash
cd package/

# Build fresh artifacts
rm -rf dist/
python3 -m build

# Check the package (validate metadata)
twine check dist/*

# Upload to PyPI (will prompt for credentials or use token)
twine upload dist/*

# Or use API token
twine upload -u __token__ -p <your-pypi-token> dist/*
```

### Option 2: TestPyPI (Testing)

Test your package on TestPyPI first:

```bash
# Create TestPyPI account at https://test.pypi.org/account/register/

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ sft-play
```

---

## Publishing with GitHub Actions (Recommended)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          cd package/
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: |
          cd package/
          python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          cd package/
          twine upload dist/*
```

**Setup:**
1. Create PyPI API token
2. Add token as GitHub secret: `PYPI_API_TOKEN`
3. Create GitHub release to trigger workflow

---

## Version Management

### Updating Version

Update version in these files:
1. `package/pyproject.toml` → `version = "x.y.z"`
2. `package/sft_play/version.py` → `__version__ = "x.y.z"`
3. `../VERSION` (main repo) if needed

### Semantic Versioning

- **Major (x.0.0)**: Breaking changes
- **Minor (0.x.0)**: New features, backward compatible
- **Patch (0.0.x)**: Bug fixes

---

## Post-Publishing

After successful publication:

1. **Verify on PyPI**: https://pypi.org/project/sft-play/
2. **Test installation**: `pip install sft-play`
3. **Update CHANGELOG.md**
4. **Create GitHub release** with notes
5. **Update main README** with installation instructions

---

## Distribution Methods

### pip (PyPI)
```bash
pip install sft-play              # Minimal
pip install sft-play[standard]    # Standard
pip install sft-play[full]        # Full
```

### uv (Fast installer)
```bash
uv pip install sft-play[standard]
```

### From GitHub (Development)
```bash
pip install git+https://github.com/xynehq/sft-play.git#subdirectory=package
```

### Local Development
```bash
cd package/
pip install -e .                  # Editable install
pip install -e ".[dev]"           # With dev dependencies
```

---

## Troubleshooting

### Build Fails
- Check `pyproject.toml` syntax
- Ensure LICENSE file exists in package/
- Verify all required files are present

### Upload Fails
- Verify PyPI credentials
- Check if version already exists (can't re-upload same version)
- Ensure package name is available

### Import Fails After Install
- Check package structure
- Verify `__init__.py` exists
- Test with `python -c "import sft_play"`

---

## Current Package Status

**Version**: 2.0.0
**Build Status**: ✓ Successfully built
**Test Status**: Pending local testing
**Publish Status**: Not yet published

**Next Steps:**
1. Test local installation
2. Update GitHub URLs in pyproject.toml (currently placeholder)
3. Test on TestPyPI
4. Publish to PyPI

---

## Resources

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- Twine Docs: https://twine.readthedocs.io/
- Packaging Guide: https://packaging.python.org/

---

*Last Updated: 2025-10-09*
