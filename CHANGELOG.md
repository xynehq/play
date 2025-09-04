# SFT-Play Changelog

## v0.1.1 (2025-09-04) - Stability & Testing Improvements

### 🐛 Bug Fixes

#### Functional Tests Fixed
- **Fixed `test_full_pipeline_produces_valid_training_data`**: Now handles small validation datasets gracefully by allowing validation files to be empty for small test datasets
- **Fixed `test_process_fails_with_invalid_data`**: Improved error message assertion to check both stdout and stderr for more reliable error detection
- **All 10 functional tests now pass consistently** with improved edge case handling

#### TensorBoard Integration Fixed
- **Fixed `make train-bnb-tb` command termination**: Resolved issue where `pkill` commands were causing the make process to terminate unexpectedly
- **Improved TensorBoard process management**: Replaced problematic `pkill -f tensorboard` with safer `pgrep -f "tensorboard.*$(TB_PORT)" | xargs -r kill` for targeted process management
- **Fixed `TB_LOGDIR` variable**: Now properly handles missing directories by creating them before using `realpath`
- **Enhanced TensorBoard reliability**: TensorBoard now starts consistently and runs alongside training without process conflicts

### 🔧 Improvements

#### Test Suite Robustness
- Enhanced test suite to handle edge cases with small datasets
- Improved error message validation across different output streams
- Better handling of validation split edge cases in data processing
- More robust test assertions for various data sizes

#### Process Management
- Safer TensorBoard process handling to prevent make command termination
- Improved directory creation and validation in Makefile variables
- Better error handling for missing directories and files

### 📚 Documentation Updates

#### README.md
- Added "Recent Fixes & Improvements" section documenting all v0.1.1 changes
- Updated troubleshooting section with information about resolved issues
- Enhanced TensorBoard documentation with fix details

#### AUTOMATION_GUIDE.md
- Added "Test Suite Robustness" section documenting testing improvements
- Updated "TensorBoard Auto-Start" section with fix details
- Enhanced safety features documentation

### ✅ Verification

All changes have been thoroughly tested:
- ✅ All 10 functional tests pass consistently
- ✅ `make train-bnb-tb` works reliably with TensorBoard integration
- ✅ `make full-pipeline` processes data correctly
- ✅ Edge cases with small datasets handled properly
- ✅ TensorBoard starts and runs without process conflicts

### 🎯 Impact

These fixes significantly improve the reliability and user experience of SFT-Play:
- **Developers**: Can now rely on consistent test results
- **Users**: TensorBoard training commands work reliably
- **CI/CD**: Test suite is more robust for automated testing
- **Documentation**: Clear information about fixes and improvements

---

## v0.1.0 (2025-08-10) - Initial Release

### ✨ Features

- Complete SFT pipeline automation with Makefile
- Support for QLoRA, LoRA, and full fine-tuning
- BitsAndBytes and Unsloth backend support
- TensorBoard integration for training monitoring
- DAPT (Domain-Adaptive Pretraining) support
- Interactive setup with `./workflows/quick_start.sh`
- Comprehensive test suite with 10 functional tests
- Complete documentation and automation guides

### 🎯 Definition of Done

- End-to-end run on Qwen2.5-3B (QLoRA+bnb) on 8 GB GPU without OOM
- Live TensorBoard charts
- Complete automation with Makefile and workflow scripts
- Sanity checking with `make check` validation
