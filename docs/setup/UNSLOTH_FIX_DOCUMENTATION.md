# Unsloth/XFormers Fix Documentation

## Overview

This document explains the comprehensive fix implemented for the Unsloth/XFormers conflict issue in the training script. The previous implementation was just a placeholder that always fell back to BitsAndBytes. Now we have a robust solution that properly supports Unsloth with graceful fallback.

## What Was Fixed

### 1. **Environment Variable Setup**
- **XFORMERS_DISABLED=1**: Completely disables XFormers to prevent conflicts
- **UNSLOTH_DISABLE_FAST_ATTENTION=1**: Disables Unsloth's fast attention that conflicts with XFormers
- **UNSLOTH_FORCE_SDPA=1**: Forces Unsloth to use SDPA (Scaled Dot Product Attention)
- **CUDA_LAUNCH_BLOCKING=1**: Better error reporting for debugging
- **TORCH_USE_CUDA_DSA=1**: Enable device-side assertions for debugging

### 2. **PyTorch Backend Configuration**
```python
torch.backends.cuda.enable_flash_sdp(False)      # Disable Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Keep memory efficient attention
torch.backends.cuda.enable_math_sdp(True)        # Keep math attention as fallback
```

### 3. **Proper Unsloth Integration**
- **Import Testing**: Tests Unsloth compatibility before attempting to use it
- **Model Loading**: Uses `FastLanguageModel.from_pretrained()` with proper parameters
- **LoRA Configuration**: Uses Unsloth's optimized `get_peft_model()` implementation
- **Attention Implementation**: Forces SDPA attention to avoid XFormers conflicts

### 4. **Robust Fallback Logic**
- **Import Errors**: Falls back to BitsAndBytes if Unsloth is not installed
- **Runtime Errors**: Falls back to BitsAndBytes if Unsloth fails to initialize
- **Model Loading Errors**: Falls back to BitsAndBytes if Unsloth model loading fails
- **LoRA Errors**: Falls back to standard PEFT if Unsloth LoRA fails

## How It Works

### Backend Detection Flow
```
1. Check if backend="unsloth" in config
2. Try to import Unsloth modules
3. Test basic Unsloth functionality
4. If successful: use_bnb = False, unsloth_available = True
5. If failed: use_bnb = True, fallback to BitsAndBytes
```

### Model Loading Flow
```
For QLoRA mode:
├── If use_bnb = True:
│   └── Use BitsAndBytesConfig + AutoModelForCausalLM
└── If use_bnb = False (Unsloth available):
    ├── Try FastLanguageModel.from_pretrained()
    ├── Force SDPA attention implementation
    └── If failed: fallback to BitsAndBytes
```

### LoRA Application Flow
```
If LoRA/QLoRA mode:
├── If unsloth_available and not use_bnb:
│   ├── Try FastLanguageModel.get_peft_model()
│   ├── Use Unsloth-specific optimizations
│   └── If failed: fallback to standard PEFT
└── Else:
    └── Use standard PEFT LoraConfig + get_peft_model()
```

## Key Features

### 1. **Unsloth-Specific Optimizations**
- **Gradient Checkpointing**: `use_gradient_checkpointing="unsloth"`
- **Memory Efficiency**: Optimized for 4-bit quantization
- **Speed**: 2x faster training compared to standard implementations

### 2. **XFormers Conflict Resolution**
- **Complete Disabling**: XFormers is completely disabled via environment variables
- **Alternative Attention**: Uses SDPA (Scaled Dot Product Attention) instead
- **Backend Forcing**: Explicitly sets attention implementation in model config

### 3. **Comprehensive Error Handling**
- **Detailed Logging**: Clear messages about what's happening and why
- **Graceful Degradation**: Always falls back to working BitsAndBytes implementation
- **User Guidance**: Helpful error messages with installation instructions

## Usage

### 1. **Training with Unsloth**
```bash
# Use the Makefile command (recommended)
make train-unsloth

# Or run directly
XFORMERS_DISABLED=1 UNSLOTH_DISABLE_FAST_ATTENTION=1 python scripts/train.py --config configs/run_unsloth.yaml
```

### 2. **Configuration**
The `configs/run_unsloth.yaml` file is properly configured:
```yaml
tuning:
  backend: unsloth     # Use Unsloth backend
  lora:
    alpha: 32          # Unsloth prefers alpha = r

train:
  bf16: true           # Unsloth works better with bfloat16
  fp16: false
  eval_steps: 1        # More frequent evaluation
```

### 3. **Testing the Fix**
```bash
# Run the comprehensive test suite
python test_unsloth_fix.py
```

## Expected Behavior

### When Unsloth is Available and Working
```
[train] Attempting to use Unsloth backend...
[train] Testing Unsloth compatibility...
[train] ✅ Unsloth import successful
[train] ✅ Unsloth backend will be used
[train] ✅ Unsloth model loaded successfully
[train] ✅ Forced model to use SDPA attention implementation
[train] Applying Unsloth LoRA configuration...
[train] ✅ Unsloth LoRA applied successfully
```

### When Unsloth is Not Available
```
[train] Attempting to use Unsloth backend...
[train] ❌ Unsloth not available (ImportError): No module named 'unsloth'
[train] 💡 Install Unsloth with: pip install unsloth
[train] Falling back to BitsAndBytes backend
```

### When Unsloth Has XFormers Conflicts
```
[train] Attempting to use Unsloth backend...
[train] ❌ Unsloth failed to initialize: XFormers conflict detected
[train] This is likely due to XFormers conflicts or CUDA compatibility issues
[train] Falling back to BitsAndBytes backend
```

## Performance Benefits

### With Unsloth (when working):
- **2x faster training** compared to standard implementations
- **Lower memory usage** due to optimized 4-bit quantization
- **Better gradient checkpointing** with Unsloth-specific optimizations

### Fallback to BitsAndBytes:
- **Reliable training** with proven BitsAndBytes implementation
- **Same accuracy** as Unsloth but potentially slower
- **Broader compatibility** across different CUDA versions

## Troubleshooting

### 1. **Unsloth Installation Issues**
```bash
# Install Unsloth
pip install unsloth

# For specific CUDA versions
pip install unsloth[cu118]  # For CUDA 11.8
pip install unsloth[cu121]  # For CUDA 12.1
```

### 2. **XFormers Conflicts**
The fix automatically handles XFormers conflicts, but if you still see issues:
```bash
# Uninstall XFormers completely
pip uninstall xformers

# Or force disable in environment
export XFORMERS_DISABLED=1
```

### 3. **CUDA Compatibility**
```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Ensure compatibility between CUDA, PyTorch, and Unsloth
```

## Monitoring and Verification

### 1. **Check Backend Usage**
Look for these log messages to confirm which backend is being used:
- `✅ Unsloth backend will be used` - Unsloth is active
- `Falling back to BitsAndBytes backend` - Using BitsAndBytes fallback

### 2. **Monitor Training Speed**
- **Unsloth**: Should show ~2x speed improvement
- **BitsAndBytes**: Standard speed but reliable

### 3. **Check Memory Usage**
- **Unsloth**: More memory efficient
- **BitsAndBytes**: Standard memory usage

## Files Modified

1. **`scripts/train.py`**: Main training script with Unsloth integration
2. **`configs/run_unsloth.yaml`**: Unsloth-specific configuration
3. **`Makefile`**: Commands for Unsloth training
4. **`test_unsloth_fix.py`**: Comprehensive test suite

## Conclusion

This fix provides a robust solution to the Unsloth/XFormers conflict while maintaining backward compatibility. The implementation:

- ✅ **Properly supports Unsloth** when available and compatible
- ✅ **Gracefully falls back** to BitsAndBytes when needed
- ✅ **Resolves XFormers conflicts** through environment variables and backend configuration
- ✅ **Provides clear feedback** about what's happening and why
- ✅ **Maintains performance** with Unsloth optimizations when possible

The training script now actually uses Unsloth instead of just being a placeholder, while ensuring that training always works regardless of the environment setup.
