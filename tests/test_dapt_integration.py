#!/usr/bin/env python3
"""
Test script for DAPT integration in SFT-Play.
Validates that all components work together without running full training.
"""

import os
import sys
import yaml
import json
from pathlib import Path

def test_config_loading():
    """Test that DAPT config loads correctly."""
    print("🔍 Testing DAPT config loading...")
    
    config_path = "configs/run_dapt.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required DAPT fields
        required_fields = ['task_mode', 'datasets', 'block_size', 'pack_factor']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        if config['task_mode'] != 'cpt_mixed':
            print(f"❌ Expected task_mode 'cpt_mixed', got '{config['task_mode']}'")
            return False
        
        print("✅ DAPT config loads correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_anchor_data():
    """Test that anchor instruction data exists and is valid."""
    print("🔍 Testing anchor instruction data...")
    
    anchor_path = "data/processed/anchor_instr.jsonl"
    if not os.path.exists(anchor_path):
        print(f"❌ Anchor file not found: {anchor_path}")
        return False
    
    try:
        with open(anchor_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            print("❌ Anchor file is empty")
            return False
        
        # Validate first line
        first_line = json.loads(lines[0])
        if 'messages' not in first_line:
            print("❌ Invalid anchor format: missing 'messages' field")
            return False
        
        print(f"✅ Anchor data valid ({len(lines)} examples)")
        return True
        
    except Exception as e:
        print(f"❌ Error reading anchor data: {e}")
        return False

def test_script_imports():
    """Test that all DAPT scripts can be imported."""
    print("🔍 Testing DAPT script imports...")
    
    scripts_to_test = [
        'scripts.ingest_docx',
        'scripts.datasets_cpt', 
        'scripts.collators_cpt'
    ]
    
    for script in scripts_to_test:
        try:
            __import__(script)
            print(f"✅ {script} imports successfully")
        except Exception as e:
            print(f"❌ Error importing {script}: {e}")
            return False
    
    return True

def test_train_script_dapt_mode():
    """Test that train.py can handle DAPT config without actually training."""
    print("🔍 Testing train.py DAPT mode...")
    
    try:
        # Import train script components
        sys.path.append('scripts')
        from train import load_config, build_cpt_or_mixed
        
        # Load DAPT config
        config = load_config("configs/run_dapt.yaml")
        
        # Test that build_cpt_or_mixed function exists and can be called
        # (We won't actually build datasets to avoid heavy computation)
        print("✅ train.py DAPT mode accessible")
        return True
        
    except Exception as e:
        print(f"❌ Error testing train.py DAPT mode: {e}")
        return False

def test_makefile_targets():
    """Test that Makefile contains DAPT targets."""
    print("🔍 Testing Makefile DAPT targets...")
    
    if not os.path.exists("Makefile"):
        print("❌ Makefile not found")
        return False
    
    try:
        with open("Makefile", 'r') as f:
            makefile_content = f.read()
        
        required_targets = ['dapt-docx:', 'dapt-train:']
        for target in required_targets:
            if target not in makefile_content:
                print(f"❌ Missing Makefile target: {target}")
                return False
        
        # Check .PHONY declaration
        if 'dapt-docx dapt-train' not in makefile_content:
            print("❌ DAPT targets not in .PHONY declaration")
            return False
        
        print("✅ Makefile DAPT targets present")
        return True
        
    except Exception as e:
        print(f"❌ Error reading Makefile: {e}")
        return False

def main():
    """Run all DAPT integration tests."""
    print("🚀 SFT-Play DAPT Integration Test")
    print("=" * 40)
    
    # Change to play directory if not already there
    if os.path.basename(os.getcwd()) != 'play':
        if os.path.exists('play'):
            os.chdir('play')
        else:
            print("❌ Not in play directory and play/ not found")
            sys.exit(1)
    
    tests = [
        test_config_loading,
        test_anchor_data,
        test_script_imports,
        test_train_script_dapt_mode,
        test_makefile_targets
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All DAPT integration tests passed!")
        print("\nNext steps:")
        print("1. Place DOCX files in data/raw/")
        print("2. Run: make dapt-docx")
        print("3. Run: make dapt-train")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
