#!/usr/bin/env python3
"""
Test script for CortexFlow dataset loader
=========================================

This script demonstrates how to use the dataset loader functions
and tests their functionality with available datasets.

Usage:
    python test_loader.py

Author: CortexFlow Team
License: GPL-3.0
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')

from src.data.loader import (
    print_dataset_summary,
    get_available_datasets,
    load_dataset_gpu_optimized,
    load_dataset_raw,
    create_cv_folds,
    get_dataset_info
)


def test_dataset_detection():
    """Test dataset detection and availability."""
    print("🔍 Testing dataset detection...")
    
    # Print summary of all datasets
    print_dataset_summary()
    
    # Get available datasets
    available = get_available_datasets()
    print(f"\n📋 Available datasets: {available}")
    
    return available


def test_raw_loading(dataset_name: str):
    """Test raw dataset loading."""
    print(f"\n🔧 Testing raw loading for {dataset_name}...")
    
    try:
        X, y = load_dataset_raw(dataset_name)
        print(f"✅ Raw loading successful:")
        print(f"   X shape: {X.shape}, dtype: {X.dtype}")
        print(f"   y shape: {y.shape}, dtype: {y.dtype}")
        print(f"   X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   y range: [{y.min():.3f}, {y.max():.3f}]")
        return True
    except Exception as e:
        print(f"❌ Raw loading failed: {e}")
        return False


def test_gpu_optimized_loading(dataset_name: str):
    """Test GPU-optimized loading with preprocessing."""
    print(f"\n⚡ Testing GPU-optimized loading for {dataset_name}...")
    
    try:
        # Load with default settings
        X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(
            dataset_name, test_size=0.2, random_state=42
        )
        
        print(f"✅ GPU-optimized loading successful:")
        print(f"   Train: X{X_train.shape} -> y{y_train.shape}")
        print(f"   Test:  X{X_test.shape} -> y{y_test.shape}")
        print(f"   X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"   y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"   Data types: X={X_train.dtype}, y={y_train.dtype}")
        
        # Test with scaler return
        X_train2, y_train2, X_test2, y_test2, scaler = load_dataset_gpu_optimized(
            dataset_name, return_scaler=True
        )
        print(f"   Scaler mean shape: {scaler.mean_.shape}")
        print(f"   Scaler scale shape: {scaler.scale_.shape}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"❌ GPU-optimized loading failed: {e}")
        return None


def test_cross_validation(X: np.ndarray, y: np.ndarray, dataset_name: str):
    """Test cross-validation fold creation."""
    print(f"\n🔄 Testing cross-validation for {dataset_name}...")
    
    try:
        # Create 5-fold CV (smaller for testing)
        folds = create_cv_folds(X, y, n_folds=5, random_state=42)
        
        print(f"✅ Cross-validation successful:")
        print(f"   Created {len(folds)} folds")
        
        for i, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
            print(f"   Fold {i+1}: Train {X_train_fold.shape[0]}, Val {X_val_fold.shape[0]}")
            
        return True
        
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        return False


def test_dataset_info():
    """Test dataset information retrieval."""
    print(f"\n📊 Testing dataset information...")
    
    try:
        for dataset_name in ['miyawaki', 'mindbigdata', 'crell', 'digit69']:
            info = get_dataset_info(dataset_name)
            print(f"   {dataset_name}: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset info failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 CortexFlow Dataset Loader Test Suite")
    print("=" * 50)
    
    # Test 1: Dataset detection
    available_datasets = test_dataset_detection()
    
    # Test 2: Dataset information
    test_dataset_info()
    
    if not available_datasets:
        print("\n⚠️  No datasets available for testing.")
        print("   Please ensure .mat files are in datasets/processed/")
        return
    
    # Test with first available dataset
    test_dataset = available_datasets[0]
    print(f"\n🎯 Running detailed tests with: {test_dataset}")
    
    # Test 3: Raw loading
    if not test_raw_loading(test_dataset):
        return
    
    # Test 4: GPU-optimized loading
    result = test_gpu_optimized_loading(test_dataset)
    if result is None:
        return
    
    X_train, y_train, X_test, y_test = result
    
    # Test 5: Cross-validation
    # Use original data for CV (combine train+test)
    X_original, y_original = load_dataset_raw(test_dataset)
    test_cross_validation(X_original, y_original, test_dataset)
    
    print("\n🎉 All tests completed!")
    print("=" * 50)
    
    # Usage examples
    print("\n📖 Usage Examples:")
    print("=" * 20)
    print("# Basic usage:")
    print("from src.data.loader import load_dataset_gpu_optimized")
    print(f"X_train, y_train, X_test, y_test = load_dataset_gpu_optimized('{test_dataset}')")
    print()
    print("# With cross-validation:")
    print("from src.data.loader import load_dataset_raw, create_cv_folds")
    print(f"X, y = load_dataset_raw('{test_dataset}')")
    print("folds = create_cv_folds(X, y, n_folds=10)")
    print()
    print("# Check available datasets:")
    print("from src.data.loader import get_available_datasets, print_dataset_summary")
    print("print_dataset_summary()")
    print("available = get_available_datasets()")


if __name__ == "__main__":
    main()
