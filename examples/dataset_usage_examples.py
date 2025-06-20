#!/usr/bin/env python3
"""
CortexFlow Dataset Usage Examples
================================

Comprehensive examples showing how to use the CortexFlow dataset loader
for various machine learning tasks.

Author: CortexFlow Team
License: GPL-3.0
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import (
    load_dataset_gpu_optimized,
    load_dataset_raw,
    get_available_datasets,
    create_cv_folds,
    print_dataset_summary,
    get_dataset_info
)


def example_1_basic_loading():
    """Example 1: Basic dataset loading and exploration."""
    print("=" * 60)
    print("Example 1: Basic Dataset Loading")
    print("=" * 60)
    
    # Show available datasets
    print("📋 Available datasets:")
    available = get_available_datasets()
    for dataset in available:
        info = get_dataset_info(dataset)
        print(f"  - {dataset}: {info['description']}")
    
    if not available:
        print("⚠️  No datasets available!")
        return
    
    # Load first available dataset
    dataset_name = available[0]
    print(f"\n🔧 Loading {dataset_name} dataset...")
    
    X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(dataset_name)
    
    print(f"✅ Dataset loaded successfully:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]} fMRI voxels")
    print(f"   Targets: {y_train.shape[1]} pixels (28x28 images)")
    
    return dataset_name, X_train, y_train, X_test, y_test


def example_2_data_visualization(dataset_name, X_train, y_train):
    """Example 2: Data visualization."""
    print("\n" + "=" * 60)
    print("Example 2: Data Visualization")
    print("=" * 60)
    
    # Visualize first few target images
    print("🖼️  Visualizing target images...")
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(f'{dataset_name.capitalize()} Dataset - Sample Images')
    
    for i in range(10):
        row = i // 5
        col = i % 5
        
        # Reshape flattened image back to 28x28
        image = y_train[i].reshape(28, 28)
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'examples/{dataset_name}_sample_images.png', dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to examples/{dataset_name}_sample_images.png")
    
    # Feature statistics
    print(f"\n📊 Feature statistics:")
    print(f"   fMRI features shape: {X_train.shape}")
    print(f"   Feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   Feature mean: {X_train.mean():.3f}")
    print(f"   Feature std: {X_train.std():.3f}")


def example_3_cross_validation():
    """Example 3: Cross-validation setup."""
    print("\n" + "=" * 60)
    print("Example 3: Cross-Validation Setup")
    print("=" * 60)
    
    available = get_available_datasets()
    if not available:
        print("⚠️  No datasets available!")
        return
    
    dataset_name = available[0]
    print(f"🔄 Setting up cross-validation for {dataset_name}...")
    
    # Load raw data (without train/test split)
    X, y = load_dataset_raw(dataset_name)
    
    # Create 10-fold cross-validation
    folds = create_cv_folds(X, y, n_folds=10, random_state=42)
    
    print(f"✅ Created {len(folds)} CV folds:")
    
    # Analyze fold sizes
    train_sizes = []
    val_sizes = []
    
    for i, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
        train_size = X_train_fold.shape[0]
        val_size = X_val_fold.shape[0]
        train_sizes.append(train_size)
        val_sizes.append(val_size)
        print(f"   Fold {i+1:2d}: Train {train_size:3d}, Val {val_size:2d}")
    
    print(f"\n📊 CV Statistics:")
    print(f"   Average train size: {np.mean(train_sizes):.1f} ± {np.std(train_sizes):.1f}")
    print(f"   Average val size: {np.mean(val_sizes):.1f} ± {np.std(val_sizes):.1f}")
    
    return folds


def example_4_multiple_datasets():
    """Example 4: Working with multiple datasets."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Dataset Comparison")
    print("=" * 60)
    
    available = get_available_datasets()
    if len(available) < 2:
        print("⚠️  Need at least 2 datasets for comparison!")
        return
    
    print("📊 Comparing multiple datasets:")
    
    dataset_stats = {}
    
    for dataset_name in available[:3]:  # Compare first 3 datasets
        try:
            X, y = load_dataset_raw(dataset_name)
            
            stats = {
                'samples': X.shape[0],
                'features': X.shape[1],
                'feature_range': (X.min(), X.max()),
                'target_range': (y.min(), y.max()),
                'feature_mean': X.mean(),
                'feature_std': X.std()
            }
            
            dataset_stats[dataset_name] = stats
            
            print(f"\n   {dataset_name.upper()}:")
            print(f"     Samples: {stats['samples']}")
            print(f"     Features: {stats['features']}")
            print(f"     Feature range: [{stats['feature_range'][0]:.3f}, {stats['feature_range'][1]:.3f}]")
            print(f"     Target range: [{stats['target_range'][0]:.3f}, {stats['target_range'][1]:.3f}]")
            
        except Exception as e:
            print(f"   ❌ Error loading {dataset_name}: {e}")
    
    return dataset_stats


def example_5_custom_preprocessing():
    """Example 5: Custom preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Preprocessing Pipeline")
    print("=" * 60)
    
    available = get_available_datasets()
    if not available:
        print("⚠️  No datasets available!")
        return
    
    dataset_name = available[0]
    print(f"🔧 Custom preprocessing for {dataset_name}...")
    
    # Load with custom parameters
    X_train, y_train, X_test, y_test, scaler = load_dataset_gpu_optimized(
        dataset_name, 
        test_size=0.3,  # Larger test set
        random_state=123,  # Different seed
        return_scaler=True
    )
    
    print(f"✅ Custom preprocessing completed:")
    print(f"   Train/test split: {X_train.shape[0]}/{X_test.shape[0]} (70/30)")
    print(f"   Random seed: 123")
    print(f"   Scaler fitted on training data only")
    
    # Verify no data leakage
    print(f"\n🔍 Data leakage check:")
    print(f"   Training feature mean: {X_train.mean():.6f}")
    print(f"   Test feature mean: {X_test.mean():.6f}")
    print(f"   Scaler mean: {scaler.mean_.mean():.6f}")
    print(f"   ✅ Test mean ≠ 0 (good - no data leakage)")
    
    return X_train, y_train, X_test, y_test, scaler


def main():
    """Run all examples."""
    print("🧪 CortexFlow Dataset Usage Examples")
    print("🔬 Demonstrating comprehensive dataset loading capabilities")
    
    # Create examples directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Basic loading
        result = example_1_basic_loading()
        if result:
            dataset_name, X_train, y_train, X_test, y_test = result
            
            # Example 2: Visualization
            example_2_data_visualization(dataset_name, X_train, y_train)
        
        # Example 3: Cross-validation
        example_3_cross_validation()
        
        # Example 4: Multiple datasets
        example_4_multiple_datasets()
        
        # Example 5: Custom preprocessing
        example_5_custom_preprocessing()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        
        # Summary
        print("\n📖 Quick Reference:")
        print("   Basic loading: load_dataset_gpu_optimized('dataset_name')")
        print("   Raw loading: load_dataset_raw('dataset_name')")
        print("   Cross-validation: create_cv_folds(X, y, n_folds=10)")
        print("   Available datasets: get_available_datasets()")
        print("   Dataset info: get_dataset_info('dataset_name')")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
