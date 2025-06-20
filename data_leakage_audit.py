#!/usr/bin/env python3
"""
Data Leakage Audit for CortexFlow Dataset Loader
===============================================

Comprehensive audit to ensure no data leakage between train and test sets.
This script performs multiple tests to verify academic integrity compliance.

Author: CortexFlow Team
License: GPL-3.0
"""

import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Add src to path
sys.path.insert(0, 'src')

from src.data.loader import (
    load_dataset_gpu_optimized,
    load_dataset_raw,
    create_cv_folds,
    get_available_datasets,
    preprocess_data
)

warnings.filterwarnings('ignore')


def test_1_basic_data_leakage():
    """Test 1: Basic data leakage check - scaler statistics."""
    print("🔍 Test 1: Basic Data Leakage Check")
    print("-" * 50)
    
    available = get_available_datasets()
    if not available:
        print("❌ No datasets available")
        return False
    
    dataset_name = available[0]
    print(f"Testing with dataset: {dataset_name}")
    
    # Load with scaler
    X_train, y_train, X_test, y_test, scaler = load_dataset_gpu_optimized(
        dataset_name, test_size=0.3, random_state=42, return_scaler=True
    )
    
    # Check 1: Training data should have mean ≈ 0, std ≈ 1 after scaling
    train_mean = X_train.mean()
    train_std = X_train.std()
    
    print(f"✓ Training data statistics:")
    print(f"  Mean: {train_mean:.6f} (should be ≈ 0)")
    print(f"  Std:  {train_std:.6f} (should be ≈ 1)")
    
    # Check 2: Test data should NOT have mean ≈ 0 (indicates no leakage)
    test_mean = X_test.mean()
    test_std = X_test.std()
    
    print(f"✓ Test data statistics:")
    print(f"  Mean: {test_mean:.6f} (should NOT be ≈ 0)")
    print(f"  Std:  {test_std:.6f} (should NOT be ≈ 1)")
    
    # Check 3: Scaler statistics should match training data
    scaler_mean = scaler.mean_.mean()
    scaler_scale = scaler.scale_.mean()
    
    print(f"✓ Scaler statistics:")
    print(f"  Mean: {scaler_mean:.6f}")
    print(f"  Scale: {scaler_scale:.6f}")
    
    # Verification
    leakage_detected = abs(test_mean) < 0.01  # Test mean too close to 0
    
    if leakage_detected:
        print("❌ POTENTIAL DATA LEAKAGE DETECTED!")
        print("   Test data mean is too close to 0, indicating possible leakage")
        return False
    else:
        print("✅ No data leakage detected in basic test")
        return True


def test_2_manual_preprocessing_comparison():
    """Test 2: Compare with manual preprocessing to verify correctness."""
    print("\n🔍 Test 2: Manual Preprocessing Comparison")
    print("-" * 50)
    
    available = get_available_datasets()
    dataset_name = available[0]
    
    # Load raw data
    X_raw, y_raw = load_dataset_raw(dataset_name)
    
    # Manual preprocessing (CORRECT way)
    X_train_manual, X_test_manual, y_train_manual, y_test_manual = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42
    )
    
    # Fit scaler on training data only
    scaler_manual = StandardScaler()
    X_train_scaled_manual = scaler_manual.fit_transform(X_train_manual)
    X_test_scaled_manual = scaler_manual.transform(X_test_manual)
    
    # Loader preprocessing
    X_train_loader, y_train_loader, X_test_loader, y_test_loader = load_dataset_gpu_optimized(
        dataset_name, test_size=0.3, random_state=42
    )
    
    # Compare results
    train_diff = np.abs(X_train_scaled_manual - X_train_loader).max()
    test_diff = np.abs(X_test_scaled_manual - X_test_loader).max()
    
    print(f"✓ Manual vs Loader comparison:")
    print(f"  Max train difference: {train_diff:.10f}")
    print(f"  Max test difference:  {test_diff:.10f}")
    
    if train_diff < 1e-6 and test_diff < 1e-6:
        print("✅ Loader preprocessing matches manual preprocessing")
        return True
    else:
        print("❌ Loader preprocessing differs from manual preprocessing")
        return False


def test_3_cross_validation_leakage():
    """Test 3: Check for data leakage in cross-validation folds."""
    print("\n🔍 Test 3: Cross-Validation Data Leakage Check")
    print("-" * 50)

    available = get_available_datasets()
    dataset_name = available[0]

    # Load raw data
    X_raw, y_raw = load_dataset_raw(dataset_name)

    # Create CV folds
    folds = create_cv_folds(X_raw, y_raw, n_folds=5, random_state=42)

    leakage_detected = False
    val_means = []

    for i, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
        # Check training fold statistics
        train_mean = X_train_fold.mean()
        train_std = X_train_fold.std()

        # Check validation fold statistics
        val_mean = X_val_fold.mean()
        val_std = X_val_fold.std()
        val_means.append(val_mean)

        print(f"  Fold {i+1}: Train mean={train_mean:.6f}, Val mean={val_mean:.6f}")

        # Training mean should always be very close to 0 (standardized)
        if abs(train_mean) > 0.001:
            print(f"    ❌ Training mean not close to 0: {train_mean:.6f}")
            leakage_detected = True

        # Training std should always be very close to 1 (standardized)
        if abs(train_std - 1.0) > 0.001:
            print(f"    ❌ Training std not close to 1: {train_std:.6f}")
            leakage_detected = True

    # Statistical test: validation means should have reasonable variation
    val_means = np.array(val_means)
    val_mean_std = val_means.std()

    print(f"\n  📊 Validation statistics across folds:")
    print(f"     Mean of val means: {val_means.mean():.6f}")
    print(f"     Std of val means:  {val_mean_std:.6f}")

    # If all validation means are too similar, it might indicate leakage
    if val_mean_std < 0.01:
        print(f"    ⚠️  Validation means have very low variation: {val_mean_std:.6f}")
        print(f"    This could indicate data leakage or very small dataset")

    # Additional verification: manual CV to compare
    print(f"\n  🔧 Manual CV verification:")
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    manual_val_means = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_raw)):
        X_train_manual = X_raw[train_idx]
        X_val_manual = X_raw[val_idx]

        # Manual preprocessing
        scaler_manual = StandardScaler()
        X_train_scaled = scaler_manual.fit_transform(X_train_manual)
        X_val_scaled = scaler_manual.transform(X_val_manual)

        manual_val_means.append(X_val_scaled.mean())

    # Compare manual vs loader CV
    manual_val_means = np.array(manual_val_means)
    diff = np.abs(val_means - manual_val_means).max()

    print(f"     Max difference from manual CV: {diff:.10f}")

    if diff < 1e-6:
        print("     ✅ CV matches manual implementation")
    else:
        print("     ❌ CV differs from manual implementation")
        leakage_detected = True

    if not leakage_detected:
        print("✅ No data leakage detected in cross-validation")
        return True
    else:
        print("❌ Data leakage detected in cross-validation")
        return False


def test_4_reproducibility_check():
    """Test 4: Check reproducibility with different random seeds."""
    print("\n🔍 Test 4: Reproducibility Check")
    print("-" * 50)
    
    available = get_available_datasets()
    dataset_name = available[0]
    
    # Load with same seed multiple times
    results = []
    for i in range(3):
        X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(
            dataset_name, test_size=0.3, random_state=42
        )
        results.append((X_train, y_train, X_test, y_test))
    
    # Check if results are identical
    all_identical = True
    for i in range(1, len(results)):
        if not np.array_equal(results[0][0], results[i][0]):
            all_identical = False
            break
    
    if all_identical:
        print("✅ Results are reproducible with same random seed")
    else:
        print("❌ Results are not reproducible")
        return False
    
    # Load with different seeds
    X_train_1, _, X_test_1, _ = load_dataset_gpu_optimized(
        dataset_name, test_size=0.3, random_state=42
    )
    X_train_2, _, X_test_2, _ = load_dataset_gpu_optimized(
        dataset_name, test_size=0.3, random_state=123
    )
    
    # Results should be different with different seeds
    different_seeds = not np.array_equal(X_train_1, X_train_2)
    
    if different_seeds:
        print("✅ Results differ with different random seeds")
        return True
    else:
        print("❌ Results are identical with different seeds (unexpected)")
        return False


def test_5_target_preprocessing_check():
    """Test 5: Check target preprocessing doesn't cause leakage."""
    print("\n🔍 Test 5: Target Preprocessing Check")
    print("-" * 50)
    
    available = get_available_datasets()
    dataset_name = available[0]
    
    # Load raw data
    X_raw, y_raw = load_dataset_raw(dataset_name)
    
    # Check original target range
    print(f"✓ Original target range: [{y_raw.min():.3f}, {y_raw.max():.3f}]")
    
    # Load processed data
    X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(dataset_name)
    
    # Check processed target ranges
    print(f"✓ Train target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"✓ Test target range:  [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    # Targets should be normalized to [0, 1] range
    train_in_range = (y_train.min() >= 0) and (y_train.max() <= 1)
    test_in_range = (y_test.min() >= 0) and (y_test.max() <= 1)
    
    if train_in_range and test_in_range:
        print("✅ Target preprocessing is correct (normalized to [0,1])")
        return True
    else:
        print("❌ Target preprocessing is incorrect")
        return False


def test_6_edge_cases():
    """Test 6: Edge cases and potential leakage scenarios."""
    print("\n🔍 Test 6: Edge Cases Check")
    print("-" * 50)
    
    available = get_available_datasets()
    dataset_name = available[0]
    
    # Test with very small test size
    try:
        X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(
            dataset_name, test_size=0.05, random_state=42
        )
        print(f"✓ Small test size works: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    except Exception as e:
        print(f"❌ Small test size failed: {e}")
        return False
    
    # Test with very large test size
    try:
        X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(
            dataset_name, test_size=0.8, random_state=42
        )
        print(f"✓ Large test size works: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    except Exception as e:
        print(f"❌ Large test size failed: {e}")
        return False
    
    print("✅ Edge cases handled correctly")
    return True


def main():
    """Run comprehensive data leakage audit."""
    print("🔒 CortexFlow Data Leakage Audit")
    print("=" * 60)
    print("Comprehensive audit to ensure no data leakage between train/test sets")
    print("=" * 60)
    
    tests = [
        ("Basic Data Leakage Check", test_1_basic_data_leakage),
        ("Manual Preprocessing Comparison", test_2_manual_preprocessing_comparison),
        ("Cross-Validation Leakage Check", test_3_cross_validation_leakage),
        ("Reproducibility Check", test_4_reproducibility_check),
        ("Target Preprocessing Check", test_5_target_preprocessing_check),
        ("Edge Cases Check", test_6_edge_cases),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - NO DATA LEAKAGE DETECTED")
        print("✅ Loader is academically compliant and ready for publication")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
        print("❌ Data leakage may be present")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
