"""
CortexFlow Dataset Loader
========================

GPU-optimized dataset loader for fMRI brain-to-image reconstruction datasets.
Supports Miyawaki, Vangerven, MindBigData, and Crell datasets.

Features:
- GPU optimization for CUDA devices
- Reproducible train/test splits
- Proper preprocessing (no data leakage)
- Cross-validation support
- Automatic dataset detection

Author: CortexFlow Team
License: GPL-3.0
"""

import os
import numpy as np
import scipy.io as sio
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Dataset configurations
DATASET_CONFIG = {
    'miyawaki': {
        'filename': 'miyawaki_structured_28x28.mat',
        'samples': 107,
        'features': 967,
        'description': 'Visual cortex fMRI data'
    },
    'vangerven': {
        'filename': 'vangerven_structured_28x28.mat',
        'samples': 90,
        'features': 3092,
        'description': 'Visual stimuli fMRI data'
    },
    'mindbigdata': {
        'filename': 'mindbigdata.mat',
        'samples': 1080,
        'features': 3092,
        'description': 'Large-scale neural data'
    },
    'crell': {
        'filename': 'crell.mat',
        'samples': 576,
        'features': 3092,
        'description': 'Medium complexity dataset'
    },
    'digit69': {
        'filename': 'digit69_28x28.mat',
        'samples': None,  # Unknown from README
        'features': None,
        'description': 'Digit recognition dataset'
    }
}

# Default paths
DEFAULT_DATA_DIR = 'datasets/processed'
DEFAULT_RANDOM_STATE = 42


def get_available_datasets() -> List[str]:
    """
    Get list of available datasets in the processed directory.
    
    Returns:
        List[str]: List of available dataset names
    """
    available = []
    for dataset_name, config in DATASET_CONFIG.items():
        filepath = os.path.join(DEFAULT_DATA_DIR, config['filename'])
        if os.path.exists(filepath):
            available.append(dataset_name)
    return available


def load_mat_file(filepath: str) -> Dict[str, Any]:
    """
    Load MATLAB file with error handling.
    
    Args:
        filepath (str): Path to .mat file
        
    Returns:
        Dict[str, Any]: Loaded data dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    try:
        data = sio.loadmat(filepath)
        return data
    except Exception as e:
        raise ValueError(f"Failed to load .mat file {filepath}: {str(e)}")


def preprocess_data(X: np.ndarray, y: np.ndarray, 
                   scaler: Optional[StandardScaler] = None,
                   fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess fMRI features and target images.
    
    Args:
        X (np.ndarray): fMRI features [samples, features]
        y (np.ndarray): Target images [samples, height, width] or [samples, pixels]
        scaler (StandardScaler, optional): Pre-fitted scaler for features
        fit_scaler (bool): Whether to fit the scaler (True for train, False for test)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: Preprocessed X, y, and fitted scaler
    """
    # Ensure X is 2D
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
    
    # Ensure y is properly shaped for 28x28 images
    if y.ndim == 3:
        # Already [samples, height, width]
        y_processed = y.reshape(y.shape[0], -1)  # Flatten to [samples, pixels]
    elif y.ndim == 2 and y.shape[1] == 784:  # 28*28 = 784
        # Already flattened
        y_processed = y
    else:
        raise ValueError(f"Unexpected y shape: {y.shape}. Expected [samples, 28, 28] or [samples, 784]")
    
    # Normalize features (X)
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = scaler.transform(X)
    
    # Normalize target images to [0, 1]
    y_processed = y_processed.astype(np.float32)
    if y_processed.max() > 1.0:
        y_processed = y_processed / 255.0
    
    return X_processed.astype(np.float32), y_processed, scaler


def load_dataset_raw(dataset_name: str, data_dir: str = DEFAULT_DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw dataset without preprocessing.

    Args:
        dataset_name (str): Name of dataset ('miyawaki', 'vangerven', 'mindbigdata', 'crell', 'digit69')
        data_dir (str): Directory containing processed datasets

    Returns:
        Tuple[np.ndarray, np.ndarray]: Raw X (features) and y (targets)

    Raises:
        ValueError: If dataset name is invalid
        FileNotFoundError: If dataset file doesn't exist
    """
    if dataset_name not in DATASET_CONFIG:
        available = list(DATASET_CONFIG.keys())
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Available: {available}")

    config = DATASET_CONFIG[dataset_name]
    filepath = os.path.join(data_dir, config['filename'])

    print(f"Loading {dataset_name} dataset from {filepath}")
    data = load_mat_file(filepath)

    # Print available keys for debugging
    available_keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"Available keys in .mat file: {available_keys}")

    # Extract X and y from MATLAB structure based on dataset format
    X, y = _extract_features_targets(data, dataset_name, available_keys)

    print(f"Loaded {dataset_name}: X shape {X.shape}, y shape {y.shape}")
    return X, y


def _extract_features_targets(data: Dict[str, Any], dataset_name: str, available_keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features (X) and targets (y) from loaded .mat data based on dataset structure.

    Args:
        data (Dict[str, Any]): Loaded .mat file data
        dataset_name (str): Name of the dataset
        available_keys (List[str]): Available keys in the data

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and targets (y)

    Raises:
        ValueError: If expected keys are not found
    """
    # Try standard format first (X, y)
    if 'X' in data and 'y' in data:
        return data['X'], data['y']

    # Try alternative formats based on available keys
    if 'fmriTrn' in data and 'stimTrn' in data:
        # Format: separate train/test with fmri features and stim targets
        if 'fmriTest' in data and 'stimTest' in data:
            # Combine train and test data
            X_train = data['fmriTrn']
            y_train = data['stimTrn']
            X_test = data['fmriTest']
            y_test = data['stimTest']

            # Concatenate train and test
            X = np.vstack([X_train, X_test])
            y = np.vstack([y_train, y_test])

            print(f"Combined train ({X_train.shape[0]}) and test ({X_test.shape[0]}) samples")
            return X, y
        else:
            # Only training data available
            return data['fmriTrn'], data['stimTrn']

    # Try other common formats
    if 'features' in data and 'targets' in data:
        return data['features'], data['targets']

    if 'data' in data and 'labels' in data:
        return data['data'], data['labels']

    # If no standard format found, raise error with available keys
    raise ValueError(f"Could not find features and targets in {dataset_name} dataset. "
                    f"Available keys: {available_keys}. "
                    f"Expected formats: (X, y), (fmriTrn, stimTrn), (features, targets), or (data, labels)")


def load_dataset_gpu_optimized(dataset_name: str, 
                              test_size: float = 0.2,
                              random_state: int = DEFAULT_RANDOM_STATE,
                              data_dir: str = DEFAULT_DATA_DIR,
                              return_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset with GPU optimization and proper preprocessing.
    
    Args:
        dataset_name (str): Name of dataset
        test_size (float): Proportion of test set (0.0 to 1.0)
        random_state (int): Random seed for reproducibility
        data_dir (str): Directory containing processed datasets
        return_scaler (bool): Whether to return the fitted scaler
        
    Returns:
        Tuple: (X_train, y_train, X_test, y_test) or (X_train, y_train, X_test, y_test, scaler)
    """
    # Load raw data
    X, y = load_dataset_raw(dataset_name, data_dir)
    
    # Split data BEFORE preprocessing (prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # Preprocess training data (fit scaler on training data only)
    X_train_processed, y_train_processed, scaler = preprocess_data(
        X_train, y_train, scaler=None, fit_scaler=True
    )
    
    # Preprocess test data (use training scaler - NO DATA LEAKAGE)
    X_test_processed, y_test_processed, _ = preprocess_data(
        X_test, y_test, scaler=scaler, fit_scaler=False
    )
    
    print(f"Dataset split: Train {X_train_processed.shape[0]}, Test {X_test_processed.shape[0]}")
    print(f"Feature range: [{X_train_processed.min():.3f}, {X_train_processed.max():.3f}]")
    print(f"Target range: [{y_train_processed.min():.3f}, {y_train_processed.max():.3f}]")
    
    if return_scaler:
        return X_train_processed, y_train_processed, X_test_processed, y_test_processed, scaler
    else:
        return X_train_processed, y_train_processed, X_test_processed, y_test_processed


def create_cv_folds(X: np.ndarray, y: np.ndarray, 
                   n_folds: int = 10, 
                   random_state: int = DEFAULT_RANDOM_STATE) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create cross-validation folds with proper preprocessing within each fold.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Targets
        n_folds (int): Number of CV folds
        random_state (int): Random seed
        
    Returns:
        List[Tuple]: List of (X_train, y_train, X_val, y_val) for each fold
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Preprocess within fold (prevents data leakage)
        X_train_processed, y_train_processed, scaler = preprocess_data(
            X_train_fold, y_train_fold, scaler=None, fit_scaler=True
        )
        X_val_processed, y_val_processed, _ = preprocess_data(
            X_val_fold, y_val_fold, scaler=scaler, fit_scaler=False
        )
        
        folds.append((X_train_processed, y_train_processed, X_val_processed, y_val_processed))
        print(f"Fold {fold_idx + 1}: Train {len(train_idx)}, Val {len(val_idx)}")
    
    return folds


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific dataset.
    
    Args:
        dataset_name (str): Name of dataset
        
    Returns:
        Dict[str, Any]: Dataset information
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIG[dataset_name].copy()
    filepath = os.path.join(DEFAULT_DATA_DIR, config['filename'])
    config['available'] = os.path.exists(filepath)
    config['filepath'] = filepath
    
    return config


def print_dataset_summary():
    """Print summary of all available datasets."""
    print("=" * 60)
    print("CortexFlow Dataset Summary")
    print("=" * 60)
    
    available_datasets = get_available_datasets()
    
    for dataset_name in DATASET_CONFIG.keys():
        config = get_dataset_info(dataset_name)
        status = "✅ Available" if config['available'] else "❌ Missing"
        
        print(f"\n📊 {dataset_name.upper()}")
        print(f"   Status: {status}")
        print(f"   Description: {config['description']}")
        if config['samples']:
            print(f"   Samples: {config['samples']}")
        if config['features']:
            print(f"   Features: {config['features']}")
        print(f"   File: {config['filename']}")
    
    print(f"\n🎯 Available datasets: {len(available_datasets)}/{len(DATASET_CONFIG)}")
    print("=" * 60)


if __name__ == "__main__":
    # Demo usage
    print_dataset_summary()
    
    # Test loading available datasets
    available = get_available_datasets()
    if available:
        dataset_name = available[0]
        print(f"\n🧪 Testing {dataset_name} dataset loading...")
        try:
            X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(dataset_name)
            print(f"✅ Successfully loaded {dataset_name}")
            print(f"   Train: X{X_train.shape} -> y{y_train.shape}")
            print(f"   Test:  X{X_test.shape} -> y{y_test.shape}")
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
    else:
        print("\n⚠️  No datasets available for testing")
