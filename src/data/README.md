# CortexFlow Dataset Loader 🧠📊

Comprehensive dataset loader for fMRI brain-to-image reconstruction datasets with GPU optimization and academic integrity compliance.

## ✨ Features

- **🚀 GPU Optimized**: Efficient loading for CUDA devices
- **🔒 No Data Leakage**: Proper preprocessing prevents information leakage
- **🔄 Cross-Validation**: Built-in CV support with fold-wise preprocessing
- **📊 Multiple Formats**: Supports various .mat file structures
- **🎯 Reproducible**: Fixed random seeds for consistent results
- **⚡ Easy to Use**: Simple API for complex preprocessing pipelines

## 📦 Supported Datasets

| Dataset | Samples | Features | Description |
|---------|---------|----------|-------------|
| **Miyawaki** | 119 | 967 | Visual cortex fMRI data |
| **MindBigData** | 1080 | 3092 | Large-scale neural data |
| **Crell** | 576 | 3092 | Medium complexity dataset |
| **Digit69** | Variable | Variable | Digit recognition dataset |
| **Vangerven** | 90 | 3092 | Visual stimuli fMRI data |

## 🚀 Quick Start

### Basic Usage

```python
from src.data.loader import load_dataset_gpu_optimized

# Load dataset with automatic preprocessing
X_train, y_train, X_test, y_test = load_dataset_gpu_optimized('miyawaki')

print(f"Training: {X_train.shape} -> {y_train.shape}")
print(f"Test: {X_test.shape} -> {y_test.shape}")
```

### Check Available Datasets

```python
from src.data.loader import get_available_datasets, print_dataset_summary

# Show summary of all datasets
print_dataset_summary()

# Get list of available datasets
available = get_available_datasets()
print(f"Available: {available}")
```

### Cross-Validation

```python
from src.data.loader import load_dataset_raw, create_cv_folds

# Load raw data
X, y = load_dataset_raw('miyawaki')

# Create 10-fold CV with proper preprocessing
folds = create_cv_folds(X, y, n_folds=10)

for i, (X_train, y_train, X_val, y_val) in enumerate(folds):
    print(f"Fold {i+1}: Train {X_train.shape[0]}, Val {X_val.shape[0]}")
```

## 🔧 API Reference

### Core Functions

#### `load_dataset_gpu_optimized(dataset_name, **kwargs)`

Load dataset with GPU optimization and preprocessing.

**Parameters:**
- `dataset_name` (str): Dataset name ('miyawaki', 'mindbigdata', 'crell', etc.)
- `test_size` (float): Test set proportion (default: 0.2)
- `random_state` (int): Random seed (default: 42)
- `return_scaler` (bool): Return fitted scaler (default: False)

**Returns:**
- `X_train, y_train, X_test, y_test` or with scaler if requested

#### `load_dataset_raw(dataset_name, data_dir)`

Load raw dataset without preprocessing.

**Parameters:**
- `dataset_name` (str): Dataset name
- `data_dir` (str): Data directory (default: 'datasets/processed')

**Returns:**
- `X, y`: Raw features and targets

#### `create_cv_folds(X, y, n_folds, random_state)`

Create cross-validation folds with proper preprocessing.

**Parameters:**
- `X, y`: Features and targets
- `n_folds` (int): Number of folds (default: 10)
- `random_state` (int): Random seed (default: 42)

**Returns:**
- List of (X_train, y_train, X_val, y_val) tuples

### Utility Functions

#### `get_available_datasets()`
Returns list of available dataset names.

#### `get_dataset_info(dataset_name)`
Returns detailed information about a dataset.

#### `print_dataset_summary()`
Prints comprehensive summary of all datasets.

## 🔬 Academic Integrity Features

### ✅ No Data Leakage
- **Training Statistics Only**: Scaler fitted on training data only
- **Test Normalization**: Uses training statistics for test preprocessing
- **CV Preprocessing**: Preprocessing performed within each fold

### ✅ Reproducible Results
- **Fixed Seeds**: Consistent random seeds for splits
- **Deterministic**: Same results across runs
- **Cross-Platform**: Works on different systems

### ✅ Proper Methodology
```python
# ❌ WRONG: Data leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses ALL data
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT: No data leakage
X_train, X_test = train_test_split(X)  # Split FIRST
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Transform test with train stats
```

## 📊 Data Format

### Input Format (.mat files)
The loader supports multiple .mat file structures:

**Standard Format:**
```matlab
X: [samples × features] - fMRI data
y: [samples × pixels] - Target images (28×28 flattened)
```

**Alternative Format:**
```matlab
fmriTrn: [train_samples × features] - Training fMRI
stimTrn: [train_samples × pixels] - Training images
fmriTest: [test_samples × features] - Test fMRI
stimTest: [test_samples × pixels] - Test images
```

### Output Format
- **Features (X)**: `[samples, features]` - Standardized fMRI data (float32)
- **Targets (y)**: `[samples, 784]` - Normalized images 0-1 range (float32)
- **Images**: 28×28 grayscale images flattened to 784 pixels

## 🛠️ Advanced Usage

### Custom Preprocessing

```python
# Load with custom parameters
X_train, y_train, X_test, y_test, scaler = load_dataset_gpu_optimized(
    'miyawaki',
    test_size=0.3,      # 30% test set
    random_state=123,   # Custom seed
    return_scaler=True  # Get fitted scaler
)

# Use scaler for new data
X_new_scaled = scaler.transform(X_new)
```

### Multiple Datasets

```python
datasets = ['miyawaki', 'mindbigdata', 'crell']
results = {}

for dataset in datasets:
    if dataset in get_available_datasets():
        X_train, y_train, X_test, y_test = load_dataset_gpu_optimized(dataset)
        results[dataset] = (X_train, y_train, X_test, y_test)
        print(f"{dataset}: {X_train.shape} -> {y_train.shape}")
```

### Error Handling

```python
try:
    X_train, y_train, X_test, y_test = load_dataset_gpu_optimized('miyawaki')
except FileNotFoundError:
    print("Dataset file not found. Please check datasets/processed/")
except ValueError as e:
    print(f"Data format error: {e}")
```

## 🧪 Testing

Run the test suite:

```bash
python test_loader.py
```

Run usage examples:

```bash
python examples/dataset_usage_examples.py
```

## 📁 File Structure

```
src/data/
├── __init__.py          # Package initialization
├── loader.py            # Main loader functions
└── README.md           # This documentation

examples/
├── dataset_usage_examples.py  # Comprehensive examples
└── test_loader.py            # Test suite
```

## 🔍 Troubleshooting

### Common Issues

**1. FileNotFoundError**
```
Solution: Ensure .mat files are in datasets/processed/
Check: get_available_datasets()
```

**2. Key Error in .mat file**
```
Solution: Loader auto-detects format, check available keys
Debug: Available keys printed during loading
```

**3. Shape Mismatch**
```
Solution: Loader handles various shapes automatically
Check: Target images should be 28×28 or 784 pixels
```

### Performance Tips

- **Memory**: Large datasets loaded as float32 for efficiency
- **GPU**: Data ready for CUDA transfer
- **Caching**: Consider caching preprocessed data for repeated use

## 📄 License

GPL-3.0 License - See LICENSE file for details.

## 👥 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure academic integrity compliance
5. Submit pull request

---

**CortexFlow Team** - Building the future of brain-computer interfaces 🧠🚀
