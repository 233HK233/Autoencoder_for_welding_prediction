import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except Exception:
    class StandardScaler:  # Minimal fallback when sklearn is unavailable.
        def __init__(self) -> None:
            self.mean_: np.ndarray | None = None
            self.scale_: np.ndarray | None = None

        def fit(self, x: np.ndarray) -> "StandardScaler":
            x = np.asarray(x, dtype=np.float64)
            self.mean_ = x.mean(axis=0)
            std = x.std(axis=0, ddof=0)
            self.scale_ = np.where(std < 1e-12, 1.0, std)
            return self

        def transform(self, x: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError("StandardScaler must be fitted before transform")
            x = np.asarray(x, dtype=np.float64)
            return (x - self.mean_) / self.scale_

        def fit_transform(self, x: np.ndarray) -> np.ndarray:
            return self.fit(x).transform(x)

# Mapping based on unique values observed in a01.csv
# 'quasistable' -> 0 (Normal)
# 'nonstationary' -> 1 (Warning/Transition)
# 'instability' -> 2 (Failure)
LABEL_MAP = {
    'quasistable': 0,
    'nonstationary': 1,
    'instability': 2
}

def load_sequences_from_folder(
    folder_path: str, 
    window_size: int = 20, 
    stride: int = 1
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load all CSV files from a folder, process them into sliding window sequences,
    and return X (sequences) and y (labels).
    """
    all_sequences = []
    all_labels = []
    
    # Sort files to ensure reproducibility
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {folder_path}")
        return [], []

    print(f"Found {len(csv_files)} datasets in {folder_path}")

    for file_path in csv_files:
        try:
            # Read CSV - assuming no header based on inspection of a01.csv
            df = pd.read_csv(file_path, header=None)
            
            # Extract features (first 18 columns) and labels (last column)
            # a01.csv has 19 columns: 0-17 data, 18 label
            data = df.iloc[:, :18].values.astype(np.float32)
            
            # Map string labels to integers
            raw_labels = df.iloc[:, 18].map(lambda x: x.strip() if isinstance(x, str) else x)
            labels = raw_labels.map(LABEL_MAP).values
            
            # Check for mapping errors
            if np.isnan(labels).any():
                print(f"Warning: Found unmapped labels in {file_path}. Unique labels: {raw_labels.unique()}")
                # Fill NaN with 0 or drop? Dropping for safety.
                valid_indices = ~np.isnan(labels)
                data = data[valid_indices]
                labels = labels[valid_indices].astype(int)

            num_samples = len(data)
            
            # Sliding window generation
            # We want sequences of length `window_size`
            # With step `stride`
            for i in range(0, num_samples - window_size + 1, stride):
                # Input sequence
                seq = data[i : i + window_size]
                
                # Label for the sequence
                # Usually we take the label of the LAST time step in the window
                label = labels[i + window_size - 1]
                
                all_sequences.append(seq)
                all_labels.append(label)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    return all_sequences, all_labels

def standardize_train_test(
    X_train_raw: np.ndarray, 
    X_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize training and test data using Z-score normalization (StandardScaler).
    The scaler is fit ONLY on the training data to prevent data leakage.
    
    Args:
        X_train_raw: Shape (N_train, Sequence_Length, Features)
        X_test_raw: Shape (N_test, Sequence_Length, Features)
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled)
    """
    # X shape: [Batch, Time, Feats]
    n_train, t_train, c_train = X_train_raw.shape
    n_test, t_test, c_test = X_test_raw.shape
    
    # Flatten to [N*T, C] for scaling
    X_train_flat = X_train_raw.reshape(-1, c_train)
    X_test_flat = X_test_raw.reshape(-1, c_test)
    
    scaler = StandardScaler()
    # Fit on TRAIN only
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    # Transform TEST using train statistics
    X_test_scaled_flat = scaler.transform(X_test_flat)
    
    # Reshape back to [Batch, Time, Feats]
    X_train_scaled = X_train_scaled_flat.reshape(n_train, t_train, c_train)
    X_test_scaled = X_test_scaled_flat.reshape(n_test, t_test, c_test)
    
    return X_train_scaled, X_test_scaled

def load_seam_csv(
    file_path: str | os.PathLike,
    n_features: int = 18,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single seam CSV (18 numeric features + 1 label).

    Returns:
        data: float32 array of shape (T, n_features)
        labels: int64 array of shape (T,) with values in {0,1,2}
    """
    df = pd.read_csv(file_path, header=None)
    if df.shape[1] < n_features + 1:
        raise ValueError(
            f"Expected at least {n_features + 1} columns (features+label), got {df.shape[1]} in {file_path}"
        )

    data = df.iloc[:, :n_features].values.astype(np.float32)
    raw_labels = df.iloc[:, n_features].astype(str).str.strip()
    labels = raw_labels.map(LABEL_MAP).to_numpy(dtype=np.int64)
    if np.isnan(labels).any():
        bad = sorted(set(raw_labels[pd.isna(labels)].tolist()))
        raise ValueError(f"Unmapped labels in {file_path}: {bad}")
    return data, labels


def standardize_per_seam_full_fit(
    seam_data: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, StandardScaler]]:
    """Fit a scaler per seam on the full seam data and transform it.

    Note: This intentionally fits on the entire seam (including what may later become test windows),
    matching the requested behavior.
    """
    scaled: Dict[str, np.ndarray] = {}
    scalers: Dict[str, StandardScaler] = {}

    for seam_id, X in seam_data.items():
        if X.ndim != 2:
            raise ValueError(f"seam_data[{seam_id!r}] must be 2D (T,F), got shape {X.shape}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float32))
        scaled[seam_id] = X_scaled.astype(np.float32)
        scalers[seam_id] = scaler

    return scaled, scalers


def load_npz_dataset(npz_path: str | os.PathLike) -> Dict[str, np.ndarray]:
    """Load a prepared dataset from .npz produced by the seam preprocessing script."""
    npz_path = str(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
