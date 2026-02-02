"""
Data Preprocessing Module

Handles data cleaning, scaling, class imbalance, and train/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class DataPreprocessor:
    """Handles all preprocessing steps for the fleet maintenance data."""

    def __init__(self,
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 random_state: int = 42,
                 use_smote: bool = True):
        """
        Initialize the preprocessor.

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
            use_smote: Whether to use SMOTE for handling class imbalance
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_smote = use_smote

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target from DataFrame.

        Args:
            df: DataFrame with features and target column

        Returns:
            Tuple of (features array, target array, feature names)
        """
        # Identify feature columns (exclude identifiers and target)
        exclude_cols = ['vehicle_id', 'timestamp', 'failure_within_30_days']
        self.feature_names = [c for c in df.columns if c not in exclude_cols]

        # Extract features and target
        X = df[self.feature_names].values
        y = df['failure_within_30_days'].values

        return X, y, self.feature_names

    def split_data(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        Uses time-aware splitting if DataFrame with timestamps is provided.

        Args:
            X: Feature array
            y: Target array
            df: Optional DataFrame for time-aware splitting

        Returns:
            Dictionary with train/val/test splits
        """
        print("Splitting data into train/val/test sets...")

        if df is not None and 'timestamp' in df.columns:
            # Time-aware split: use chronological ordering
            sorted_indices = df['timestamp'].argsort().values

            n_samples = len(sorted_indices)
            train_end = int(n_samples * (1 - self.test_size - self.val_size))
            val_end = int(n_samples * (1 - self.test_size))

            train_idx = sorted_indices[:train_end]
            val_idx = sorted_indices[train_end:val_end]
            test_idx = sorted_indices[val_end:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        else:
            # Random split with stratification
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            val_ratio = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=y_temp
            )

        print(f"  Train set: {len(X_train):,} samples")
        print(f"  Val set:   {len(X_val):,} samples")
        print(f"  Test set:  {len(X_test):,} samples")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def encode_labels(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Encode string labels to integers.

        Args:
            y: Array of string labels
            fit: Whether to fit the encoder (True for training data)

        Returns:
            Array of integer labels
        """
        if fit:
            return self.label_encoder.fit_transform(y)
        return self.label_encoder.transform(y)

    def decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(y)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler and transform features."""
        self.is_fitted = True
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming.")
        return self.scaler.transform(X)

    def apply_smote(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    sampling_strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance.

        Args:
            X: Feature array
            y: Target array (encoded)
            sampling_strategy: SMOTE sampling strategy

        Returns:
            Resampled X and y arrays
        """
        if not self.use_smote:
            return X, y

        print("Applying SMOTE for class balancing...")
        print(f"  Before SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}")

        smote = SMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print(f"  After SMOTE:  {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

        return X_resampled, y_resampled

    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced classes.

        Args:
            y: Encoded target array

        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)

        return dict(zip(classes, weights))

    def preprocess_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Raw DataFrame with features and target

        Returns:
            Dictionary with all preprocessed data and metadata
        """
        print("\n" + "="*50)
        print("PREPROCESSING PIPELINE")
        print("="*50)

        # Prepare features and target
        X, y, feature_names = self.prepare_data(df)
        print(f"\nTotal samples: {len(X):,}")
        print(f"Total features: {len(feature_names)}")

        # Encode labels
        y_encoded = self.encode_labels(y, fit=True)
        print(f"\nClasses: {list(self.label_encoder.classes_)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Split data
        splits = self.split_data(X, y_encoded, df)

        # Scale features (fit on train only)
        print("\nScaling features...")
        splits['X_train'] = self.fit_transform(splits['X_train'])
        splits['X_val'] = self.transform(splits['X_val'])
        splits['X_test'] = self.transform(splits['X_test'])

        # Apply SMOTE to training data only
        if self.use_smote:
            splits['X_train'], splits['y_train'] = self.apply_smote(
                splits['X_train'],
                splits['y_train']
            )

        # Calculate class weights (on original distribution)
        class_weights = self.get_class_weights(y_encoded)
        print(f"\nClass weights: {class_weights}")

        return {
            'splits': splits,
            'feature_names': feature_names,
            'class_weights': class_weights,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'classes': list(self.label_encoder.classes_)
        }

    def save(self, output_dir: str) -> None:
        """Save the preprocessor artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, output_path / 'scaler.joblib')
        joblib.dump(self.label_encoder, output_path / 'label_encoder.joblib')
        joblib.dump(self.feature_names, output_path / 'feature_names.joblib')

        print(f"Preprocessor saved to: {output_path}")

    def load(self, input_dir: str) -> None:
        """Load preprocessor artifacts."""
        input_path = Path(input_dir)

        self.scaler = joblib.load(input_path / 'scaler.joblib')
        self.label_encoder = joblib.load(input_path / 'label_encoder.joblib')
        self.feature_names = joblib.load(input_path / 'feature_names.joblib')
        self.is_fitted = True

        print(f"Preprocessor loaded from: {input_path}")


def preprocess_data(df: pd.DataFrame,
                    use_smote: bool = True,
                    random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience function for preprocessing.

    Args:
        df: DataFrame with features and target
        use_smote: Whether to apply SMOTE
        random_state: Random seed

    Returns:
        Preprocessed data dictionary
    """
    preprocessor = DataPreprocessor(use_smote=use_smote, random_state=random_state)
    result = preprocessor.preprocess_pipeline(df)
    result['preprocessor'] = preprocessor
    return result


if __name__ == "__main__":
    # Test preprocessing
    from data_generator import generate_fleet_data
    from feature_engineering import engineer_features

    # Generate and engineer features
    df = generate_fleet_data(n_vehicles=10, n_days=90, random_seed=42)
    df = engineer_features(df)

    # Preprocess
    result = preprocess_data(df)

    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"\nTrain shape: {result['splits']['X_train'].shape}")
    print(f"Val shape:   {result['splits']['X_val'].shape}")
    print(f"Test shape:  {result['splits']['X_test'].shape}")
