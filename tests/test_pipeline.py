"""
Unit Tests for Fleet Predictive Maintenance Pipeline

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataGenerator:
    """Tests for data generation module."""

    def test_generate_data_shape(self):
        """Test that generated data has correct shape."""
        from data_generator import generate_fleet_data

        df = generate_fleet_data(n_vehicles=5, n_days=30, random_seed=42)

        # Should have 5 vehicles * 30 days = 150 records
        assert len(df) == 150
        assert df['vehicle_id'].nunique() == 5

    def test_generate_data_columns(self):
        """Test that generated data has required columns."""
        from data_generator import generate_fleet_data

        df = generate_fleet_data(n_vehicles=2, n_days=10, random_seed=42)

        required_columns = [
            'vehicle_id', 'timestamp', 'mileage', 'engine_hours',
            'engine_temp', 'oil_pressure', 'battery_voltage',
            'brake_pad_thickness', 'error_code_count', 'ambient_temp',
            'load_weight', 'idle_time_pct', 'hard_brake_events',
            'days_since_maintenance', 'failure_within_30_days'
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_generate_data_values_in_range(self):
        """Test that generated values are within realistic ranges."""
        from data_generator import generate_fleet_data

        df = generate_fleet_data(n_vehicles=5, n_days=30, random_seed=42)

        # Engine temp should be positive and below 300
        assert df['engine_temp'].min() > 0
        assert df['engine_temp'].max() < 300

        # Oil pressure should be positive
        assert df['oil_pressure'].min() >= 0

        # Battery voltage should be reasonable
        assert df['battery_voltage'].min() > 10
        assert df['battery_voltage'].max() < 16

        # Brake pad thickness should be non-negative
        assert df['brake_pad_thickness'].min() >= 0

    def test_failure_types(self):
        """Test that failure types are valid."""
        from data_generator import generate_fleet_data

        df = generate_fleet_data(n_vehicles=10, n_days=60, random_seed=42)

        valid_failures = ['None', 'Engine', 'Brakes', 'Battery']
        for failure_type in df['failure_within_30_days'].unique():
            assert failure_type in valid_failures

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        from data_generator import generate_fleet_data

        df1 = generate_fleet_data(n_vehicles=5, n_days=10, random_seed=42)
        df2 = generate_fleet_data(n_vehicles=5, n_days=10, random_seed=42)

        pd.testing.assert_frame_equal(df1, df2)


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    def test_feature_creation(self):
        """Test that features are created."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features

        df = generate_fleet_data(n_vehicles=3, n_days=30, random_seed=42)
        original_cols = len(df.columns)

        df_features = engineer_features(df)

        # Should have more columns after feature engineering
        assert len(df_features.columns) > original_cols

    def test_rolling_features(self):
        """Test that rolling features are created."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features

        df = generate_fleet_data(n_vehicles=2, n_days=30, random_seed=42)
        df_features = engineer_features(df)

        # Check for rolling mean features
        rolling_cols = [c for c in df_features.columns if 'rolling_mean' in c]
        assert len(rolling_cols) > 0

    def test_no_missing_values(self):
        """Test that feature engineering handles missing values."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features

        df = generate_fleet_data(n_vehicles=3, n_days=30, random_seed=42)
        df_features = engineer_features(df)

        # Check for NaN in numeric columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        assert not df_features[numeric_cols].isna().any().any()


class TestPreprocessing:
    """Tests for preprocessing module."""

    def test_data_splitting(self):
        """Test that data is split correctly."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features
        from preprocessing import preprocess_data

        df = generate_fleet_data(n_vehicles=5, n_days=60, random_seed=42)
        df = engineer_features(df)

        result = preprocess_data(df, use_smote=False)
        splits = result['splits']

        # Check that splits exist
        assert 'X_train' in splits
        assert 'X_val' in splits
        assert 'X_test' in splits

        # Check shapes
        total_samples = len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test'])
        assert total_samples == len(df)

    def test_label_encoding(self):
        """Test that labels are properly encoded."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features
        from preprocessing import preprocess_data

        df = generate_fleet_data(n_vehicles=5, n_days=30, random_seed=42)
        df = engineer_features(df)

        result = preprocess_data(df, use_smote=False)

        # Labels should be integers
        assert result['splits']['y_train'].dtype in [np.int32, np.int64]

    def test_scaling(self):
        """Test that features are scaled."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features
        from preprocessing import preprocess_data

        df = generate_fleet_data(n_vehicles=5, n_days=30, random_seed=42)
        df = engineer_features(df)

        result = preprocess_data(df, use_smote=False)

        # After scaling, mean should be close to 0 and std close to 1
        X_train = result['splits']['X_train']
        assert np.abs(X_train.mean()) < 1  # Mean close to 0
        assert np.abs(X_train.std() - 1) < 1  # Std close to 1


class TestModels:
    """Tests for model training module."""

    def test_model_training(self):
        """Test that models can be trained."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features
        from preprocessing import preprocess_data
        from models import ModelTrainer

        # Use small dataset for speed
        df = generate_fleet_data(n_vehicles=10, n_days=60, random_seed=42)
        df = engineer_features(df)
        preprocessed = preprocess_data(df, use_smote=False)

        splits = preprocessed['splits']
        trainer = ModelTrainer(class_weights=preprocessed['class_weights'])

        # Train just one model for speed
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=100, random_state=42)

        results = trainer.train_model(
            model,
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val']
        )

        # Check results
        assert 'val_accuracy' in results
        assert results['val_accuracy'] > 0
        assert results['val_accuracy'] <= 1

    def test_business_impact_calculation(self):
        """Test business impact calculation."""
        from models import ModelTrainer

        trainer = ModelTrainer()
        trainer.best_model_name = "Test"

        # Create sample predictions
        y_test = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # None=0, others=1,2,3
        y_pred = np.array([0, 1, 1, 0, 2, 0, 3, 3])
        class_names = ['None', 'Battery', 'Brakes', 'Engine']

        impact = trainer.calculate_business_impact(y_test, y_pred, class_names)

        assert 'total_savings' in impact
        assert 'true_positives' in impact
        assert 'false_negatives' in impact


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline(self):
        """Test the complete pipeline from data generation to prediction."""
        from data_generator import generate_fleet_data
        from feature_engineering import engineer_features
        from preprocessing import preprocess_data
        from models import train_and_evaluate

        # Generate small dataset
        df = generate_fleet_data(n_vehicles=10, n_days=30, random_seed=42)

        # Engineer features
        df = engineer_features(df)

        # Preprocess
        preprocessed = preprocess_data(df, use_smote=True)

        # Train (using default models)
        results = train_and_evaluate(preprocessed, output_dir='models')

        # Check results
        assert results['best_model_name'] is not None
        assert results['test_results']['accuracy'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
