"""
Prediction Module

Handles inference for predicting vehicle component failures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import joblib


class FailurePredictor:
    """Predicts vehicle component failures using trained model."""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize the predictor by loading model artifacts.

        Args:
            model_dir: Directory containing saved model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False

    def load(self) -> None:
        """Load all model artifacts."""
        print(f"Loading model from {self.model_dir}...")

        # Find model file
        model_files = list(self.model_dir.glob("*_model.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model file found in {self.model_dir}")

        self.model = joblib.load(model_files[0])
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        self.label_encoder = joblib.load(self.model_dir / "label_encoder.joblib")
        self.feature_names = joblib.load(self.model_dir / "feature_names.joblib")

        self.is_loaded = True
        print(f"Model loaded successfully: {model_files[0].name}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on raw features.

        Args:
            X: Feature array (unscaled)

        Returns:
            Array of predicted class labels
        """
        if not self.is_loaded:
            self.load()

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred_encoded = self.model.predict(X_scaled)

        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> pd.DataFrame:
        """
        Get prediction probabilities for each class.

        Args:
            X: Feature array (unscaled)

        Returns:
            DataFrame with probabilities for each class
        """
        if not self.is_loaded:
            self.load()

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)
            return pd.DataFrame(
                proba,
                columns=self.label_encoder.classes_
            )
        else:
            raise ValueError("Model does not support probability predictions")

    def predict_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction from a dictionary of feature values.

        Args:
            data: Dictionary with feature names as keys

        Returns:
            Dictionary with prediction and probabilities
        """
        if not self.is_loaded:
            self.load()

        # Create feature array in correct order
        X = np.array([[data.get(f, 0) for f in self.feature_names]])

        # Get prediction and probabilities
        prediction = self.predict(X)[0]
        proba_df = self.predict_proba(X)

        result = {
            'prediction': prediction,
            'probabilities': proba_df.iloc[0].to_dict(),
            'confidence': float(proba_df.iloc[0].max()),
            'risk_level': self._calculate_risk_level(proba_df.iloc[0])
        }

        return result

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a DataFrame.

        Args:
            df: DataFrame with feature columns

        Returns:
            DataFrame with predictions and probabilities added
        """
        if not self.is_loaded:
            self.load()

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Extract features in correct order
        X = df[self.feature_names].values

        # Get predictions
        df = df.copy()
        df['predicted_failure'] = self.predict(X)

        # Get probabilities
        proba_df = self.predict_proba(X)
        for col in proba_df.columns:
            df[f'prob_{col}'] = proba_df[col].values

        # Add risk level
        df['risk_level'] = proba_df.apply(self._calculate_risk_level, axis=1)

        return df

    def _calculate_risk_level(self, proba: pd.Series) -> str:
        """
        Calculate risk level based on failure probabilities.

        Args:
            proba: Series with probabilities for each class

        Returns:
            Risk level string (Low, Medium, High, Critical)
        """
        # If 'None' class exists, use 1 - P(None) as failure probability
        # Otherwise, use max probability of critical components (Engine, Brakes)
        if 'None' in proba.index:
            failure_prob = 1 - proba.get('None', 0)
        else:
            # Use max of critical failure probabilities
            critical_failures = ['Engine', 'Brakes']
            critical_probs = [proba.get(f, 0) for f in critical_failures if f in proba.index]
            failure_prob = max(critical_probs) if critical_probs else proba.max()

        if failure_prob < 0.2:
            return 'Low'
        elif failure_prob < 0.4:
            return 'Medium'
        elif failure_prob < 0.7:
            return 'High'
        else:
            return 'Critical'

    def predict_vehicle(self,
                       vehicle_data: pd.DataFrame,
                       vehicle_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict failure for a specific vehicle.

        Args:
            vehicle_data: DataFrame with vehicle telematics
            vehicle_id: Optional vehicle ID to filter

        Returns:
            Dictionary with prediction details
        """
        if vehicle_id:
            vehicle_data = vehicle_data[vehicle_data['vehicle_id'] == vehicle_id]

        if len(vehicle_data) == 0:
            raise ValueError(f"No data found for vehicle {vehicle_id}")

        # Get latest reading
        latest = vehicle_data.sort_values('timestamp').iloc[-1]

        # Make prediction
        result = self.predict_from_dict(latest.to_dict())

        result['vehicle_id'] = vehicle_id or latest.get('vehicle_id', 'Unknown')
        result['timestamp'] = str(latest.get('timestamp', ''))

        return result

    def get_fleet_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction summary for entire fleet.

        Args:
            df: DataFrame with all vehicle data

        Returns:
            Summary DataFrame grouped by vehicle
        """
        # Get predictions for all data
        df_pred = self.predict_from_dataframe(df)

        # Build aggregation dict dynamically based on available columns
        agg_dict = {
            'timestamp': 'max',
            'predicted_failure': 'last',
            'risk_level': 'last',
            'mileage': 'last',
            'days_since_maintenance': 'last'
        }

        # Add probability columns that exist
        prob_cols = [c for c in df_pred.columns if c.startswith('prob_')]
        for col in prob_cols:
            agg_dict[col] = 'last'

        # Group by vehicle and get latest prediction
        summary = df_pred.groupby('vehicle_id').agg(agg_dict).reset_index()

        # Sort by risk (Critical first)
        risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        summary['risk_sort'] = summary['risk_level'].map(risk_order)
        summary = summary.sort_values('risk_sort').drop('risk_sort', axis=1)

        return summary


def load_predictor(model_dir: str = "models") -> FailurePredictor:
    """
    Convenience function to load a predictor.

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        Loaded FailurePredictor instance
    """
    predictor = FailurePredictor(model_dir)
    predictor.load()
    return predictor


if __name__ == "__main__":
    # Test prediction
    from data_generator import generate_fleet_data
    from feature_engineering import engineer_features

    # Generate test data
    print("Generating test data...")
    df = generate_fleet_data(n_vehicles=5, n_days=30, random_seed=99)
    df = engineer_features(df)

    # Load predictor
    try:
        predictor = load_predictor()

        # Predict for one vehicle
        result = predictor.predict_vehicle(df, vehicle_id='V001')
        print(f"\nPrediction for V001:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Probabilities: {result['probabilities']}")

        # Fleet summary
        print("\nFleet Summary:")
        summary = predictor.get_fleet_summary(df)
        print(summary)

    except FileNotFoundError as e:
        print(f"\nModel not found. Please train the model first using: python main.py train")
