"""
Feature Engineering Module

Creates derived features from raw telematics data to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Creates engineered features from raw telematics data."""

    # Features to compute rolling statistics for
    ROLLING_FEATURES = [
        'engine_temp',
        'oil_pressure',
        'battery_voltage',
        'error_code_count',
        'hard_brake_events',
        'load_weight'
    ]

    # Windows for rolling calculations (in days)
    ROLLING_WINDOWS = [7, 14, 30]

    def __init__(self):
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.

        Args:
            df: Raw telematics DataFrame

        Returns:
            DataFrame with original and engineered features
        """
        print("Engineering features...")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by vehicle and time for rolling calculations
        df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)

        # Create all feature types
        df = self._create_rolling_features(df)
        df = self._create_rate_of_change_features(df)
        df = self._create_threshold_features(df)
        df = self._create_interaction_features(df)
        df = self._create_time_features(df)
        df = self._create_cumulative_features(df)

        # Fill any NaN values created by rolling calculations
        df = self._handle_missing_values(df)

        print(f"Created {len([c for c in df.columns if c not in ['vehicle_id', 'timestamp', 'failure_within_30_days']])} features")

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean and std features."""
        print("  - Creating rolling statistics...")

        for feature in self.ROLLING_FEATURES:
            for window in self.ROLLING_WINDOWS:
                # Rolling mean
                df[f'{feature}_rolling_mean_{window}d'] = (
                    df.groupby('vehicle_id')[feature]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

                # Rolling std
                df[f'{feature}_rolling_std_{window}d'] = (
                    df.groupby('vehicle_id')[feature]
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )

                # Rolling max
                df[f'{feature}_rolling_max_{window}d'] = (
                    df.groupby('vehicle_id')[feature]
                    .transform(lambda x: x.rolling(window, min_periods=1).max())
                )

                # Rolling min
                df[f'{feature}_rolling_min_{window}d'] = (
                    df.groupby('vehicle_id')[feature]
                    .transform(lambda x: x.rolling(window, min_periods=1).min())
                )

        return df

    def _create_rate_of_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rate of change (trend) features."""
        print("  - Creating rate of change features...")

        trend_features = ['battery_voltage', 'oil_pressure', 'brake_pad_thickness', 'engine_temp']

        for feature in trend_features:
            # Daily change
            df[f'{feature}_daily_change'] = (
                df.groupby('vehicle_id')[feature]
                .transform(lambda x: x.diff())
            )

            # 7-day trend (slope)
            df[f'{feature}_7d_trend'] = (
                df.groupby('vehicle_id')[feature]
                .transform(lambda x: x.diff(7) / 7)
            )

            # Acceleration (change in rate of change)
            df[f'{feature}_acceleration'] = (
                df.groupby('vehicle_id')[f'{feature}_daily_change']
                .transform(lambda x: x.diff())
            )

        return df

    def _create_threshold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary features for critical thresholds."""
        print("  - Creating threshold features...")

        # Engine-related thresholds
        df['engine_temp_warning'] = (df['engine_temp'] > 230).astype(int)
        df['engine_temp_critical'] = (df['engine_temp'] > 245).astype(int)
        df['oil_pressure_warning'] = (df['oil_pressure'] < 22).astype(int)
        df['oil_pressure_critical'] = (df['oil_pressure'] < 18).astype(int)

        # Battery thresholds
        df['battery_voltage_warning'] = (df['battery_voltage'] < 12.2).astype(int)
        df['battery_voltage_critical'] = (df['battery_voltage'] < 11.8).astype(int)

        # Brake thresholds
        df['brake_pad_warning'] = (df['brake_pad_thickness'] < 4).astype(int)
        df['brake_pad_critical'] = (df['brake_pad_thickness'] < 2.5).astype(int)

        # Maintenance overdue
        df['maintenance_overdue'] = (df['days_since_maintenance'] > 90).astype(int)
        df['maintenance_critical'] = (df['days_since_maintenance'] > 120).astype(int)

        # Error code flags
        df['has_error_codes'] = (df['error_code_count'] > 0).astype(int)
        df['multiple_errors'] = (df['error_code_count'] > 2).astype(int)

        # Combined risk flags
        df['engine_risk_flag'] = (
            (df['engine_temp_warning'] == 1) | (df['oil_pressure_warning'] == 1)
        ).astype(int)

        df['high_risk_flag'] = (
            (df['engine_temp_critical'] == 1) |
            (df['oil_pressure_critical'] == 1) |
            (df['battery_voltage_critical'] == 1) |
            (df['brake_pad_critical'] == 1)
        ).astype(int)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        print("  - Creating interaction features...")

        # Load-related interactions
        df['load_mileage_interaction'] = df['load_weight'] * df['mileage'] / 1e9
        df['load_brake_wear'] = df['load_weight'] * (12 - df['brake_pad_thickness'])

        # Temperature interactions
        df['temp_differential'] = df['engine_temp'] - df['ambient_temp']
        df['cold_start_stress'] = ((df['ambient_temp'] < 32) & (df['engine_temp'] > 200)).astype(int)

        # Engine stress
        df['engine_stress_index'] = (
            (df['engine_temp'] / 200) *
            (65 / (df['oil_pressure'] + 1)) *
            (df['engine_hours'] / 5000)
        )

        # Battery health composite
        df['battery_health_index'] = (
            df['battery_voltage'] / 14.5 *
            np.exp(-df.groupby('vehicle_id').cumcount() / 1000)
        )

        # Brake wear rate
        df['brake_wear_rate'] = (
            df['hard_brake_events'] * df['load_weight'] / 10000
        )

        # Overall vehicle health score (normalized 0-100)
        df['health_score'] = 100 - (
            df['engine_temp_warning'] * 15 +
            df['oil_pressure_warning'] * 20 +
            df['battery_voltage_warning'] * 15 +
            df['brake_pad_warning'] * 15 +
            df['maintenance_overdue'] * 10 +
            df['multiple_errors'] * 10 +
            np.clip(df['error_code_count'] * 3, 0, 15)
        )
        df['health_score'] = df['health_score'].clip(0, 100)

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        print("  - Creating time-based features...")

        # Day of week (workday patterns)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Month (seasonal patterns)
        df['month'] = df['timestamp'].dt.month

        # Season encoding
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,   # Winter
            3: 1, 4: 1, 5: 1,    # Spring
            6: 2, 7: 2, 8: 2,    # Summer
            9: 3, 10: 3, 11: 3   # Fall
        })

        # Days in service (from first reading)
        df['days_in_service'] = df.groupby('vehicle_id').cumcount()

        return df

    def _create_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cumulative features."""
        print("  - Creating cumulative features...")

        # Cumulative error codes
        df['cumulative_errors'] = (
            df.groupby('vehicle_id')['error_code_count']
            .transform('cumsum')
        )

        # Cumulative hard brake events
        df['cumulative_hard_brakes'] = (
            df.groupby('vehicle_id')['hard_brake_events']
            .transform('cumsum')
        )

        # Cumulative warning days
        df['cumulative_warning_days'] = (
            df.groupby('vehicle_id')['engine_risk_flag']
            .transform('cumsum')
        )

        # Rolling warning frequency (last 30 days)
        df['recent_warning_frequency'] = (
            df.groupby('vehicle_id')['high_risk_flag']
            .transform(lambda x: x.rolling(30, min_periods=1).mean())
        )

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values created by feature engineering."""
        print("  - Handling missing values...")

        # Fill NaN with 0 for rate of change features (first observations)
        rate_cols = [c for c in df.columns if 'change' in c or 'trend' in c or 'acceleration' in c]
        df[rate_cols] = df[rate_cols].fillna(0)

        # Fill rolling std with 0 (not enough data for std calculation)
        std_cols = [c for c in df.columns if '_std_' in c]
        df[std_cols] = df[std_cols].fillna(0)

        # Forward fill any remaining NaN values within each vehicle
        df = df.groupby('vehicle_id').apply(
            lambda x: x.ffill().bfill()
        ).reset_index(drop=True)

        # Final fallback: fill with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature names (excluding identifiers and target)."""
        exclude_cols = ['vehicle_id', 'timestamp', 'failure_within_30_days']
        return [c for c in df.columns if c not in exclude_cols]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer features.

    Args:
        df: Raw telematics DataFrame

    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.create_features(df)


if __name__ == "__main__":
    # Test with sample data
    from data_generator import generate_fleet_data

    # Generate small sample
    df = generate_fleet_data(n_vehicles=5, n_days=60, random_seed=42)

    # Engineer features
    df_features = engineer_features(df)

    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"After feature engineering: {len(df_features.columns)}")
    print(f"\nNew features sample:")
    print(df_features.columns.tolist()[-20:])
