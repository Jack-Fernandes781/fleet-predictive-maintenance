"""
Synthetic Fleet Telematics Data Generator

Generates realistic vehicle telematics data with component failure patterns
for Engine, Brakes, and Battery failures.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class FleetDataGenerator:
    """Generates synthetic telematics data for a fleet of vehicles."""

    # Realistic parameter ranges
    VEHICLE_CONFIGS = {
        'initial_mileage_range': (10000, 150000),
        'daily_mileage_range': (100, 400),
        'initial_engine_hours_range': (500, 5000),
        'initial_brake_pad_range': (8, 12),  # mm
        'initial_battery_age_days': (0, 730),  # 0-2 years
    }

    # Normal operating ranges
    NORMAL_RANGES = {
        'engine_temp': (180, 220),  # Fahrenheit
        'oil_pressure': (25, 65),   # PSI
        'battery_voltage': (12.4, 14.5),  # Volts
        'ambient_temp': (20, 90),   # Fahrenheit
        'load_weight': (0, 45000),  # lbs
        'idle_time_pct': (5, 40),   # percentage
    }

    # Failure thresholds
    FAILURE_THRESHOLDS = {
        'engine_temp_critical': 250,
        'oil_pressure_critical': 15,
        'battery_voltage_critical': 11.8,
        'brake_pad_critical': 2.5,  # mm
    }

    def __init__(self,
                 n_vehicles: int = 50,
                 start_date: str = "2023-01-01",
                 n_days: int = 365,
                 readings_per_day: int = 1,
                 random_seed: int = 42):
        """
        Initialize the data generator.

        Args:
            n_vehicles: Number of vehicles in the fleet
            start_date: Start date for data generation
            n_days: Number of days to generate data for
            readings_per_day: Number of readings per vehicle per day
            random_seed: Random seed for reproducibility
        """
        self.n_vehicles = n_vehicles
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.n_days = n_days
        self.readings_per_day = readings_per_day
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def _initialize_vehicle(self, vehicle_id: int) -> dict:
        """Initialize a vehicle with starting parameters."""
        return {
            'vehicle_id': f"V{vehicle_id:03d}",
            'mileage': np.random.uniform(*self.VEHICLE_CONFIGS['initial_mileage_range']),
            'engine_hours': np.random.uniform(*self.VEHICLE_CONFIGS['initial_engine_hours_range']),
            'brake_pad_thickness': np.random.uniform(*self.VEHICLE_CONFIGS['initial_brake_pad_range']),
            'battery_age_days': np.random.randint(*self.VEHICLE_CONFIGS['initial_battery_age_days']),
            'days_since_maintenance': np.random.randint(0, 90),
            'cumulative_hard_brakes': 0,
            'degradation_factor': np.random.uniform(0.8, 1.2),  # Vehicle-specific wear rate
        }

    def _generate_daily_reading(self,
                                vehicle: dict,
                                day_idx: int,
                                current_date: datetime) -> dict:
        """Generate a single daily reading for a vehicle."""

        # Daily variations
        daily_mileage = np.random.uniform(*self.VEHICLE_CONFIGS['daily_mileage_range'])
        daily_engine_hours = daily_mileage / 35  # Approximate avg speed

        # Update cumulative values
        vehicle['mileage'] += daily_mileage
        vehicle['engine_hours'] += daily_engine_hours
        vehicle['battery_age_days'] += 1
        vehicle['days_since_maintenance'] += 1

        # Brake pad wear (faster with heavy loads and hard braking)
        base_wear = 0.008 * vehicle['degradation_factor']  # mm per day
        vehicle['brake_pad_thickness'] = max(0, vehicle['brake_pad_thickness'] - base_wear)

        # Seasonal ambient temperature
        day_of_year = current_date.timetuple().tm_yday
        seasonal_temp = 55 + 35 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        ambient_temp = seasonal_temp + np.random.normal(0, 10)
        ambient_temp = np.clip(ambient_temp, -10, 110)

        # Generate sensor readings
        load_weight = np.random.uniform(*self.NORMAL_RANGES['load_weight'])
        idle_time_pct = np.random.uniform(*self.NORMAL_RANGES['idle_time_pct'])

        # Hard brake events (more likely with heavy loads)
        hard_brake_events = np.random.poisson(1 + load_weight / 20000)
        vehicle['cumulative_hard_brakes'] += hard_brake_events

        # Additional brake wear from hard braking
        vehicle['brake_pad_thickness'] = max(
            0,
            vehicle['brake_pad_thickness'] - hard_brake_events * 0.002
        )

        # Engine temperature (affected by ambient temp and load)
        base_engine_temp = np.random.uniform(*self.NORMAL_RANGES['engine_temp'])
        engine_temp = base_engine_temp + (ambient_temp - 55) * 0.3 + load_weight / 5000

        # Oil pressure (degrades with engine hours and high temps)
        engine_hour_factor = max(0, 1 - vehicle['engine_hours'] / 15000)
        base_oil_pressure = np.random.uniform(*self.NORMAL_RANGES['oil_pressure'])
        oil_pressure = base_oil_pressure * engine_hour_factor
        oil_pressure -= max(0, (engine_temp - 220) * 0.5)
        oil_pressure = max(5, oil_pressure)

        # Battery voltage (degrades with age, affected by extreme temps)
        battery_age_factor = max(0.7, 1 - vehicle['battery_age_days'] / 1500)
        base_voltage = np.random.uniform(*self.NORMAL_RANGES['battery_voltage'])
        battery_voltage = base_voltage * battery_age_factor

        # Extreme temps affect battery
        if ambient_temp < 20 or ambient_temp > 95:
            battery_voltage -= 0.3

        # Error codes (more likely with degraded components)
        error_probability = 0.05
        if engine_temp > 230:
            error_probability += 0.1
        if oil_pressure < 20:
            error_probability += 0.15
        if battery_voltage < 12.2:
            error_probability += 0.1
        if vehicle['brake_pad_thickness'] < 4:
            error_probability += 0.1

        error_code_count = np.random.poisson(error_probability * 3)

        # Determine failure (will occur in next 30 days)
        failure_type = self._determine_failure(
            engine_temp=engine_temp,
            oil_pressure=oil_pressure,
            battery_voltage=battery_voltage,
            brake_pad_thickness=vehicle['brake_pad_thickness'],
            engine_hours=vehicle['engine_hours'],
            battery_age_days=vehicle['battery_age_days'],
            hard_brake_events=hard_brake_events,
            load_weight=load_weight,
            days_since_maintenance=vehicle['days_since_maintenance']
        )

        return {
            'vehicle_id': vehicle['vehicle_id'],
            'timestamp': current_date,
            'mileage': round(vehicle['mileage'], 1),
            'engine_hours': round(vehicle['engine_hours'], 1),
            'engine_temp': round(engine_temp, 1),
            'oil_pressure': round(oil_pressure, 1),
            'battery_voltage': round(battery_voltage, 2),
            'brake_pad_thickness': round(vehicle['brake_pad_thickness'], 2),
            'error_code_count': error_code_count,
            'ambient_temp': round(ambient_temp, 1),
            'load_weight': round(load_weight, 0),
            'idle_time_pct': round(idle_time_pct, 1),
            'hard_brake_events': hard_brake_events,
            'days_since_maintenance': vehicle['days_since_maintenance'],
            'failure_within_30_days': failure_type
        }

    def _determine_failure(self,
                          engine_temp: float,
                          oil_pressure: float,
                          battery_voltage: float,
                          brake_pad_thickness: float,
                          engine_hours: float,
                          battery_age_days: int,
                          hard_brake_events: int,
                          load_weight: float,
                          days_since_maintenance: int) -> str:
        """
        Determine if a failure will occur within 30 days based on current readings.
        Uses probabilistic logic with realistic failure patterns.
        """

        # Calculate failure probabilities
        engine_failure_prob = 0.0
        brake_failure_prob = 0.0
        battery_failure_prob = 0.0

        # Engine failure probability
        if engine_temp > 240:
            engine_failure_prob += 0.3
        elif engine_temp > 230:
            engine_failure_prob += 0.15

        if oil_pressure < 18:
            engine_failure_prob += 0.35
        elif oil_pressure < 22:
            engine_failure_prob += 0.15

        if engine_hours > 10000:
            engine_failure_prob += 0.1

        if days_since_maintenance > 120:
            engine_failure_prob += 0.1

        # Brake failure probability
        if brake_pad_thickness < 2:
            brake_failure_prob += 0.5
        elif brake_pad_thickness < 3:
            brake_failure_prob += 0.25
        elif brake_pad_thickness < 4:
            brake_failure_prob += 0.1

        if hard_brake_events > 5:
            brake_failure_prob += 0.1

        if load_weight > 40000:
            brake_failure_prob += 0.05

        # Battery failure probability
        if battery_voltage < 11.8:
            battery_failure_prob += 0.4
        elif battery_voltage < 12.0:
            battery_failure_prob += 0.2
        elif battery_voltage < 12.2:
            battery_failure_prob += 0.1

        if battery_age_days > 1000:
            battery_failure_prob += 0.15
        elif battery_age_days > 730:
            battery_failure_prob += 0.08

        # Add randomness and determine failure
        engine_failure_prob = min(0.95, engine_failure_prob + np.random.uniform(-0.05, 0.05))
        brake_failure_prob = min(0.95, brake_failure_prob + np.random.uniform(-0.05, 0.05))
        battery_failure_prob = min(0.95, battery_failure_prob + np.random.uniform(-0.05, 0.05))

        # Determine which failure (if any) will occur
        failures = []
        if np.random.random() < engine_failure_prob:
            failures.append(('Engine', engine_failure_prob))
        if np.random.random() < brake_failure_prob:
            failures.append(('Brakes', brake_failure_prob))
        if np.random.random() < battery_failure_prob:
            failures.append(('Battery', battery_failure_prob))

        if not failures:
            return 'None'

        # Return the most likely failure
        failures.sort(key=lambda x: x[1], reverse=True)
        return failures[0][0]

    def generate(self) -> pd.DataFrame:
        """Generate the complete dataset."""

        print(f"Generating telematics data for {self.n_vehicles} vehicles over {self.n_days} days...")

        all_readings = []

        for v_idx in range(self.n_vehicles):
            vehicle = self._initialize_vehicle(v_idx + 1)

            for day_idx in range(self.n_days):
                current_date = self.start_date + timedelta(days=day_idx)

                for _ in range(self.readings_per_day):
                    reading = self._generate_daily_reading(vehicle, day_idx, current_date)
                    all_readings.append(reading)

                # Simulate maintenance events (resets some parameters)
                if vehicle['days_since_maintenance'] > 90 and np.random.random() < 0.1:
                    vehicle['days_since_maintenance'] = 0
                    vehicle['brake_pad_thickness'] = min(12, vehicle['brake_pad_thickness'] + 5)

            if (v_idx + 1) % 10 == 0:
                print(f"  Processed {v_idx + 1}/{self.n_vehicles} vehicles...")

        df = pd.DataFrame(all_readings)

        # Print summary statistics
        print(f"\nDataset generated: {len(df):,} records")
        print(f"\nFailure distribution:")
        print(df['failure_within_30_days'].value_counts())
        print(f"\nFailure rate: {(df['failure_within_30_days'] != 'None').mean():.1%}")

        return df

    def save(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Save the generated data to CSV."""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "data" / "raw" / "fleet_telematics.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nData saved to: {output_path}")

        return str(output_path)


def generate_fleet_data(n_vehicles: int = 50,
                        n_days: int = 365,
                        random_seed: int = 42,
                        output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to generate and save fleet telematics data.

    Args:
        n_vehicles: Number of vehicles in the fleet
        n_days: Number of days to generate data for
        random_seed: Random seed for reproducibility
        output_path: Optional path to save the data

    Returns:
        DataFrame with generated telematics data
    """
    generator = FleetDataGenerator(
        n_vehicles=n_vehicles,
        n_days=n_days,
        random_seed=random_seed
    )

    df = generator.generate()

    if output_path is not None:
        generator.save(df, output_path)
    else:
        generator.save(df)

    return df


if __name__ == "__main__":
    # Generate sample data when run directly
    df = generate_fleet_data()
    print("\nSample records:")
    print(df.head(10))
