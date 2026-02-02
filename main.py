#!/usr/bin/env python
"""
Fleet Predictive Maintenance - CLI Entry Point

Usage:
    python main.py generate    - Generate synthetic telematics data
    python main.py train       - Train models on generated data
    python main.py predict     - Make predictions for the fleet
    python main.py pipeline    - Run complete pipeline (generate + train + predict)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def generate_data(args):
    """Generate synthetic fleet telematics data."""
    from src.data_generator import generate_fleet_data

    print("\n" + "="*60)
    print("GENERATING SYNTHETIC FLEET TELEMATICS DATA")
    print("="*60)

    df = generate_fleet_data(
        n_vehicles=args.n_vehicles,
        n_days=args.n_days,
        random_seed=args.seed,
        output_path=args.output
    )

    print(f"\nGenerated {len(df):,} records for {args.n_vehicles} vehicles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def train_models(args):
    """Train ML models on the data."""
    import pandas as pd
    from src.feature_engineering import engineer_features
    from src.preprocessing import preprocess_data
    from src.models import train_and_evaluate

    print("\n" + "="*60)
    print("TRAINING PREDICTIVE MAINTENANCE MODELS")
    print("="*60)

    # Load data
    data_path = args.data or "data/raw/fleet_telematics.csv"
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records")

    # Feature engineering
    print("\n" + "-"*40)
    df = engineer_features(df)

    # Preprocessing
    print("\n" + "-"*40)
    preprocessed = preprocess_data(df, use_smote=not args.no_smote)

    # Save preprocessor
    preprocessed['preprocessor'].save(args.model_dir)

    # Train and evaluate
    print("\n" + "-"*40)
    results = train_and_evaluate(preprocessed, output_dir=args.model_dir)

    # Save processed data for EDA
    processed_path = Path("data/processed/fleet_telematics_processed.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to: {processed_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest model: {results['best_model_name']}")
    print(f"Test F1 (macro): {results['test_results']['f1_macro']:.3f}")
    print(f"Estimated savings: ${results['business_impact']['total_savings']:,.0f}")

    return results


def predict(args):
    """Make predictions for fleet vehicles."""
    import pandas as pd
    from src.feature_engineering import engineer_features
    from src.predict import load_predictor

    print("\n" + "="*60)
    print("FLEET FAILURE PREDICTIONS")
    print("="*60)

    # Load data
    data_path = args.data or "data/raw/fleet_telematics.csv"
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Feature engineering
    df = engineer_features(df)

    # Load predictor
    predictor = load_predictor(args.model_dir)

    if args.vehicle_id:
        # Predict for specific vehicle
        result = predictor.predict_vehicle(df, vehicle_id=args.vehicle_id)

        print(f"\nPrediction for {result['vehicle_id']}:")
        print(f"  Timestamp:  {result['timestamp']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Risk Level: {result['risk_level']}")
        print("\n  Failure Probabilities:")
        for failure_type, prob in sorted(result['probabilities'].items()):
            print(f"    {failure_type}: {prob:.1%}")
    else:
        # Fleet summary
        summary = predictor.get_fleet_summary(df)

        print("\nFleet Risk Summary:")
        print("-" * 80)
        print(summary.to_string(index=False))

        # Risk distribution
        print("\nRisk Level Distribution:")
        print(summary['risk_level'].value_counts().to_string())

        # Vehicles needing attention
        high_risk = summary[summary['risk_level'].isin(['High', 'Critical'])]
        if len(high_risk) > 0:
            print(f"\n[WARNING] {len(high_risk)} vehicles require immediate attention!")
            for _, row in high_risk.iterrows():
                print(f"   - {row['vehicle_id']}: {row['predicted_failure']} ({row['risk_level']})")


def run_pipeline(args):
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print("RUNNING COMPLETE PREDICTIVE MAINTENANCE PIPELINE")
    print("="*60)

    # Step 1: Generate data
    args.output = None  # Use default path
    generate_data(args)

    # Step 2: Train models
    args.data = None  # Use default path
    args.no_smote = False
    train_models(args)

    # Step 3: Make predictions
    args.vehicle_id = None
    predict(args)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Fleet Predictive Maintenance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate                    Generate synthetic data
  python main.py generate -n 100 -d 180     Generate for 100 vehicles, 180 days
  python main.py train                       Train models
  python main.py predict                     Get fleet predictions
  python main.py predict -v V001            Predict for specific vehicle
  python main.py pipeline                    Run complete pipeline
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic telematics data')
    gen_parser.add_argument('-n', '--n-vehicles', type=int, default=50,
                           help='Number of vehicles (default: 50)')
    gen_parser.add_argument('-d', '--n-days', type=int, default=365,
                           help='Number of days (default: 365)')
    gen_parser.add_argument('-s', '--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    gen_parser.add_argument('-o', '--output', type=str, default=None,
                           help='Output file path')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--data', type=str, default=None,
                             help='Path to input data CSV')
    train_parser.add_argument('--model-dir', type=str, default='models',
                             help='Directory to save models')
    train_parser.add_argument('--no-smote', action='store_true',
                             help='Disable SMOTE for class balancing')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('--data', type=str, default=None,
                            help='Path to input data CSV')
    pred_parser.add_argument('--model-dir', type=str, default='models',
                            help='Directory with saved models')
    pred_parser.add_argument('-v', '--vehicle-id', type=str, default=None,
                            help='Specific vehicle ID to predict')

    # Pipeline command
    pipe_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipe_parser.add_argument('-n', '--n-vehicles', type=int, default=50,
                            help='Number of vehicles (default: 50)')
    pipe_parser.add_argument('-d', '--n-days', type=int, default=365,
                            help='Number of days (default: 365)')
    pipe_parser.add_argument('-s', '--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    pipe_parser.add_argument('--model-dir', type=str, default='models',
                            help='Directory to save models')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'generate':
        generate_data(args)
    elif args.command == 'train':
        train_models(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'pipeline':
        run_pipeline(args)


if __name__ == "__main__":
    main()
