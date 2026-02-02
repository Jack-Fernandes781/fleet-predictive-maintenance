"""
Flask REST API for Fleet Predictive Maintenance

Endpoints:
    POST /predict - Predict failure for vehicle telemetry
    GET /health - Health check
    GET /fleet-summary - Get predictions for entire fleet
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import load_predictor
from feature_engineering import engineer_features
import pandas as pd
import numpy as np

app = Flask(__name__)

# Global predictor (loaded on first request)
predictor = None


def get_predictor():
    """Lazy load the predictor."""
    global predictor
    if predictor is None:
        model_dir = Path(__file__).parent.parent / "models"
        predictor = load_predictor(str(model_dir))
    return predictor


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Fleet Predictive Maintenance API',
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict failure for vehicle telemetry data.

    Request body should contain vehicle sensor readings:
    {
        "mileage": 75000,
        "engine_hours": 3500,
        "engine_temp": 215,
        "oil_pressure": 35,
        "battery_voltage": 12.8,
        "brake_pad_thickness": 5.5,
        "error_code_count": 0,
        "ambient_temp": 72,
        "load_weight": 25000,
        "idle_time_pct": 15,
        "hard_brake_events": 2,
        "days_since_maintenance": 45
    }
    """
    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Required fields
        required_fields = [
            'mileage', 'engine_hours', 'engine_temp', 'oil_pressure',
            'battery_voltage', 'brake_pad_thickness', 'error_code_count',
            'ambient_temp', 'load_weight', 'idle_time_pct',
            'hard_brake_events', 'days_since_maintenance'
        ]

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400

        # Create DataFrame for feature engineering
        df = pd.DataFrame([data])
        df['vehicle_id'] = data.get('vehicle_id', 'API_REQUEST')
        df['timestamp'] = pd.Timestamp.now()
        df['failure_within_30_days'] = 'None'  # Placeholder

        # Engineer features
        df_features = engineer_features(df)

        # Get predictor
        pred = get_predictor()

        # Make prediction
        result = pred.predict_from_dict(df_features.iloc[0].to_dict())

        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 3),
            'risk_level': result['risk_level'],
            'probabilities': {k: round(v, 3) for k, v in result['probabilities'].items()},
            'recommendation': get_recommendation(result)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict failures for multiple vehicles.

    Request body should be a list of vehicle telemetry readings.
    """
    try:
        data = request.get_json()

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected a list of vehicle readings'}), 400

        results = []
        for i, vehicle_data in enumerate(data):
            # Create DataFrame
            df = pd.DataFrame([vehicle_data])
            df['vehicle_id'] = vehicle_data.get('vehicle_id', f'VEHICLE_{i}')
            df['timestamp'] = pd.Timestamp.now()
            df['failure_within_30_days'] = 'None'

            # Engineer features
            df_features = engineer_features(df)

            # Get prediction
            pred = get_predictor()
            result = pred.predict_from_dict(df_features.iloc[0].to_dict())

            results.append({
                'vehicle_id': df['vehicle_id'].iloc[0],
                'prediction': result['prediction'],
                'confidence': round(result['confidence'], 3),
                'risk_level': result['risk_level']
            })

        return jsonify({
            'success': True,
            'predictions': results,
            'total_vehicles': len(results),
            'high_risk_count': sum(1 for r in results if r['risk_level'] in ['High', 'Critical'])
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_recommendation(result):
    """Generate maintenance recommendation based on prediction."""
    prediction = result['prediction']
    risk_level = result['risk_level']

    if prediction == 'None' and risk_level == 'Low':
        return "No immediate action required. Continue regular monitoring."

    recommendations = {
        'Engine': "Schedule engine inspection. Check oil levels, coolant, and temperature sensors.",
        'Brakes': "Inspect brake system immediately. Check pad thickness and brake fluid.",
        'Battery': "Test battery health. Consider replacement if voltage remains low."
    }

    urgency = {
        'Low': "Routine check recommended within 30 days.",
        'Medium': "Schedule inspection within 2 weeks.",
        'High': "Urgent: Schedule service within 1 week.",
        'Critical': "CRITICAL: Immediate service required. Do not operate until inspected."
    }

    rec = recommendations.get(prediction, "General inspection recommended.")
    urg = urgency.get(risk_level, "")

    return f"{rec} {urg}"


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("FLEET PREDICTIVE MAINTENANCE API")
    print("="*50)
    print("\nEndpoints:")
    print("  GET  /health         - Health check")
    print("  POST /predict        - Predict single vehicle")
    print("  POST /predict-batch  - Predict multiple vehicles")
    print("\nStarting server on http://localhost:5000")
    print("="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
