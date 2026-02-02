# Fleet Predictive Maintenance

A machine learning system that predicts vehicle component failures using telematics data, enabling proactive maintenance and reducing unplanned breakdowns.

## The Problem

Unplanned vehicle breakdowns cost carriers **$500-800 per incident** in labor, towing, and missed deliveries. Traditional preventive maintenance wastes money by replacing parts too early based on fixed schedules rather than actual condition.

## The Solution

This project uses classification models trained on telematics data to predict component failures **30 days in advance**, allowing fleet managers to:
- Schedule maintenance during planned downtime
- Reduce emergency repairs by catching failures early
- Optimize parts replacement timing
- Estimate cost savings from predictive vs. reactive maintenance

## Components Predicted

| Component | Key Indicators | Business Impact |
|-----------|---------------|-----------------|
| **Engine** | Temperature, oil pressure, engine hours | Highest cost failures |
| **Brakes** | Pad thickness, hard brake events, load weight | Safety critical |
| **Battery** | Voltage, age, ambient temperature | Common roadside failures |

## Project Structure

```
fleet-predictive-maintenance/
├── data/
│   ├── raw/                    # Generated telematics data
│   └── processed/              # Feature-engineered data
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── src/
│   ├── data_generator.py      # Synthetic data generation
│   ├── feature_engineering.py # Feature creation
│   ├── preprocessing.py       # Data preparation & SMOTE
│   ├── models.py              # Model training & evaluation
│   └── predict.py             # Inference pipeline
├── app/
│   └── api.py                 # REST API endpoint
├── tests/
│   └── test_pipeline.py       # Unit tests
├── models/                     # Saved model artifacts
├── main.py                     # CLI entry point
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
cd fleet-predictive-maintenance
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main.py pipeline
```

This will:
1. Generate synthetic telematics data for 50 vehicles over 1 year
2. Engineer features and train multiple models
3. Evaluate on test set and calculate business impact
4. Output fleet risk predictions

### 3. Individual Commands

```bash
# Generate data only
python main.py generate -n 100 -d 365

# Train models
python main.py train

# Make predictions
python main.py predict
python main.py predict -v V001  # Specific vehicle
```

### 4. Run API Server

```bash
python app/api.py
```

Then send predictions:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mileage": 85000,
    "engine_hours": 4200,
    "engine_temp": 235,
    "oil_pressure": 28,
    "battery_voltage": 12.1,
    "brake_pad_thickness": 3.5,
    "error_code_count": 2,
    "ambient_temp": 85,
    "load_weight": 35000,
    "idle_time_pct": 22,
    "hard_brake_events": 4,
    "days_since_maintenance": 75
  }'
```

## Models Compared

| Model | Val Accuracy | Val F1 (Macro) | Notes |
|-------|-------------|----------------|-------|
| Logistic Regression | ~75% | ~0.65 | Baseline, interpretable |
| Random Forest | ~82% | ~0.72 | Good feature importance |
| Gradient Boosting | ~84% | ~0.75 | Best overall performance |
| SVM | ~78% | ~0.68 | Slower training |

*Results vary based on random seed and data generation*

## Feature Engineering

The system creates 100+ features from raw telematics:

**Rolling Statistics (7, 14, 30 day windows)**
- Mean, std, min, max for sensor readings

**Rate of Change**
- Daily change and 7-day trends
- Acceleration (change in rate of change)

**Threshold Flags**
- Warning levels (e.g., engine_temp > 230°F)
- Critical levels (e.g., brake_pad < 2.5mm)

**Interaction Features**
- Load × mileage interaction
- Engine stress index
- Battery health composite

**Cumulative Features**
- Total error codes
- Warning day counts

## Business Impact

The model calculates estimated cost savings:

```
Cost Parameters:
  Unplanned breakdown: $650 (avg)
  Preventive maintenance: $150
  False alarm inspection: $50

Example Output:
  Correctly predicted failures: 45
  Missed failures: 8
  False alarms: 12

  Cost without model: $34,450
  Cost with model: $12,350
  TOTAL SAVINGS: $22,100 (64%)
```

## Evaluation Metrics

- **Precision/Recall per class** - Critical for imbalanced failure data
- **F1-Score (Macro)** - Balances performance across all failure types
- **ROC-AUC** - Overall discrimination ability
- **Confusion Matrix** - Detailed error analysis
- **Cost Savings** - Business-relevant impact metric

## Key Insights

1. **Class Imbalance**: Failures are rare (~10-15% of readings). SMOTE is used to balance training data.

2. **Feature Importance**: Rolling averages and threshold flags are typically most predictive.

3. **Seasonal Effects**: Battery failures increase in extreme temperature months.

4. **Lead Time**: 30-day prediction window provides actionable maintenance scheduling.

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

## Exploratory Data Analysis

Open the Jupyter notebook for visualizations:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Includes:
- Failure type distributions
- Sensor reading distributions by failure type
- Correlation heatmaps
- Time-series trends
- Feature importance plots

## Future Improvements

- [ ] Add SHAP values for model explainability
- [ ] Implement time-series models (LSTM)
- [ ] Add real-time streaming predictions
- [ ] Create dashboard with Streamlit/Dash
- [ ] Add hyperparameter tuning
- [ ] Support for additional component types

## Tech Stack

- **Python 3.9+**
- **scikit-learn** - ML models
- **imbalanced-learn** - SMOTE for class balancing
- **pandas/numpy** - Data processing
- **matplotlib/seaborn** - Visualization
- **Flask** - REST API
- **pytest** - Testing

## License

MIT License - Feel free to use for learning and portfolio purposes.
