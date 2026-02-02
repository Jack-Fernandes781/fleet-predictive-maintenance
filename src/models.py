"""
Model Training and Evaluation Module

Trains multiple classification models and evaluates their performance
for predicting vehicle component failures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score


# Business cost parameters
COST_PARAMS = {
    'cost_per_unplanned_breakdown': 650,  # Average of $500-800
    'cost_per_preventive_maintenance': 150,
    'cost_per_false_alarm': 50,  # Inspection cost for false positive
}


class ModelTrainer:
    """Trains and evaluates multiple classification models."""

    def __init__(self, class_weights: Optional[Dict[int, float]] = None, random_state: int = 42):
        """
        Initialize the model trainer.

        Args:
            class_weights: Dictionary of class weights for imbalanced data
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.class_weights = class_weights
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None

    def get_models(self) -> Dict[str, Any]:
        """Define the models to train."""
        return {
            'Logistic Regression': LogisticRegression(
                class_weight=self.class_weights,
                random_state=self.random_state,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight=self.class_weights,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5
            ),
            'SVM': SVC(
                class_weight=self.class_weights,
                random_state=self.random_state,
                probability=True,
                kernel='rbf'
            )
        }

    def train_model(self,
                    model: Any,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train a single model and evaluate on validation set.

        Args:
            model: sklearn model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary with training results
        """
        # Train the model
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Probabilities (for ROC-AUC)
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val)
        else:
            y_val_proba = None

        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision_weighted': precision_score(y_val, y_val_pred, average='weighted'),
            'val_recall_weighted': recall_score(y_val, y_val_pred, average='weighted'),
            'val_f1_weighted': f1_score(y_val, y_val_pred, average='weighted'),
            'val_precision_macro': precision_score(y_val, y_val_pred, average='macro'),
            'val_recall_macro': recall_score(y_val, y_val_pred, average='macro'),
            'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred),
        }

        # ROC-AUC for multi-class
        if y_val_proba is not None:
            try:
                results['val_roc_auc'] = roc_auc_score(
                    y_val, y_val_proba, multi_class='ovr', average='weighted'
                )
            except ValueError:
                results['val_roc_auc'] = None

        return results

    def train_all_models(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray) -> Dict[str, Dict]:
        """
        Train all models and collect results.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary with all model results
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)

        models = self.get_models()

        for name, model in models.items():
            print(f"\nTraining {name}...")

            results = self.train_model(model, X_train, y_train, X_val, y_val)

            self.models[name] = model
            self.results[name] = results

            print(f"  Train Accuracy: {results['train_accuracy']:.3f}")
            print(f"  Val Accuracy:   {results['val_accuracy']:.3f}")
            print(f"  Val F1 (macro): {results['val_f1_macro']:.3f}")
            if results.get('val_roc_auc'):
                print(f"  Val ROC-AUC:    {results['val_roc_auc']:.3f}")

        # Determine best model (by macro F1 - balances all classes)
        self.best_model_name = max(
            self.results.keys(),
            key=lambda k: self.results[k]['val_f1_macro']
        )
        self.best_model = self.models[self.best_model_name]

        print(f"\nBest model: {self.best_model_name}")

        return self.results

    def evaluate_on_test(self,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        class_names: List[str]) -> Dict[str, Any]:
        """
        Evaluate the best model on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            class_names: List of class names

        Returns:
            Dictionary with test evaluation results
        """
        print("\n" + "="*50)
        print(f"TEST SET EVALUATION - {self.best_model_name}")
        print("="*50)

        y_pred = self.best_model.predict(X_test)

        if hasattr(self.best_model, 'predict_proba'):
            y_proba = self.best_model.predict_proba(X_test)
        else:
            y_proba = None

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=class_names, columns=class_names))

        # Per-class metrics
        test_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_per_class': precision_score(y_test, y_pred, average=None),
            'recall_per_class': recall_score(y_test, y_pred, average=None),
            'f1_per_class': f1_score(y_test, y_pred, average=None),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        if y_proba is not None:
            try:
                test_results['roc_auc'] = roc_auc_score(
                    y_test, y_proba, multi_class='ovr', average='weighted'
                )
                print(f"\nROC-AUC (weighted): {test_results['roc_auc']:.3f}")
            except ValueError:
                pass

        return test_results

    def calculate_business_impact(self,
                                 y_test: np.ndarray,
                                 y_pred: np.ndarray,
                                 class_names: List[str]) -> Dict[str, float]:
        """
        Calculate business impact metrics (cost savings).

        Args:
            y_test: True labels
            y_pred: Predicted labels
            class_names: List of class names

        Returns:
            Dictionary with business impact metrics
        """
        print("\n" + "="*50)
        print("BUSINESS IMPACT ANALYSIS")
        print("="*50)

        # Find index for 'None' class (no failure)
        none_idx = class_names.index('None') if 'None' in class_names else 0
        failure_indices = [i for i, name in enumerate(class_names) if name != 'None']

        # True positives: Correctly predicted failures
        true_positives = sum(
            (y_test == i) & (y_pred == i)
            for i in failure_indices
        ).sum()

        # False negatives: Missed failures (predicted None, was failure)
        false_negatives = sum(
            (y_test == i) & (y_pred == none_idx)
            for i in failure_indices
        ).sum()

        # False positives: False alarms (predicted failure, was None)
        false_positives = sum(
            (y_test == none_idx) & (y_pred == i)
            for i in failure_indices
        ).sum()

        # Cost calculations
        cost_without_model = (true_positives + false_negatives) * COST_PARAMS['cost_per_unplanned_breakdown']
        cost_missed_failures = false_negatives * COST_PARAMS['cost_per_unplanned_breakdown']
        cost_preventive = true_positives * COST_PARAMS['cost_per_preventive_maintenance']
        cost_false_alarms = false_positives * COST_PARAMS['cost_per_false_alarm']
        cost_with_model = cost_missed_failures + cost_preventive + cost_false_alarms

        savings = cost_without_model - cost_with_model
        savings_pct = (savings / cost_without_model * 100) if cost_without_model > 0 else 0

        # Print results
        print(f"\nScenario Analysis (Test Set):")
        print(f"  Total actual failures:    {true_positives + false_negatives}")
        print(f"  Correctly predicted:      {true_positives}")
        print(f"  Missed failures:          {false_negatives}")
        print(f"  False alarms:             {false_positives}")

        print(f"\nCost Analysis:")
        print(f"  Cost without model:       ${cost_without_model:,.0f}")
        print(f"  Cost with model:          ${cost_with_model:,.0f}")
        print(f"    - Missed failures:      ${cost_missed_failures:,.0f}")
        print(f"    - Preventive maint:     ${cost_preventive:,.0f}")
        print(f"    - False alarm inspections: ${cost_false_alarms:,.0f}")
        print(f"\n  TOTAL SAVINGS:            ${savings:,.0f} ({savings_pct:.1f}%)")

        return {
            'true_positives': int(true_positives),
            'false_negatives': int(false_negatives),
            'false_positives': int(false_positives),
            'cost_without_model': cost_without_model,
            'cost_with_model': cost_with_model,
            'total_savings': savings,
            'savings_percentage': savings_pct
        }

    def get_feature_importance(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the best model.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importances (if available)
        """
        if not hasattr(self.best_model, 'feature_importances_'):
            print(f"{self.best_model_name} doesn't support feature importance.")
            return None

        importance = self.best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("\n" + "="*50)
        print("TOP 15 MOST IMPORTANT FEATURES")
        print("="*50)
        print(importance_df.head(15).to_string(index=False))

        return importance_df

    def save_model(self, output_dir: str) -> str:
        """Save the best model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = output_path / f"{self.best_model_name.replace(' ', '_').lower()}_model.joblib"
        joblib.dump(self.best_model, model_path)

        # Also save all models
        all_models_path = output_path / "all_models.joblib"
        joblib.dump(self.models, all_models_path)

        print(f"\nModel saved to: {model_path}")
        return str(model_path)

    def get_comparison_summary(self) -> pd.DataFrame:
        """Get a summary comparison of all models."""
        summary = []
        for name, results in self.results.items():
            summary.append({
                'Model': name,
                'Train Acc': f"{results['train_accuracy']:.3f}",
                'Val Acc': f"{results['val_accuracy']:.3f}",
                'Val F1 (weighted)': f"{results['val_f1_weighted']:.3f}",
                'Val F1 (macro)': f"{results['val_f1_macro']:.3f}",
                'Val Recall (macro)': f"{results['val_recall_macro']:.3f}",
                'Val ROC-AUC': f"{results.get('val_roc_auc', 'N/A'):.3f}" if results.get('val_roc_auc') else 'N/A'
            })

        df = pd.DataFrame(summary)

        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        print(df.to_string(index=False))

        return df


def train_and_evaluate(preprocessed_data: Dict[str, Any],
                       output_dir: str = "models") -> Dict[str, Any]:
    """
    Convenience function to train and evaluate all models.

    Args:
        preprocessed_data: Dictionary from preprocessing pipeline
        output_dir: Directory to save models

    Returns:
        Dictionary with all results
    """
    splits = preprocessed_data['splits']
    class_weights = preprocessed_data['class_weights']
    class_names = preprocessed_data['classes']
    feature_names = preprocessed_data['feature_names']

    # Initialize trainer
    trainer = ModelTrainer(class_weights=class_weights)

    # Train all models
    training_results = trainer.train_all_models(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val']
    )

    # Compare models
    comparison_df = trainer.get_comparison_summary()

    # Evaluate best model on test set
    test_results = trainer.evaluate_on_test(
        splits['X_test'], splits['y_test'],
        class_names
    )

    # Calculate business impact
    business_impact = trainer.calculate_business_impact(
        splits['y_test'], test_results['y_pred'],
        class_names
    )

    # Get feature importance
    feature_importance = trainer.get_feature_importance(feature_names)

    # Save model
    model_path = trainer.save_model(output_dir)

    return {
        'trainer': trainer,
        'training_results': training_results,
        'test_results': test_results,
        'business_impact': business_impact,
        'feature_importance': feature_importance,
        'comparison_df': comparison_df,
        'model_path': model_path,
        'best_model_name': trainer.best_model_name
    }


if __name__ == "__main__":
    from data_generator import generate_fleet_data
    from feature_engineering import engineer_features
    from preprocessing import preprocess_data

    # Generate data
    df = generate_fleet_data(n_vehicles=20, n_days=120, random_seed=42)

    # Engineer features
    df = engineer_features(df)

    # Preprocess
    preprocessed = preprocess_data(df)

    # Train and evaluate
    results = train_and_evaluate(preprocessed)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
