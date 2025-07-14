"""
Utility functions for the Titanic Survival Prediction API.

This module provides helper functions for model loading, data preprocessing,
security, and monitoring.
"""

import os
import json
import time
import hashlib
import hmac
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger()


class ModelManager:
    """Manages model loading, caching, and prediction."""

    def __init__(self, model_path: str = "model"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metrics = None
        self.load_time = None
        self.model_version = "1.0.0"

    def load_model(self) -> bool:
        """Load the trained model and components."""
        try:
            logger.info("Loading model components",
                        model_path=str(self.model_path))

            # Load model
            model_file = self.model_path / "model.pkl"
            if not model_file.exists():
                logger.error("Model file not found",
                             model_file=str(model_file))
                return False

            self.model = joblib.load(model_file)

            # Load scaler
            scaler_file = self.model_path / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)

            # Load feature names
            feature_names_file = self.model_path / "feature_names.json"
            if feature_names_file.exists():
                with open(feature_names_file, 'r') as f:
                    self.feature_names = json.load(f)

            # Load metrics
            metrics_file = self.model_path / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.model_metrics = json.load(f)

            self.load_time = time.time()
            logger.info("Model loaded successfully",
                        model_version=self.model_version,
                        load_time=self.load_time)

            return True

        except Exception as e:
            logger.error("Failed to load model", error=str(e), exc_info=True)
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def preprocess_passenger(self, passenger_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single passenger for prediction with advanced feature engineering."""
        # Create DataFrame with expected structure
        df = pd.DataFrame([passenger_data])

        # ===== FEATURE ENGINEERING =====

        # 1. Extract title from Name
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Group rare titles
        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Mlle': 'Miss',
            'Countess': 'Rare',
            'Ms': 'Miss',
            'Lady': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Mme': 'Mrs',
            'Capt': 'Rare',
            'Sir': 'Rare'
        }
        df['Title'] = df['Title'].map(title_mapping)

        # 2. Create age groups
        df['AgeGroup'] = pd.cut(df['Age'],
                                bins=[0, 12, 18, 65, 100],
                                labels=['Child', 'Teen', 'Adult', 'Elderly'],
                                include_lowest=True)

        # 3. Extract cabin letter (if Cabin is not null)
        df['CabinLetter'] = df['Cabin'].str[0] if df['Cabin'].notna().any() else 'Unknown'

        # 4. Create IsAlone feature
        df['IsAlone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)

        # 5. Create family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # 6. Create fare per person
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']

        # ===== IMPROVED MISSING VALUE HANDLING =====

        # Age: fill with median by title (fallback to overall median)
        title_age_median = df.groupby('Title')['Age'].median()
        df['Age'] = df['Age'].fillna(df['Title'].map(title_age_median))
        df['Age'] = df['Age'].fillna(df['Age'].median())  # Fallback

        # Embarked: fill with mode
        df['Embarked'] = df['Embarked'].fillna('S')

        # Fare: fill with median by class (fallback to overall median)
        class_fare_median = df.groupby('Pclass')['Fare'].median()
        df['Fare'] = df['Fare'].fillna(df['Pclass'].map(class_fare_median))
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Fallback

        # CabinLetter: fill unknown with 'Unknown'
        df['CabinLetter'] = df['CabinLetter'].fillna('Unknown')

        # ===== FEATURE SELECTION =====

        # Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'Cabin']
        df = df.drop(
            [col for col in columns_to_drop if col in df.columns], axis=1)

        # ===== ENCODING =====

        # Create dummy variables for categorical features
        categorical_features = ['Embarked', 'Sex',
                                'Pclass', 'Title', 'AgeGroup', 'CabinLetter']

        # Initialize list to store dummy dataframes
        dummy_dfs = []

        for feature in categorical_features:
            if feature in df.columns:
                dummies = pd.get_dummies(df[feature], prefix=feature)
                dummy_dfs.append(dummies)
                df = df.drop(feature, axis=1)

        # Join all dummy variables
        if dummy_dfs:
            df = df.join(dummy_dfs)

        # Ensure all expected columns are present
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0

        # Reorder columns to match training data
        if self.feature_names:
            df = df[self.feature_names]

        # Scale features if scaler is available
        if self.scaler:
            return self.scaler.transform(df.values)
        else:
            return df.values

    def predict(self, passenger_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Make prediction for a single passenger."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        try:
            # Preprocess passenger data
            features = self.preprocess_passenger(passenger_data)

            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0][1]

            return bool(prediction), float(probability)
        except Exception as e:
            logger.error("Prediction failed in ModelManager",
                         error=str(e), exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, passengers_data: List[Dict[str, Any]]) -> List[Tuple[bool, float]]:
        """Make predictions for multiple passengers."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        results = []
        for passenger_data in passengers_data:
            result = self.predict(passenger_data)
            results.append(result)

        return results

    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance from the model."""
        if not self.is_loaded() or not hasattr(self.model, 'feature_importances_'):
            return []

        importance = self.model.feature_importances_
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(importance))]

        feature_importance = []
        for name, imp in zip(feature_names, importance):
            feature_importance.append({
                "feature": name,
                "importance": float(imp)
            })

        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        return feature_importance


class SecurityManager:
    """Manages API security and authentication."""

    def __init__(self, api_keys: Optional[List[str]] = None):
        self.api_keys = api_keys or []
        self.valid_api_keys = set(api_keys) if api_keys else set()

    def validate_api_key(self, api_key: Optional[str]) -> bool:
        """Validate API key."""
        if not self.valid_api_keys:
            return True  # No API keys configured, allow all requests

        if not api_key:
            return False

        return api_key in self.valid_api_keys

    def generate_api_key(self, user_id: str) -> str:
        """Generate a new API key for a user."""
        timestamp = str(int(time.time()))
        message = f"{user_id}:{timestamp}"

        # In production, use a proper secret key
        secret = os.getenv("API_SECRET_KEY", "default-secret-key")
        signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{user_id}.{timestamp}.{signature}"


class MetricsCollector:
    """Collects and tracks API metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.prediction_times = []

    def record_request(self, prediction_time: float, success: bool = True):
        """Record a request."""
        self.request_count += 1
        if not success:
            self.error_count += 1

        self.prediction_times.append(prediction_time)

        # Keep only last 1000 prediction times
        if len(self.prediction_times) > 1000:
            self.prediction_times = self.prediction_times[-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self.start_time

        avg_prediction_time = (
            np.mean(self.prediction_times) if self.prediction_times else 0
        )

        error_rate = (
            self.error_count / self.request_count if self.request_count > 0 else 0
        )

        return {
            "uptime": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_prediction_time": avg_prediction_time,
            "min_prediction_time": min(self.prediction_times) if self.prediction_times else 0,
            "max_prediction_time": max(self.prediction_times) if self.prediction_times else 0
        }


def validate_passenger_data(data: Dict[str, Any]) -> List[str]:
    """Validate passenger data and return list of errors."""
    errors = []

    required_fields = ['passenger_id', 'pclass', 'sex',
                       'age', 'sibsp', 'parch', 'fare', 'embarked']

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if 'age' in data and (data['age'] < 0 or data['age'] > 120):
        errors.append("Age must be between 0 and 120")

    if 'fare' in data and data['fare'] < 0:
        errors.append("Fare must be non-negative")

    if 'pclass' in data and data['pclass'] not in [1, 2, 3]:
        errors.append("Pclass must be 1, 2, or 3")

    if 'sex' in data and data['sex'] not in ['male', 'female']:
        errors.append("Sex must be 'male' or 'female'")

    if 'embarked' in data and data['embarked'] not in ['S', 'C', 'Q']:
        errors.append("Embarked must be 'S', 'C', or 'Q'")

    return errors


def format_error_response(error: str, detail: Optional[str] = None) -> Dict[str, Any]:
    """Format error response."""
    return {
        "error": error,
        "detail": detail,
        "timestamp": datetime.utcnow().isoformat()
    }


# Global instances
model_manager = ModelManager()
security_manager = SecurityManager()
metrics_collector = MetricsCollector()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return model_manager


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    return security_manager


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector
