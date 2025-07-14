#!/usr/bin/env python3
"""
Titanic Survival Prediction - Training Pipeline

This script implements a complete ML pipeline for training a binary classifier
to predict Titanic passenger survival. It includes data preprocessing, model
training, evaluation, and profiling.
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import psutil
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import structlog
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class TitanicPipeline:
    """Complete ML pipeline for Titanic survival prediction."""

    def __init__(self, data_path: str = "data/train.csv", model_path: str = "model"):
        self.data_path = data_path
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)

        # Initialize components
        self.scaler = StandardScaler()

        # Create individual models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        self.lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )

        # Create VotingClassifier ensemble
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('xgb', self.xgb_model),
                ('lr', self.lr_model)
            ],
            voting='soft'  # Use probabilities for voting
        )

        # Performance tracking
        self.start_time = None
        self.memory_usage = []

    def log_system_info(self) -> None:
        """Log system information for profiling."""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()

        logger.info("System Information",
                    cpu_count=cpu_count,
                    memory_total_gb=memory.total / (1024**3),
                    memory_available_gb=memory.available / (1024**3))

    def start_profiling(self) -> None:
        """Start performance profiling."""
        self.start_time = time.time()
        self.memory_usage = []
        logger.info("Started pipeline profiling")

    def log_memory_usage(self, stage: str) -> None:
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        self.memory_usage.append({
            'stage': stage,
            'memory_mb': memory_mb,
            'timestamp': time.time()
        })

        logger.info(f"Memory usage at {stage}",
                    stage=stage,
                    memory_mb=memory_mb)

    def load_data(self) -> pd.DataFrame:
        """Load and validate training data."""
        logger.info("Loading training data", data_path=self.data_path)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Training data not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        logger.info("Data loaded successfully",
                    shape=df.shape,
                    columns=list(df.columns))

        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data for training with advanced feature engineering."""
        logger.info("Starting advanced data preprocessing")

        # Create a copy to avoid modifying original data
        data = df.copy()

        # ===== FEATURE ENGINEERING =====
        # 1. Extract title from Name
        data['Title'] = data['Name'].str.extract(
            r' ([A-Za-z]+)\.', expand=False)
        # Agrupar títulos raros
        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
        }
        data['Title'] = data['Title'].map(title_mapping).fillna('Rare')

        # 2. Create age groups (bins)
        data['AgeBin'] = pd.qcut(
            data['Age'], 4, labels=False, duplicates='drop')
        # 3. Create fare bins
        data['FareBin'] = pd.qcut(
            data['Fare'], 4, labels=False, duplicates='drop')
        # 4. Age missing indicator
        data['AgeMissing'] = data['Age'].isnull().astype(int)
        # 5. HasCabin indicator
        data['HasCabin'] = data['Cabin'].notnull().astype(int)
        # 6. Ticket prefix
        data['TicketPrefix'] = data['Ticket'].str.extract(
            r'([A-Za-z]+)', expand=False).fillna('None')
        # 7. Extract cabin letter (if Cabin is not null)
        data['CabinLetter'] = data['Cabin'].str[0].fillna('Unknown')
        # 8. Create IsAlone feature
        data['IsAlone'] = ((data['SibSp'] + data['Parch']) == 0).astype(int)
        # 9. Create family size feature
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        # 10. Create fare per person
        data['FarePerPerson'] = data['Fare'] / data['FamilySize']

        # ===== IMPROVED MISSING VALUE HANDLING =====
        # Age: fill with median by title
        title_age_median = data.groupby('Title')['Age'].median()
        data['Age'] = data['Age'].fillna(data['Title'].map(title_age_median))
        data['Age'] = data['Age'].fillna(data['Age'].median())
        # Embarked: fill with mode
        data['Embarked'] = data['Embarked'].fillna('S')
        # Fare: fill with median by class
        class_fare_median = data.groupby('Pclass')['Fare'].median()
        data['Fare'] = data['Fare'].fillna(
            data['Pclass'].map(class_fare_median))
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        # CabinLetter: fill unknown with 'Unknown'
        data['CabinLetter'] = data['CabinLetter'].fillna('Unknown')

        # ===== FEATURE SELECTION =====
        columns_to_drop = ['Name', 'Ticket', 'Cabin']
        data = data.drop(columns_to_drop, axis=1)

        # ===== ENCODING =====
        categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title',
                                'AgeGroup', 'CabinLetter', 'TicketPrefix', 'AgeBin', 'FareBin']
        dummy_dfs = []
        for feature in categorical_features:
            if feature in data.columns:
                dummies = pd.get_dummies(data[feature], prefix=feature)
                dummy_dfs.append(dummies)
                data = data.drop(feature, axis=1)
        if dummy_dfs:
            data = data.join(dummy_dfs)

        # ===== FEATURE SCALING PREPARATION =====
        X = data.drop('Survived', axis=1)
        y = data['Survived']
        X.set_index('PassengerId', inplace=True)
        logger.info("Advanced preprocessing completed",
                    features_shape=X.shape,
                    target_shape=y.shape,
                    feature_names=list(X.columns),
                    new_features_added=[
                        'Title', 'AgeBin', 'FareBin', 'AgeMissing', 'HasCabin', 'TicketPrefix',
                        'CabinLetter', 'IsAlone', 'FamilySize', 'FarePerPerson'
                    ])
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model and return performance metrics."""
        logger.info("Starting model training")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train individual models first (for feature importance)
        self.rf_model.fit(X_train_scaled, y_train)
        self.xgb_model.fit(X_train_scaled, y_train)
        self.lr_model.fit(X_train_scaled, y_train)

        # Train ensemble model
        self.model.fit(X_train_scaled, y_train)

        # Store column names for later use
        self.X_columns = X.columns.tolist()

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()

        # Feature importance (average from RandomForest and XGBoost)
        rf_importance = dict(
            zip(X.columns, [float(imp) for imp in self.rf_model.feature_importances_]))
        xgb_importance = dict(
            zip(X.columns, [float(imp) for imp in self.xgb_model.feature_importances_]))

        # Average the importances
        feature_importance = {}
        for feature in X.columns:
            feature_importance[feature] = (
                rf_importance[feature] + xgb_importance[feature]) / 2

        logger.info("Model training completed",
                    accuracy=metrics['accuracy'],
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    f1_score=metrics['f1_score'],
                    roc_auc=metrics['roc_auc'])

        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def save_model(self) -> None:
        """Save the trained model and preprocessing components."""
        logger.info("Saving model and components")

        # Save ensemble model
        model_file = self.model_path / "model.pkl"
        joblib.dump(self.model, model_file)

        # Save individual models for potential use
        rf_file = self.model_path / "random_forest.pkl"
        xgb_file = self.model_path / "xgboost.pkl"
        lr_file = self.model_path / "logistic_regression.pkl"

        joblib.dump(self.rf_model, rf_file)
        joblib.dump(self.xgb_model, xgb_file)
        joblib.dump(self.lr_model, lr_file)

        # Save scaler
        scaler_file = self.model_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)

        # Save feature names
        feature_names_file = self.model_path / "feature_names.json"
        # Get feature names from the model if available, otherwise use stored column names
        try:
            feature_names = list(self.model.feature_names_in_)
        except AttributeError:
            # Fallback for older scikit-learn versions or when feature_names_in_ is not available
            feature_names = self.X_columns if hasattr(
                self, 'X_columns') else []
        with open(feature_names_file, 'w') as f:
            json.dump(feature_names, f)

        logger.info("Model saved successfully",
                    model_file=str(model_file),
                    scaler_file=str(scaler_file))

    def save_metrics(self, results: Dict[str, Any]) -> None:
        """Save training metrics and results."""
        metrics_file = self.model_path / "metrics.json"

        # Prepare metrics for saving
        # Convert float32 to float for JSON serialization
        metrics = results['metrics'].copy()
        feature_importance = {
            k: float(v) for k, v in results['feature_importance'].items()}

        save_data = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_info': {
                'timestamp': time.time(),
                'memory_usage': self.memory_usage,
                'total_time': time.time() - self.start_time if self.start_time else None
            }
        }

        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info("Metrics saved", metrics_file=str(metrics_file))

    def save_confusion_matrix(self, y_true, y_pred):
        """Save confusion matrix as PNG image."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(self.model_path / "confusion_matrix.png")
        plt.close()
        logger.info("Confusion matrix saved", file=str(
            self.model_path / "confusion_matrix.png"))

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the training results."""
        metrics = results['metrics']

        print("\n" + "="*50)
        print("TITANIC SURVIVAL PREDICTION - TRAINING SUMMARY")
        print("="*50)

        print(f"\nModel Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(
            f"  CV Score:   {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

        print(f"\nTop 5 Most Important Features:")
        feature_importance = results['feature_importance']
        sorted_features = sorted(feature_importance.items(),
                                 key=lambda x: x[1], reverse=True)[:5]

        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")

        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\nTraining completed in {total_time:.2f} seconds")

        print("="*50)

    def run(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        try:
            self.log_system_info()
            self.start_profiling()

            # Load data
            self.log_memory_usage("data_loading")
            df = self.load_data()

            # Preprocess data
            self.log_memory_usage("preprocessing")
            X, y = self.preprocess_data(df)

            # Train model
            self.log_memory_usage("training")
            results = self.train_model(X, y)

            # Save model and metrics
            self.log_memory_usage("saving")
            self.save_model()
            self.save_metrics(results)
            # Save confusion matrix
            self.save_confusion_matrix(results['y_test'], results['y_pred'])

            # Print summary
            self.print_summary(results)

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error("Pipeline failed", error=str(e), exc_info=True)
            raise


def main():
    """Main entry point for the training pipeline."""
    try:
        # Create pipeline
        pipeline = TitanicPipeline()

        # Run pipeline
        results = pipeline.run()

        # Exit with success
        sys.exit(0)

    except Exception as e:
        logger.error("Training pipeline failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
