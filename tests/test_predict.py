"""
Unit tests for the Titanic Survival Prediction API.

This module contains comprehensive tests for the API endpoints,
model functionality, and data validation.
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from api.app import app
from api.schemas import Passenger, Sex, Embarked
from api.utils import ModelManager, SecurityManager, MetricsCollector


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_passenger():
    """Create a sample passenger for testing."""
    return {
        "passenger_id": 1,
        "pclass": 1,
        "sex": "female",
        "age": 29.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 211.3375,
        "embarked": "S"
    }


@pytest.fixture
def sample_passenger_schema():
    """Create a sample passenger schema."""
    return Passenger(
        passenger_id=1,
        pclass=1,
        sex=Sex.FEMALE,
        age=29.0,
        sibsp=0,
        parch=0,
        fare=211.3375,
        embarked=Embarked.S
    )


class TestSchemas:
    """Test Pydantic schemas."""

    def test_passenger_schema_valid(self, sample_passenger):
        """Test valid passenger schema."""
        passenger = Passenger(**sample_passenger)
        assert passenger.passenger_id == 1
        assert passenger.sex == Sex.FEMALE
        assert passenger.embarked == Embarked.S

    def test_passenger_schema_invalid_age(self):
        """Test invalid age validation."""
        invalid_data = {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 150,  # Invalid age
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S"
        }

        with pytest.raises(ValueError, match="Age must be between 0 and 120"):
            Passenger(**invalid_data)

    def test_passenger_schema_invalid_fare(self):
        """Test invalid fare validation."""
        invalid_data = {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": -10,  # Invalid fare
            "embarked": "S"
        }

        with pytest.raises(ValueError, match="Fare must be non-negative"):
            Passenger(**invalid_data)


class TestModelManager:
    """Test ModelManager functionality."""

    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        manager = ModelManager()
        assert manager.model is None
        assert manager.scaler is None
        assert manager.model_version == "1.0.0"

    @patch('api.utils.joblib.load')
    @patch('builtins.open')
    def test_load_model_success(self, mock_open, mock_load):
        """Test successful model loading."""
        manager = ModelManager()

        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            # Mock JSON loading
            mock_open.return_value.__enter__.return_value.read.return_value = '["feature1", "feature2"]'

            result = manager.load_model()
            assert result is True
            assert manager.is_loaded() is True

    def test_load_model_failure(self):
        """Test model loading failure."""
        manager = ModelManager()

        # Mock file not existing
        with patch('pathlib.Path.exists', return_value=False):
            result = manager.load_model()
            assert result is False
            assert manager.is_loaded() is False

    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        manager = ModelManager()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            manager.predict({"passenger_id": 1})


class TestSecurityManager:
    """Test SecurityManager functionality."""

    def test_security_manager_no_keys(self):
        """Test security manager with no API keys."""
        manager = SecurityManager()
        assert manager.validate_api_key(None) is True
        assert manager.validate_api_key("any-key") is True

    def test_security_manager_with_keys(self):
        """Test security manager with API keys."""
        api_keys = ["key1", "key2"]
        manager = SecurityManager(api_keys)

        assert manager.validate_api_key("key1") is True
        assert manager.validate_api_key("key2") is True
        assert manager.validate_api_key("invalid-key") is False
        assert manager.validate_api_key(None) is False

    def test_generate_api_key(self):
        """Test API key generation."""
        manager = SecurityManager()
        api_key = manager.generate_api_key("test-user")

        assert isinstance(api_key, str)
        assert "test-user" in api_key


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert collector.request_count == 0
        assert collector.error_count == 0
        assert len(collector.prediction_times) == 0

    def test_record_request_success(self):
        """Test recording successful request."""
        collector = MetricsCollector()
        collector.record_request(0.1, success=True)

        assert collector.request_count == 1
        assert collector.error_count == 0
        assert len(collector.prediction_times) == 1
        assert collector.prediction_times[0] == 0.1

    def test_record_request_error(self):
        """Test recording failed request."""
        collector = MetricsCollector()
        collector.record_request(0.1, success=False)

        assert collector.request_count == 1
        assert collector.error_count == 1
        assert len(collector.prediction_times) == 1

    def test_get_metrics(self):
        """Test getting metrics."""
        collector = MetricsCollector()
        collector.record_request(0.1, success=True)
        collector.record_request(0.2, success=False)

        metrics = collector.get_metrics()

        assert metrics["request_count"] == 2
        assert metrics["error_count"] == 1
        assert metrics["error_rate"] == 0.5
        assert metrics["avg_prediction_time"] == 0.15


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Titanic Survival Prediction API"
        assert data["version"] == "1.0.0"

    def test_examples_endpoint(self, client):
        """Test examples endpoint."""
        response = client.get("/examples")
        assert response.status_code == 200

        data = response.json()
        assert "single_prediction" in data
        assert "batch_prediction" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "uptime" in data
        assert "request_count" in data
        assert "error_count" in data

    @patch('api.utils.get_model_manager')
    def test_predict_endpoint_model_not_loaded(self, mock_get_manager, client):
        """Test prediction endpoint when model is not loaded."""
        # Mock model manager
        mock_manager = Mock()
        mock_manager.is_loaded.return_value = False
        mock_get_manager.return_value = mock_manager

        response = client.post("/predict", json={"passenger": {}})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    @patch('api.utils.get_model_manager')
    @patch('api.utils.get_metrics_collector')
    def test_predict_endpoint_success(self, mock_get_collector, mock_get_manager, client, sample_passenger):
        """Test successful prediction."""
        # Mock model manager
        mock_manager = Mock()
        mock_manager.is_loaded.return_value = True
        mock_manager.predict.return_value = (True, 0.85)
        mock_manager.model_version = "1.0.0"
        mock_get_manager.return_value = mock_manager

        # Mock metrics collector
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector

        response = client.post(
            "/predict", json={"passenger": sample_passenger})
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert data["prediction"]["survived"] is True
        assert data["prediction"]["survival_probability"] == 0.85

    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_passenger = {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 150,  # Invalid age
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S"
        }

        response = client.post(
            "/predict", json={"passenger": invalid_passenger})
        assert response.status_code == 422  # Validation error

    @patch('api.utils.get_model_manager')
    def test_batch_predict_endpoint(self, mock_get_manager, client):
        """Test batch prediction endpoint."""
        # Mock model manager
        mock_manager = Mock()
        mock_manager.is_loaded.return_value = True
        mock_manager.predict_batch.return_value = [(True, 0.85), (False, 0.25)]
        mock_manager.model_version = "1.0.0"
        mock_get_manager.return_value = mock_manager

        batch_data = {
            "passengers": [
                {
                    "passenger_id": 1,
                    "pclass": 1,
                    "sex": "female",
                    "age": 29.0,
                    "sibsp": 0,
                    "parch": 0,
                    "fare": 211.3375,
                    "embarked": "S"
                },
                {
                    "passenger_id": 2,
                    "pclass": 3,
                    "sex": "male",
                    "age": 23.0,
                    "sibsp": 0,
                    "parch": 0,
                    "fare": 7.9250,
                    "embarked": "S"
                }
            ]
        }

        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["total_passengers"] == 2


class TestDataValidation:
    """Test data validation functions."""

    def test_validate_passenger_data_valid(self, sample_passenger):
        """Test validation of valid passenger data."""
        from api.utils import validate_passenger_data

        errors = validate_passenger_data(sample_passenger)
        assert len(errors) == 0

    def test_validate_passenger_data_missing_field(self):
        """Test validation with missing field."""
        from api.utils import validate_passenger_data

        invalid_data = {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            # Missing sibsp, parch, fare, embarked
        }

        errors = validate_passenger_data(invalid_data)
        assert len(errors) > 0
        assert any("Missing required field" in error for error in errors)

    def test_validate_passenger_data_invalid_age(self):
        """Test validation with invalid age."""
        from api.utils import validate_passenger_data

        invalid_data = {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 150,  # Invalid age
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S"
        }

        errors = validate_passenger_data(invalid_data)
        assert len(errors) > 0
        assert any("Age must be between 0 and 120" in error for error in errors)

    def test_validate_passenger_data_invalid_fare(self):
        """Test validation with invalid fare."""
        from api.utils import validate_passenger_data

        invalid_data = {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": -10,  # Invalid fare
            "embarked": "S"
        }

        errors = validate_passenger_data(invalid_data)
        assert len(errors) > 0
        assert any("Fare must be non-negative" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])
