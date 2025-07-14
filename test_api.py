#!/usr/bin/env python3
"""
Test script for the Titanic Survival Prediction API.

This script tests the API endpoints with example requests
to verify the system is working correctly.
"""

import requests
import json
import time
from typing import Dict, Any


class APITester:
    """Test the Titanic Survival Prediction API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_health(self) -> bool:
        """Test the health endpoint."""
        print("🏥 Testing health endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data}")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False

    def test_root(self) -> bool:
        """Test the root endpoint."""
        print("🏠 Testing root endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Root endpoint: {data}")
                return True
            else:
                print(f"✗ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Root endpoint error: {e}")
            return False

    def test_examples(self) -> bool:
        """Test the examples endpoint."""
        print("📝 Testing examples endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/examples")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Examples endpoint: {len(data)} examples")
                return True
            else:
                print(f"✗ Examples endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Examples endpoint error: {e}")
            return False

    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint."""
        print("🎯 Testing single prediction...")

        passenger_data = {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S"
        }

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"passenger": passenger_data},
                headers={"X-API-Key": "test-key"}  # API key for testing
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Single prediction: {data}")
                return True
            else:
                print(f"✗ Single prediction failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Single prediction error: {e}")
            return False

    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint."""
        print("📦 Testing batch prediction...")

        passengers_data = {
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

        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=passengers_data,
                headers={"X-API-Key": "test-key"}  # API key for testing
            )

            if response.status_code == 200:
                data = response.json()
                print(
                    f"✓ Batch prediction: {len(data['predictions'])} predictions")
                return True
            else:
                print(f"✗ Batch prediction failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Batch prediction error: {e}")
            return False

    def test_model_info(self) -> bool:
        """Test model information endpoint."""
        print("📊 Testing model info...")

        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Model info: {data['model_version']}")
                return True
            else:
                print(f"✗ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Model info error: {e}")
            return False

    def test_metrics(self) -> bool:
        """Test metrics endpoint."""
        print("📈 Testing metrics...")

        try:
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Metrics: {data}")
                return True
            else:
                print(f"✗ Metrics failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Metrics error: {e}")
            return False

    def test_invalid_request(self) -> bool:
        """Test invalid request handling."""
        print("❌ Testing invalid request...")

        try:
            # Test with invalid passenger data
            invalid_data = {
                "passenger": {
                    "passenger_id": 1,
                    "pclass": 1,
                    "sex": "female",
                    "age": 150,  # Invalid age
                    "sibsp": 0,
                    "parch": 0,
                    "fare": 211.3375,
                    "embarked": "S"
                }
            }

            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data,
                headers={"X-API-Key": "test-key"}
            )

            if response.status_code == 422:  # Validation error
                print("✓ Invalid request properly rejected")
                return True
            else:
                print(
                    f"✗ Invalid request not properly handled: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Invalid request test error: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests."""
        print("=" * 50)
        print("🧪 RUNNING API TESTS")
        print("=" * 50)

        tests = [
            ("Health Check", self.test_health),
            ("Root Endpoint", self.test_root),
            ("Examples", self.test_examples),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Model Info", self.test_model_info),
            ("Metrics", self.test_metrics),
            ("Invalid Request", self.test_invalid_request),
        ]

        results = {}

        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"✗ Test failed with exception: {e}")
                results[test_name] = False

        return results

    def print_summary(self, results: Dict[str, bool]):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️ Some tests failed. Please check the API configuration.")


def main():
    """Main function to run API tests."""
    print("🚢 Titanic Survival Prediction API Tester")
    print("=" * 50)

    # Wait a bit for API to start
    print("Waiting for API to be ready...")
    time.sleep(2)

    # Create tester and run tests
    tester = APITester()
    results = tester.run_all_tests()
    tester.print_summary(results)

    # Exit with appropriate code
    if all(results.values()):
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)


if __name__ == "__main__":
    main()
