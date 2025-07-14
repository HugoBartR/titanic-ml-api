#!/usr/bin/env python3
"""
Complete Titanic ML System Runner

This script orchestrates the complete ML pipeline:
1. Download/prepare data
2. Train the model
3. Start the API server
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path


def print_banner():
    """Print system banner."""
    print("=" * 60)
    print("🚢 TITANIC SURVIVAL PREDICTION SYSTEM")
    print("=" * 60)
    print("Complete ML Pipeline & API")
    print("=" * 60)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'fastapi',
        'uvicorn', 'structlog', 'psutil'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False

    print("✓ All dependencies are installed!")
    return True


def download_data():
    """Download or prepare the dataset."""
    print("\n📥 Downloading/preparing dataset...")

    data_dir = Path("data")
    train_file = data_dir / "train.csv"

    if train_file.exists():
        print("✓ Dataset already exists")
        return True

    try:
        # Run the download script
        result = subprocess.run([sys.executable, "download_data.py"],
                                capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Dataset prepared successfully")
            return True
        else:
            print(f"✗ Failed to prepare dataset: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error preparing dataset: {e}")
        return False


def train_model():
    """Train the ML model."""
    print("\n🤖 Training model...")

    model_dir = Path("model")
    model_file = model_dir / "model.pkl"

    if model_file.exists():
        print("✓ Model already exists, skipping training")
        return True

    try:
        # Run the training script
        result = subprocess.run([sys.executable, "train.py"],
                                capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Model trained successfully")
            print(result.stdout)
            return True
        else:
            print(f"✗ Training failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error during training: {e}")
        return False


def start_api():
    """Start the API server."""
    print("\n🌐 Starting API server...")

    try:
        # Change to api directory
        os.chdir("api")

        # Start the API server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])

        print("✓ API server started!")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🏥 Health Check: http://localhost:8000/health")
        print("📊 Metrics: http://localhost:8000/metrics")
        print("\nPress Ctrl+C to stop the server")

        return process

    except Exception as e:
        print(f"✗ Error starting API: {e}")
        return None


def main():
    """Main function to run the complete system."""
    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Download data
    if not download_data():
        print("Failed to prepare dataset")
        sys.exit(1)

    # Train model
    if not train_model():
        print("Failed to train model")
        sys.exit(1)

    # Start API
    api_process = start_api()
    if not api_process:
        print("Failed to start API")
        sys.exit(1)

    try:
        # Wait for the API process
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api_process.terminate()
        api_process.wait()
        print("✓ System stopped")


if __name__ == "__main__":
    main()
