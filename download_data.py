#!/usr/bin/env python3
"""
Download Titanic dataset for training.

This script downloads the Titanic dataset from a reliable source
and prepares it for the training pipeline.
"""

import os
import pandas as pd
import requests
from pathlib import Path


def download_titanic_data():
    """Download Titanic dataset from reliable source."""

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # URLs for different data sources
    data_sources = [
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    ]

    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"

    print("Downloading Titanic dataset...")

    for url in data_sources:
        try:
            print(f"Trying to download from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Read the data
            df = pd.read_csv(url)

            # Check if this looks like Titanic data
            expected_columns = ['PassengerId', 'Survived',
                                'Pclass', 'Name', 'Sex', 'Age']
            if all(col in df.columns for col in expected_columns):
                print("✓ Valid Titanic dataset found!")

                # Split into train and test sets (80/20 split)
                train_size = int(0.8 * len(df))
                train_df = df.iloc[:train_size]
                test_df = df.iloc[train_size:]

                # Save datasets
                train_df.to_csv(train_file, index=False)
                test_df.to_csv(test_file, index=False)

                print(f"✓ Dataset saved:")
                print(
                    f"  - Training set: {train_file} ({len(train_df)} samples)")
                print(f"  - Test set: {test_file} ({len(test_df)} samples)")

                # Print dataset info
                print("\nDataset Information:")
                print(f"  - Total samples: {len(df)}")
                print(f"  - Features: {list(df.columns)}")
                print(f"  - Missing values: {df.isnull().sum().sum()}")
                print(f"  - Survival rate: {df['Survived'].mean():.2%}")

                return True

        except Exception as e:
            print(f"✗ Failed to download from {url}: {e}")
            continue

    print("✗ Failed to download dataset from all sources")
    return False


def create_sample_data():
    """Create sample data for testing if download fails."""
    print("Creating sample data for testing...")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create sample Titanic data
    sample_data = {
        'PassengerId': range(1, 892),
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] * 89 + [0, 1],
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2] * 89 + [3, 1],
        'Name': [f'Passenger {i}' for i in range(1, 892)],
        'Sex': ['male', 'female'] * 445 + ['male'],
        'Age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14] * 89 + [22, 38],
        'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1] * 89 + [1, 1],
        'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0] * 89 + [0, 0],
        'Ticket': [f'Ticket {i}' for i in range(1, 892)],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07] * 89 + [7.25, 71.28],
        'Cabin': [None] * 891,
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C'] * 89 + ['S', 'C']
    }

    df = pd.DataFrame(sample_data)

    # Split into train and test
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Save datasets
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    print("✓ Sample data created:")
    print(
        f"  - Training set: {data_dir / 'train.csv'} ({len(train_df)} samples)")
    print(f"  - Test set: {data_dir / 'test.csv'} ({len(test_df)} samples)")


def main():
    """Main function to download or create Titanic dataset."""
    print("=" * 50)
    print("TITANIC DATASET DOWNLOADER")
    print("=" * 50)

    # Try to download real data
    if download_titanic_data():
        print("\n✓ Dataset download completed successfully!")
    else:
        print("\n⚠ Could not download real dataset, creating sample data...")
        create_sample_data()
        print("\n⚠ Using sample data for testing purposes.")
        print("   For production use, please obtain the real Titanic dataset.")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
