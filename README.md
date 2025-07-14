# Titanic Survival Prediction API

A machine learning API that predicts passenger survival on the Titanic using a Random Forest classifier.

## Features

- **Binary Classification**: Predicts survival (1) or death (0) for Titanic passengers
- **RESTful API**: FastAPI-based API with automatic documentation
- **Model Training**: Automated training pipeline with cross-validation
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Health Checks**: Built-in monitoring and health endpoints

## Model Performance

- **Accuracy**: 77.62%
- **Precision**: 72.22%
- **Recall**: 69.64%
- **F1-Score**: 70.91%
- **ROC-AUC**: 79.56%

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/HugoBartR/titanic-ml-api.git
   cd titanic-ml-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\Activate.ps1
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Download data and train model**
   ```bash
   python download_data.py
   python train.py
   ```

5. **Run the API**
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```

6. **Access the API**
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**
   ```bash
   docker build -f Dockerfile.api -t titanic-api .
   docker run -p 8000:8000 titanic-api
   ```

## API Endpoints

### Health Check
```http
GET /health
```

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "passenger": {
    "passenger_id": 1,
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.3375,
    "embarked": "S"
  }
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "passengers": [
    {
      "passenger_id": 1,
      "pclass": 1,
      "sex": "female",
      "age": 29,
      "sibsp": 0,
      "parch": 0,
      "fare": 211.3375,
      "embarked": "S"
    }
  ]
}
```

### Metrics
```http
GET /metrics
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. **Tests**: Runs pytest with coverage
2. **Trains Model**: Downloads data and retrains the model
3. **Builds Docker Image**: Creates optimized container
4. **Deploys**: (Configure your deployment target)

### Workflow Triggers
- Push to `main` or `master` branch
- Pull requests to `main` or `master` branch

### Secrets Required
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password/token

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=api --cov-report=html

# Run specific test file
pytest tests/test_predict.py -v
```

## Project Structure

```
ML/
├── api/                    # API application
│   ├── app.py             # FastAPI application
│   ├── schemas.py         # Pydantic models
│   └── utils.py           # Model manager
├── data/                  # Dataset files
├── model/                 # Trained model artifacts
├── tests/                 # Test files
├── .github/workflows/     # CI/CD workflows
├── Dockerfile.api         # API Docker image
├── docker-compose.yml     # Local development
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── train.py              # Model training script
└── download_data.py      # Data download script
```

## Model Features

The model uses the following features:
- **Passenger Class** (1st, 2nd, 3rd)
- **Sex** (Male/Female)
- **Age** (with missing value imputation)
- **SibSp** (Number of siblings/spouses)
- **Parch** (Number of parents/children)
- **Fare** (Ticket price)
- **Embarked** (Port of embarkation)

## Monitoring

The API includes built-in monitoring:
- Request/response logging
- Performance metrics
- Health checks
- Error tracking

## Confusion Matrix

The confusion matrix below shows the performance of the classifier on the validation set:

![Confusion Matrix](model/confusion_matrix.png)

This matrix provides a detailed breakdown of true positives, true negatives, false positives, and false negatives, helping to better understand the model's strengths and weaknesses.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the test files for usage examples

---

## Conclusion

### Evaluation Metrics
The model achieves an accuracy of **77.62%** and an F1-score of **70.91%** on the validation set. This indicates a solid baseline performance, outperforming random guessing and simple heuristics. However, there is room for improvement through advanced feature engineering and hyperparameter tuning. The model is suitable for an initial deployment, but further iterations could enhance its robustness and predictive power for production use.

### Key Features
The most important features identified by the model are:
- **Sex**: Gender is the strongest predictor, as women historically had a higher survival rate.
- **Pclass**: Ticket class reflects access to lifeboats and resources.
- **Fare** and **Age**: Indicate socioeconomic status and vulnerability.
These insights were determined using the RandomForest feature importances.

### Production Deployment & MLOps
For production, I recommend the following stack and best practices:
- **Docker** for containerization and portability.
- **FastAPI** to serve the model as a REST API.
- **CI/CD** with GitHub Actions for automated testing, building, and deployment.
- **Cloud**: Deploy on AWS ECS/Fargate, GCP Cloud Run, or Azure Container Apps for scalability and reliability.
- **MLOps**: Use data/model versioning (DVC/MLflow), monitoring (Prometheus/Grafana/Sentry), alerting, and automated testing.
- **Automation**: The entire pipeline (training, testing, building, deployment) is automated and reproducible.
- **Scalability**: The system can scale horizontally using load balancers and orchestrators like Kubernetes if needed.

This approach ensures a robust, maintainable, and production-ready ML system. 