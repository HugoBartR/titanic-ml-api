# Titanic Survival Prediction API

A machine learning API that predicts passenger survival on the Titanic using an **ensemble (VotingClassifier) of Random Forest, XGBoost, and Logistic Regression**.

## Features

- **Binary Classification**: Predicts survival (1) or death (0) for Titanic passengers
- **RESTful API**: FastAPI-based API with automatic documentation
- **Model Training**: Automated training pipeline with cross-validation
- **Ensemble Model**: Combines RandomForest, XGBoost, and LogisticRegression for higher accuracy
- **Advanced Feature Engineering**: Includes binning, missing indicators, ticket/cabin parsing, and more
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Health Checks**: Built-in monitoring and health endpoints

## Model Performance (Final Ensemble)

- **Accuracy**: 79.72%
- **Precision**: 75.47%
- **Recall**: 71.43%
- **F1-Score**: 73.39%
- **ROC-AUC**: 81.44%
- **CV Score**: 80.85%

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

## Model Features (Final)

The model uses the following features (with advanced engineering):
- **Passenger Class** (1st, 2nd, 3rd)
- **Sex** (Male/Female)
- **Age** (with missing value imputation)
- **SibSp** (Number of siblings/spouses)
- **Parch** (Number of parents/children)
- **Fare** (Ticket price)
- **Embarked** (Port of embarkation)
- **Title** (extracted from Name, grouped)
- **FamilySize** (SibSp + Parch + 1)
- **IsAlone** (FamilySize == 1)
- **FarePerPerson** (Fare / FamilySize)
- **AgeBin** (binned Age)
- **FareBin** (binned Fare)
- **AgeMissing** (indicator if Age was missing)
- **HasCabin** (indicator if Cabin present)
- **CabinLetter** (first letter of Cabin)
- **TicketPrefix** (prefix from Ticket)

## Model Ensemble

The final model is a **VotingClassifier** (soft voting) combining:
- RandomForestClassifier
- XGBoostClassifier
- LogisticRegression

This ensemble approach improved both accuracy and robustness, leveraging the strengths of each algorithm.

## Model Performance (Detailed)

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 79.72%  |
| Precision   | 75.47%  |
| Recall      | 71.43%  |
| F1-Score    | 73.39%  |
| ROC-AUC     | 81.44%  |
| CV Score    | 80.85%  |

**Top 5 Most Important Features (ensemble average):**
1. Sex_male
2. Title_Mr
3. Sex_female
4. FarePerPerson
5. Fare

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

### Confusion Matrix Interpretation

The confusion matrix shows that the model correctly identifies most survivors and non-survivors. However, there are slightly more false negatives than false positives, meaning the model is more likely to miss a survivor than to incorrectly predict survival for a non-survivor. In the Titanic context, this means the model is somewhat conservative, prioritizing the identification of non-survivors.

This trade-off may be acceptable depending on the business goal: if the priority is to minimize the risk of missing survivors, further tuning or adjusting the classification threshold could be considered.

## Conclusion

### Evaluation Metrics
The final ensemble model achieves an accuracy of **79.72%** and an F1-score of **73.39%** on the validation set. This is a significant improvement over the baseline, thanks to advanced feature engineering and model ensembling. The model is robust and suitable for production deployment.

### Key Features
The most important features identified by the ensemble are:
- **Sex**: Gender is the strongest predictor, as women historically had a higher survival rate.
- **Title**: Extracted from Name, provides social status and gender cues.
- **FarePerPerson**: Normalizes fare by family size, capturing socioeconomic status.
- **Pclass**: Ticket class reflects access to lifeboats and resources.
- **Fare** and **Age**: Indicate socioeconomic status and vulnerability.

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