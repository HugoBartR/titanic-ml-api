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
1. Sex_male (indicates male gender - lower survival rate)
2. Title_Mr (adult male title - lower survival rate)
3. FarePerPerson (socioeconomic status per family member)
4. Fare (ticket price - wealth indicator)
5. Pclass_3 (third class - lower survival rate)

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

## Conclusions

### Model Performance Analysis

The ensemble model achieves **79.72% accuracy** and **73.39% F1-score**, demonstrating strong predictive performance for the Titanic survival classification task. The model's performance metrics indicate:

- **High Precision (75.47%)**: When the model predicts survival, it's correct 75% of the time
- **Good Recall (71.43%)**: The model identifies 71% of actual survivors
- **Strong ROC-AUC (81.44%)**: Excellent discriminative ability between classes
- **Consistent CV Performance (80.85%)**: Low variance across cross-validation folds

### Feature Importance Insights

The top 5 most important features reveal critical survival patterns:

1. **Sex_male** - The strongest predictor, indicating men had significantly lower survival rates (women had priority access to lifeboats)
2. **Title_Mr** - Social status indicator; "Mr" titles (adult men) had poor survival rates compared to women and children
3. **FarePerPerson** - Socioeconomic status normalized by family size, capturing wealth distribution within families
4. **Fare** - Raw ticket price, indicating wealth and access to better accommodations and information
5. **Pclass_3** - Third class passengers had much lower survival rates due to limited access to lifeboats

### Historical Context Validation

These findings align perfectly with historical accounts of the Titanic disaster:
- **Gender Bias**: Women had priority access to lifeboats, explaining why Sex is the top predictor
- **Social Class**: Higher fares and titles indicate better access to lifeboats and information
- **Family Dynamics**: FarePerPerson captures family wealth distribution and survival strategies

### Model Robustness

The ensemble approach provides several advantages:
- **Reduced Overfitting**: Combining three different algorithms improves generalization
- **Better Feature Understanding**: Each algorithm captures different patterns in the data
- **Stable Predictions**: Voting classifier reduces variance in predictions
- **Comprehensive Evaluation**: Cross-validation confirms model reliability

### Business Implications

For real-world applications, this model demonstrates:
- **Actionable Insights**: Clear identification of survival factors
- **Risk Assessment**: Can predict survival probability for similar scenarios
- **Resource Allocation**: Understanding which factors most influence outcomes
- **Historical Analysis**: Validates historical accounts with data-driven evidence

### Production Deployment & MLOps

For production deployment, the system implements industry best practices:

- **Containerization**: Docker ensures consistent deployment across environments
- **API-First Design**: FastAPI provides automatic documentation and type safety
- **Monitoring**: Built-in health checks and performance metrics
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **Scalability**: Horizontal scaling capability with load balancers
- **MLOps Integration**: Model versioning, monitoring, and automated retraining

### Future Improvements

Potential enhancements for the next iteration:
- **Feature Engineering**: Additional domain-specific features (cabin location, ticket patterns)
- **Model Tuning**: Hyperparameter optimization for individual models
- **Ensemble Diversity**: Adding different algorithm types (SVM, Neural Networks)
- **Data Augmentation**: Synthetic data generation for underrepresented classes
- **Real-time Monitoring**: Model drift detection and performance tracking

This comprehensive approach ensures a robust, maintainable, and production-ready ML system that not only predicts accurately but also provides valuable insights into the underlying survival patterns. 