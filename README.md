# Fintech Transaction Risk Intelligence System

## Overview
This project implements a comprehensive risk intelligence system for fintech transactions. It includes:
1. A fully-featured risk scoring model notebook
2. A Streamlit dashboard for visualizing and monitoring transactions
3. A REST API for real-time risk scoring of transactions

## Interface

**STREAMLIT UI**

<img width="1455" height="899" alt="Screenshot 2025-08-06 at 6 32 48 PM" src="https://github.com/user-attachments/assets/fbab15ef-af01-4057-aba4-cfcb838ca080" />


_________________________

**FASTAPI DOCUMENTATION**

<img width="1122" height="846" alt="Screenshot 2025-08-06 at 6 28 47 PM" src="https://github.com/user-attachments/assets/ae621e86-9701-47e3-86d2-100e22ebd443" />

_________________________

## Project Structure
```
fintech_transaction_risk_intelligence_system/
├── api/                             # API module with modular structure
│   ├── config.py                    # API configuration settings
│   ├── helpers/                     # Helper functions
│   │   └── prediction.py            # Prediction utility functions
│   ├── main.py                      # Main API application
│   ├── routers/                     # API routes
│   │   ├── health.py                # Health check endpoints
│   │   ├── model.py                 # Model info endpoints
│   │   └── prediction.py            # Prediction endpoints
│   └── schemas/                     # Pydantic models
│       └── models.py                # Request/response schemas
│
├── dashboard/                       # Dashboard module with modular structure
│   ├── components/                  # Reusable UI components
│   ├── main.py                      # Main dashboard application
│   ├── pages/                       # Dashboard pages
│   │   ├── about.py                 # About page
│   │   ├── model_insights.py        # Model performance page
│   │   ├── transaction_analyzer.py  # Transaction analysis page
│   │   └── trend_analysis.py        # Trend analysis page
│   └── utils/                       # Dashboard utilities
│       ├── data_loader.py           # Data loading functions
│       ├── prediction.py            # Prediction utilities
│       └── visualization.py         # Visualization utilities
│
├── datasets/                        # Data files
│   └── fintech_sample_fintech_transactions.xls  # Sample dataset
│
├── docker/                          # Docker configuration
│   ├── Dockerfile                   # Docker image definition
│   └── docker-compose.yml           # Docker compose config
│
├── models/                          # Model files
│   ├── fraud_detection_model.pkl    # Serialized model
│   └── model_metadata.json          # Model metadata
│
├── scripts/                         # Analysis scripts
│   └── risk_model.ipynb             # Model training notebook
│
├── utils/                           # Shared utilities
│   └── feature_engineering.py       # Feature engineering module
│
├── tests/
│   └──__init__.py                   # Makes tests a proper Python package
│   └──run_tests.py                  # Script to run all tests
│   └──test_api.py                   # Tests for API endpoints
│   └──test_dashboard.py             # Tests for dashboard components
│   └──test_feature_engineering.py   # Tests for feature engineering
│   └──test_predictions.py           # Prediction tests
│
├── run_api.py                       # API runner script
├── run_dashboard.py                 # Dashboard runner script
├── run_docker.sh                    # Docker start script
└── requirements.txt                 # Project dependencies
```

## Key Features

### 1. Risk Scoring Model (`scripts/risk_model.ipynb`)
- **Data Preprocessing**: Handles missing values, outliers, and data transformation
- **Feature Engineering**: Creates temporal features, interaction features, and categorical encodings
- **Advanced Modeling**: Implements ensemble models with hyperparameter tuning
- **Class Imbalance Handling**: Uses SMOTE and class weighting
- **Temporal Pattern Analysis**: Detects unusual patterns in transaction sequences
- **Model Evaluation**: Comprehensive metrics and performance visualization
- **Model Explainability**: SHAP values for understanding model decisions
- **Clustering Analysis**: Unsupervised learning for customer segmentation

### 2. Streamlit Dashboard (`dashboard/`)
- **Transaction Analyzer**: Real-time risk scoring for individual transactions
- **Risk Trends**: Visualization of risk patterns over time
- **Model Insights**: Explainable AI component showing feature importance
- **User Clustering**: Segmentation of users based on transaction patterns
- **Interactive Filters**: Explore data by different dimensions

### 3. REST API (`api/`)
- **Real-time Scoring**: Endpoint for single transaction risk evaluation
- **Batch Processing**: Process multiple transactions in one request
- **Explainability**: Returns reasoning behind risk scores
- **Model Information**: Endpoint for model metadata and performance metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/udhtaz/fintech_transaction_risk_intelligence_system.git
```

2. Install dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Running the Application

### Option 1: Running Components Separately

1. Run the API server:
```bash
python run_api.py
```

2. Run the Streamlit dashboard:
```bash
python run_dashboard.py
```

### Option 2: Running with Docker

1. Start both services using Docker:
```bash
./run_docker.sh
```

## API Usage

### 1. Make a prediction for a single transaction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 156.78,
    "transaction_time": "2023-05-15T14:30:00",
    "time_of_day": "Afternoon",
    "day_of_week": "Monday",
    "device_type": "Mobile",
    "is_foreign_transaction": 0,
    "user_id": "U12345",
    "merchant_category": "Retail"
  }'
```

### 2. Get model information:

```bash
curl -X GET "http://localhost:8000/model/info"
```

## Dashboard Usage

Navigate to the dashboard at http://localhost:8501 and use the following features:

- **Transaction Risk Analysis**: Enter transaction details to get a risk score
- **User Analysis**: View patterns for specific users
- **Trend Analysis**: Visualize risk patterns over time
- **Model Insights**: Explore feature importance

## Testing

Run the test suite to verify the system components:

```bash
cd tests
python run_tests.py
```

Individual tests can also be run:

```bash
python -m unittest tests/test_predictions.py
python -m unittest tests/test_feature_engineering.py
python -m unittest tests/test_api.py
python -m unittest tests/test_dashboard.py
```

## Technology Stack
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **API**: FastAPI
- **Model Serialization**: Joblib
- **Containerization**: Docker
- **Testing**: Unittest, TestClient

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
