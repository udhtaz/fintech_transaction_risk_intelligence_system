"""
Helper functions for the Fintech Transaction Risk Intelligence API
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ..config import settings
from utils.feature_engineering import engineer_features

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fraud_detection_api")

# Global variables for model and metadata
model = None
metadata = None

def load_model() -> Tuple[Any, Dict]:
    """
    Load model and metadata from disk
    
    Returns:
        Tuple[Any, Dict]: Loaded model and metadata
    """
    # Check if model files exist
    if not os.path.exists(settings.MODEL_PATH) or not os.path.exists(settings.METADATA_PATH):
        raise FileNotFoundError(f"Model files not found at {settings.MODEL_PATH} or {settings.METADATA_PATH}")
    
    # Load model and metadata
    model = joblib.load(settings.MODEL_PATH)
    
    with open(settings.METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Model loaded successfully. Version: {metadata.get('model_version', 'unknown')}")
    return model, metadata

def preprocess_transaction(transaction_data: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """
    Preprocess a transaction for prediction
    
    Args:
        transaction_data (Union[Dict, List[Dict]]): Transaction data
        
    Returns:
        pd.DataFrame: Preprocessed transaction data
    """
    # Convert to DataFrame
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
    else:
        df = pd.DataFrame(transaction_data)
    
    # Handle compatibility between different field names
    if 'amount' in df.columns and 'transaction_amount' not in df.columns:
        df['transaction_amount'] = df['amount']
    
    if 'user_id' in df.columns and 'customer_id' not in df.columns:
        df['customer_id'] = df['user_id']
    
    # Convert transaction_time to datetime if present
    if 'transaction_time' in df.columns and df['transaction_time'].iloc[0]:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
    
    # Ensure all required fields exist
    required_fields = ['is_foreign_transaction', 'is_high_risk_country', 'previous_fraud_flag']
    for field in required_fields:
        if field not in df.columns:
            df[field] = 0
    
    logger.info(f"Processing transaction data with columns: {df.columns.tolist()}")
    
    # Apply feature engineering
    df = engineer_features(df)
    
    return df

def make_prediction(transaction_data: Union[Dict, List[Dict]]) -> Dict:
    """
    Make a prediction for a transaction
    
    Args:
        transaction_data (Union[Dict, List[Dict]]): Transaction data
        
    Returns:
        Dict: Prediction results
    """
    global model, metadata
    
    start_time = time.time()
    
    # Check if model is loaded
    if model is None:
        try:
            model, metadata = load_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Model not loaded: {str(e)}")
    
    # Preprocess the transaction
    input_df = preprocess_transaction(transaction_data)
    
    # Make prediction
    try:
        pred_proba = model.predict_proba(input_df)
        risk_score = pred_proba[0][1]  # Probability of positive class
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise RuntimeError(f"Error making prediction: {str(e)}")
    
    # Get optimal threshold from metadata or use default
    threshold = metadata.get('optimal_threshold', 0.5)
    
    # Determine prediction and risk level
    is_fraudulent = risk_score >= threshold
    
    if risk_score < settings.LOW_RISK_THRESHOLD:
        risk_level = "Low Risk"
    elif risk_score < settings.HIGH_RISK_THRESHOLD:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    
    # Generate explanation
    explanation = {
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "is_fraudulent": bool(is_fraudulent),
        "confidence": float(max(risk_score, 1-risk_score)),
        "model_version": metadata.get('model_version', 'unknown'),
        "threshold_used": float(threshold),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add top features if available
    try:
        # Get preprocessor and model from pipeline
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['model']
        
        # Transform input data
        X_transformed = preprocessor.transform(input_df)
        
        # Get feature names
        feature_names = get_feature_names(preprocessor, input_df)
        
        # Get feature importance
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = classifier.feature_importances_
            
            # Get top 5 features
            top_indices = feature_importance.argsort()[-5:][::-1]
            top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in top_indices]
            top_importance = [float(feature_importance[i]) for i in top_indices]
            
            explanation["top_features"] = dict(zip(top_features, top_importance))
    except Exception as e:
        logger.warning(f"Could not generate feature importance: {str(e)}")
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "status": "success",
        "prediction": int(is_fraudulent),
        "probability": float(risk_score),
        "explanation": explanation,
        "processing_time_ms": processing_time
    }

def get_feature_names(preprocessor, input_df: pd.DataFrame) -> List[str]:
    """
    Get feature names from preprocessor
    
    Args:
        preprocessor: Scikit-learn preprocessor
        input_df (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of feature names
    """
    feature_names = []
    
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                transformed_names = transformer.get_feature_names_out(cols)
                feature_names.extend(transformed_names)
            except:
                feature_names.extend([f"{name}_{i}" for i in range(transformer.transform(input_df[cols]).shape[1])])
        else:
            feature_names.extend(cols)
    
    return feature_names

# Initialize model on module import
try:
    model, metadata = load_model()
except Exception as e:
    logger.error(f"Error loading model at startup: {str(e)}")
