"""
Prediction and risk assessment utilities
"""

import pandas as pd
import numpy as np
import sys
import os

# Import the feature_engineering module directly
import importlib.util
import os
import sys

# Get absolute path to the feature engineering module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
feature_engineering_path = os.path.join(root_dir, 'utils', 'feature_engineering.py')

# Load the module directly
print(f"Loading module from: {feature_engineering_path}")
print(f"File exists: {os.path.exists(feature_engineering_path)}")

# Use importlib to load the module directly
spec = importlib.util.spec_from_file_location("feature_engineering", feature_engineering_path)
feature_engineering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_engineering)

# Get the engineer_features function
engineer_features = feature_engineering.engineer_features

def predict_transaction(model, input_data):
    """
    Make predictions on transaction data
    
    Args:
        model: The trained model
        input_data: Dictionary or DataFrame of transaction data
        
    Returns:
        Risk scores (probabilities)
    """
    # Convert input to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    
    # Handle compatibility between different field names
    if 'amount' in input_df.columns and 'transaction_amount' not in input_df.columns:
        input_df['transaction_amount'] = input_df['amount']
    
    if 'user_id' in input_df.columns and 'customer_id' not in input_df.columns:
        input_df['customer_id'] = input_df['user_id']
    
    # Ensure required fields exist
    required_fields = ['is_foreign_transaction', 'is_high_risk_country', 'previous_fraud_flag']
    for field in required_fields:
        if field not in input_df.columns:
            input_df[field] = 0
    
    # Apply feature engineering
    input_df = engineer_features(input_df)
    
    # Make prediction
    pred_proba = model.predict_proba(input_df)
    
    # Get risk scores (probability of positive class)
    risk_scores = pred_proba[:, 1]
    
    return risk_scores

def get_risk_level(score):
    """
    Convert risk score to risk level and color
    
    Args:
        score: Risk score (0-1)
        
    Returns:
        Tuple of (risk_level, color)
    """
    if score < 0.3:
        return "Low Risk", "green"
    elif score < 0.7:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"
