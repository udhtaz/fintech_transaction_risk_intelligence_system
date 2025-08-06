"""
Visualization utilities for model explanations and plots
"""

import matplotlib.pyplot as plt
import shap
import io
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import os
import sys
import streamlit as st

# Import custom feature engineering
import importlib.util
import os
import sys

# Get absolute path to the feature engineering module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
feature_engineering_path = os.path.join(root_dir, 'utils', 'feature_engineering.py')

# Use importlib to load the module directly
spec = importlib.util.spec_from_file_location("feature_engineering", feature_engineering_path)
feature_engineering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_engineering)

# Get the engineer_features function
engineer_features = feature_engineering.engineer_features

def generate_shap_explanation(model, input_data, feature_list):
    """
    Generate SHAP explanation for a single transaction
    
    Args:
        model: Trained model pipeline
        input_data: Dictionary or DataFrame with transaction data
        feature_list: List of features expected by the model
    
    Returns:
        Tuple of (shap_values, feature_names, X_transformed)
    """
    # Convert input_data to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Ensure input has all required features
    for feature in feature_list:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    if 'user_id' in input_df.columns and 'customer_id' not in input_df.columns:
        input_df['customer_id'] = input_df['user_id']
    
    # Ensure required fields exist
    required_fields = ['is_foreign_transaction', 'is_high_risk_country', 'previous_fraud_flag']
    for field in required_fields:
        if field not in input_df.columns:
            input_df[field] = 0
    
    # Apply feature engineering
    input_df = engineer_features(input_df)
    
    # Get preprocessor and model from pipeline
    try:
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['model']
        
        # Transform input data using preprocessor
        try:
            X_transformed = preprocessor.transform(input_df)
        except Exception as e:
            # If transformation fails, create a simple feature vector
            st.warning(f"Error during feature transformation: {str(e)}")
            X_transformed = np.zeros((1, len(input_df.columns)))
        
        # Create explainer based on model type
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            
            # Handle multi-class output
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # For binary classification, focus on positive class (class 1)
                shap_values = shap_values[1]
        except Exception as e:
            # Fallback to KernelExplainer or provide dummy values
            try:
                background = X_transformed[:100] if X_transformed.shape[0] >= 100 else X_transformed
                explainer = shap.KernelExplainer(classifier.predict_proba, background)
                shap_values = explainer.shap_values(X_transformed)[1]  # Class 1
            except Exception as inner_e:
                st.warning(f"Could not generate SHAP values: {str(inner_e)}")
                shap_values = np.zeros(X_transformed.shape[1])
        
        # Get feature names with error handling
        feature_names = get_feature_names(preprocessor, input_df)
        
        # Make sure dimensions match
        if len(feature_names) != X_transformed.shape[1]:
            st.warning(f"Feature names count ({len(feature_names)}) doesn't match feature count ({X_transformed.shape[1]})")
            # Create generic feature names
            feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
        
    except Exception as e:
        st.error(f"Error generating model explanation: {str(e)}")
        # Return dummy values to prevent the app from crashing
        X_transformed = np.zeros((1, len(input_df.columns)))
        shap_values = np.zeros(len(input_df.columns))
        feature_names = list(input_df.columns)
    
    # Return SHAP values and feature names
    return shap_values, feature_names, X_transformed

def get_feature_names(preprocessor, input_df):
    """Get feature names from preprocessor"""
    feature_names = []
    try:
        # First, try to get the feature names directly from the preprocessor if it has this attribute
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                # Try to get all feature names at once
                return list(preprocessor.get_feature_names_out())
            except:
                pass
                
        # Otherwise, go transformer by transformer
        for name, transformer, cols in preprocessor.transformers_:
            # Filter out columns that don't exist in the input dataframe
            valid_cols = [col for col in cols if col in input_df.columns]
            
            if not valid_cols:
                # Skip this transformer if none of its columns exist
                continue
                
            if hasattr(transformer, 'get_feature_names_out') and hasattr(transformer, 'categories_'):
                # Check if transformer is fitted before using get_feature_names_out
                try:
                    transformed_names = transformer.get_feature_names_out(valid_cols)
                    feature_names.extend(transformed_names)
                except Exception as e:
                    # Fallback for unfitted transformers
                    for col in valid_cols:
                        feature_names.append(f"{col}")
            else:
                # For non-transforming columns or simple transformers
                feature_names.extend(valid_cols)
    except Exception as e:
        # If all else fails, just use column names from input_df
        st.warning(f"Error getting feature names: {str(e)}")
        feature_names = list(input_df.columns)
    
    # Make sure we have at least some feature names
    if not feature_names:
        feature_names = list(input_df.columns)
        
    return feature_names

def plot_shap_summary(shap_values, X, feature_names):
    """
    Generate and save SHAP summary plot
    
    Args:
        shap_values: SHAP values from explainer
        X: Transformed feature data
        feature_names: List of feature names
        
    Returns:
        Base64 encoded PNG image
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    
    # Encode plot as base64 string
    return base64.b64encode(buf.read()).decode()

def plot_shap_waterfall(shap_values, X, feature_names):
    """
    Generate and save SHAP waterfall plot
    
    Args:
        shap_values: SHAP values from explainer
        X: Transformed feature data
        feature_names: List of feature names
        
    Returns:
        Base64 encoded PNG image
    """
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=shap_values.sum(1).mean(), 
                                         data=X[0], 
                                         feature_names=feature_names), show=False)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    
    # Encode plot as base64 string
    return base64.b64encode(buf.read()).decode()
