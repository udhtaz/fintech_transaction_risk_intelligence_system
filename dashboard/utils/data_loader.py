"""
Utility functions for loading models and data
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load model and metadata
@st.cache_resource
def load_model():
    # Get the absolute path to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    model_path = os.path.join(root_dir, 'models', 'fraud_detection_model.pkl')
    metadata_path = os.path.join(root_dir, 'models', 'model_metadata.json')
    
    try:
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load sample data for demonstration
@st.cache_data
def load_sample_data():
    try:
        # Get the absolute path to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        dataset_path = os.path.join(root_dir, 'datasets', 'fintech_sample_fintech_transactions.xls')
        df = pd.read_excel(dataset_path)
        return df
    except Exception as e:
        st.error(f"Sample data file not found: {str(e)}")
        return None
