"""
Transaction analyzer page for the dashboard
"""

import streamlit as st
import pandas as pd
import datetime
import json
import os
import sys

# Configure imports to work with the way the dashboard is run
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, root_dir)

from dashboard.utils.prediction import predict_transaction, get_risk_level
from dashboard.utils.visualization import generate_shap_explanation, plot_shap_waterfall

def show_transaction_analyzer(model, metadata):
    """Display transaction analyzer page"""
    st.header("Transaction Risk Analyzer")
    st.markdown("Enter transaction details to get a real-time fraud risk assessment.")
    
    # Two tabs: Form Input and JSON Input
    tab1, tab2 = st.tabs(["Form Input", "JSON Input"])
    
    with tab1:
        # Create a form for transaction input
        with st.form("transaction_form"):
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Transaction details
                transaction_amount = st.number_input("Transaction Amount", min_value=0.01, value=100.00, step=10.0)
                transaction_time = st.date_input("Transaction Date", value=datetime.date.today())
                time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
                day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
            with col2:
                # Additional features based on your dataset
                device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
                is_foreign = st.selectbox("Foreign Transaction", ["No", "Yes"])
                is_high_risk_country = st.selectbox("High-Risk Country", ["No", "Yes"])
                previous_fraud = st.selectbox("Previous Fraud Flag", ["No", "Yes"])
                customer_id = st.text_input("Customer ID", value="CUST12345")
                merchant_category = st.selectbox("Merchant Category", ["Retail", "Food", "Travel", "Entertainment", "Services"])
            
            # Submit button
            submitted = st.form_submit_button("Analyze Transaction")
            
            if submitted:
                # Prepare input data
                input_data = {
                    "transaction_amount": transaction_amount,
                    "transaction_time": pd.to_datetime(transaction_time),
                    "time_of_day": time_of_day,
                    "day_of_week": day_of_week,
                    "device_type": device_type,
                    "is_foreign_transaction": 1 if is_foreign == "Yes" else 0,
                    "is_high_risk_country": 1 if is_high_risk_country == "Yes" else 0,
                    "previous_fraud_flag": 1 if previous_fraud == "Yes" else 0,
                    "customer_id": customer_id,
                    "merchant_category": merchant_category
                }
                
                # Display results
                display_prediction_results(model, metadata, input_data)
    
    with tab2:
        # JSON input for advanced users
        st.subheader("JSON Input")
        st.markdown("Paste a JSON object representing the transaction.")
        
        # Load sample JSON input
        try:
            # Get the absolute path to the root directory and then to the sample file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            sample_path = os.path.join(root_dir, 'utils', 'sample_input.json')
            with open(sample_path, 'r') as f:
                sample_json = json.load(f)
            sample_str = json.dumps(sample_json, indent=2)
        except Exception as e:
            print(f"Error loading sample JSON: {e}")
            sample_str = '{\n  "amount": 100,\n  "time_of_day": "Morning"\n}'
        
        json_input = st.text_area("Transaction JSON", value=sample_str, height=300)
        
        if st.button("Analyze JSON"):
            try:
                input_data = json.loads(json_input)
                
                # Convert strings to appropriate types if needed
                if "transaction_time" in input_data and isinstance(input_data["transaction_time"], str):
                    input_data["transaction_time"] = pd.to_datetime(input_data["transaction_time"])
                
                # Display results
                display_prediction_results(model, metadata, input_data, show_api_response=True)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.markdown("Please check your JSON format and ensure it contains the required fields.")

def display_prediction_results(model, metadata, input_data, show_api_response=False):
    """Display prediction results for a transaction"""
    # Make prediction
    risk_score = predict_transaction(model, input_data)[0]
    risk_level, risk_color = get_risk_level(risk_score)
    
    # Display results
    st.subheader("Risk Assessment Results")
    
    # Create columns for risk score and level
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Risk Score", f"{risk_score:.2%}")
        
    with res_col2:
        st.markdown(f"<h3 style='color: {risk_color};'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
    
    # SHAP explanation code is commented out to avoid display issues
    # The code is preserved for future use if needed
    """
    # Generate SHAP explanation
    shap_values, feature_names, X_transformed = generate_shap_explanation(model, input_data, metadata['features'])
    
    # Display SHAP plots
    st.subheader("Explanation of Risk Factors")
    waterfall_plot = plot_shap_waterfall(shap_values, X_transformed, feature_names)
    st.image(f"data:image/png;base64,{waterfall_plot}", caption="Impact of Each Factor on Risk Score")
    """
    
    # Display API-like response if requested
    if show_api_response:
        st.subheader("API Response")
        response = {
            "status": "success",
            "prediction": int(risk_score >= 0.5),  # Binary prediction
            "probability": float(risk_score),
            "explanation": {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "is_fraudulent": bool(risk_score >= 0.5),
                "confidence": float(max(risk_score, 1-risk_score)),
                "model_version": metadata['model_version'],
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }
        st.json(response)
