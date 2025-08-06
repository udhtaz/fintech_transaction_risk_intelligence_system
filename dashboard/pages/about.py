"""
About page for the dashboard
"""

import streamlit as st

def show_about():
    """Display about page"""
    st.header("About This Application")
    st.markdown("""
    ### Fintech Transaction Risk Intelligence System
    
    This dashboard is part of an AI-powered fraud detection system designed to identify potentially fraudulent or high-risk financial transactions in real-time.
    
    #### Key Features:
    
    - **Risk Scoring**: Assess transaction risk in real-time
    - **Temporal Analysis**: Identify risk patterns across time periods
    - **Model Explainability**: Understand why transactions are flagged as risky
    - **Interactive Visualization**: Explore fraud trends and patterns
    
    #### How to Use:
    
    1. **Transaction Analyzer**: Input transaction details to get an instant risk assessment
    2. **Trend Analysis**: Explore temporal patterns in fraud rates
    3. **Model Insights**: Understand model performance and feature importance
    
    #### Technical Implementation:
    
    - **Frontend**: Streamlit dashboard
    - **Backend**: REST API using FastAPI (see api_server.py)
    - **Model**: Advanced machine learning model trained on historical transaction data
    - **Explainability**: SHAP (SHapley Additive exPlanations) for transparent predictions
    
    #### Contact:
    
    For technical support or questions, please contact the development team.
    """)
