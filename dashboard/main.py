"""
Streamlit Dashboard for Fintech Transaction Risk Intelligence System

This app provides a user-friendly interface to:
1. Input new transactions and get fraud risk scores
2. Visualize key features and predictions
3. Show daily/weekly fraud trend analysis
"""

import streamlit as st
import os
import sys

# Configure imports to work with the way the dashboard is run
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))

# Import utilities and pages - use relative imports properly
from dashboard.utils.data_loader import load_model, load_sample_data
from dashboard.pages.transaction_analyzer import show_transaction_analyzer
from dashboard.pages.trend_analysis import show_trend_analysis
from dashboard.pages.model_insights import show_model_insights
from dashboard.pages.about import show_about

# Set page configuration
st.set_page_config(
    page_title="Fintech Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function to run the Streamlit app"""
    # Title and description
    st.title("💳 Fintech Transaction Risk Intelligence System")
    st.markdown("""
    This dashboard provides real-time fraud risk scoring for financial transactions.
    Use the sidebar to navigate between different sections of the dashboard.
    """)
    
    # Load model and data
    model, metadata = load_model()
    sample_data = load_sample_data()
    
    if model is None or metadata is None:
        st.error("Failed to load model or metadata. Please check the model files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Transaction Analyzer", "Trend Analysis", "Model Insights", "About"])
    
    # Display selected page
    if page == "Transaction Analyzer":
        show_transaction_analyzer(model, metadata)
    elif page == "Trend Analysis":
        show_trend_analysis(sample_data)
    elif page == "Model Insights":
        show_model_insights(model, metadata, sample_data)
    else:  # About
        show_about()

if __name__ == "__main__":
    main()
