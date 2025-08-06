"""
Model insights page for the dashboard
"""

import streamlit as st
from ..utils.visualization import generate_shap_explanation, plot_shap_summary

def show_model_insights(model, metadata, sample_data):
    """Display model insights page"""
    st.header("Model Performance and Insights")
    st.markdown("Understand the model's performance metrics and feature importance.")
    
    # Display model metadata
    st.subheader("Model Information")
    st.json(metadata)
    
    # Display model metrics
    if 'model_metrics' in metadata:
        st.subheader("Performance Metrics")
        metrics = metadata['model_metrics']
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
    
    # Display feature importance
    st.subheader("Feature Importance")
    st.markdown("The relative importance of each feature in the model's predictions.")
    
    # Load sample data for feature importance
    if sample_data is not None:
        # Select a sample for SHAP values
        sample_row = sample_data.iloc[0].to_dict()
        
        try:
            # Generate SHAP explanation
            shap_values, feature_names, X_transformed = generate_shap_explanation(model, sample_row, metadata['features'])
            
            # Display SHAP summary plot
            summary_plot = plot_shap_summary(shap_values, X_transformed, feature_names)
            st.image(f"data:image/png;base64,{summary_plot}", caption="Feature Importance (SHAP Values)")
            
            st.markdown("""
            **How to interpret this chart:**
            - Features are ranked by importance from top to bottom
            - Red points indicate higher feature values, blue points indicate lower values
            - Position on x-axis shows impact on prediction (right = higher risk, left = lower risk)
            """)
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            st.markdown("Feature importance visualization is not available.")
    
    # Trade-off Analysis
    st.subheader("Precision-Recall Trade-off")
    st.markdown("""
    In fraud detection, there's an important trade-off between precision and recall:
    
    - **High Precision, Low Recall**: Fewer false alarms but might miss some fraud
    - **High Recall, Low Precision**: Catches most fraud but generates more false alarms
    
    The business impact:
    
    - **False Positives**: Legitimate customers falsely flagged as fraudulent
        - Customer friction and dissatisfaction
        - Lost transaction revenue
        - Unnecessary manual review costs
        
    - **False Negatives**: Fraudulent transactions that slip through
        - Direct financial losses
        - Potential regulatory penalties
        - Brand and reputation damage
        
    The optimal threshold depends on your business priorities and risk tolerance.
    """)
