"""
Trend analysis page for the dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def show_trend_analysis(sample_data):
    """Display trend analysis page"""
    st.header("Fraud Trend Analysis")
    st.markdown("Analysis of fraud patterns over time and across different dimensions.")
    
    if sample_data is not None:
        # Process sample data for visualization
        if 'label_code' not in sample_data.columns:
            np.random.seed(42)  # For reproducibility
            sample_data['label_code'] = np.random.choice([0, 1], size=len(sample_data), p=[0.95, 0.05])
        
        # Ensure we have temporal features
        if 'transaction_time' in sample_data.columns:
            sample_data['transaction_time'] = pd.to_datetime(sample_data['transaction_time'])
            sample_data['date'] = sample_data['transaction_time'].dt.date
            sample_data['hour'] = sample_data['transaction_time'].dt.hour
            sample_data['day_of_week'] = sample_data['transaction_time'].dt.day_name()
        
        # Time period selector
        time_period = st.selectbox("Select Time Period", ["Daily", "Weekly", "Monthly"])
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Temporal Trends", "Feature Distributions", "Risk Heatmap"])
        
        with tab1:
            show_temporal_trends(sample_data, time_period)
        
        with tab2:
            show_feature_distributions(sample_data)
        
        with tab3:
            show_risk_heatmap(sample_data)
    else:
        st.error("Sample data not available. Please upload a dataset or check the file path.")

def show_temporal_trends(data, time_period):
    """Show temporal trends tab content"""
    st.subheader("Fraud Rate Over Time")
    
    if 'date' in data.columns:
        # Group by appropriate time period
        if time_period == "Daily":
            grouped = data.groupby('date')
        elif time_period == "Weekly":
            data['week'] = data['transaction_time'].dt.isocalendar().week
            grouped = data.groupby('week')
        else:  # Monthly
            data['month'] = data['transaction_time'].dt.month
            grouped = data.groupby('month')
        
        # Calculate fraud rate
        fraud_rate = grouped['label_code'].mean()
        transaction_count = grouped.size()
        
        # Create figure with dual y-axis
        fig = go.Figure()
        
        # Add fraud rate line
        fig.add_trace(
            go.Scatter(
                x=fraud_rate.index,
                y=fraud_rate.values,
                name='Fraud Rate',
                line=dict(color='red', width=3)
            )
        )
        
        # Add transaction count bars
        fig.add_trace(
            go.Bar(
                x=transaction_count.index,
                y=transaction_count.values,
                name='Transaction Count',
                opacity=0.5,
                marker_color='blue'
            )
        )
        
        # Set y-axes titles
        fig.update_layout(
            title=f'Fraud Rate and Transaction Volume by {time_period} Period',
            xaxis_title=time_period,
            yaxis=dict(
                title='Fraud Rate',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                range=[0, fraud_rate.max() * 1.1]
            ),
            yaxis2=dict(
                title='Transaction Count',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show fraud rate by day of week
        if 'day_of_week' in data.columns:
            st.subheader("Fraud Rate by Day of Week")
            
            # Order days of week
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # Calculate fraud rate by day
            day_fraud = data.groupby('day_of_week')['label_code'].mean().reindex(days_order)
            
            # Create bar chart
            fig = px.bar(
                x=day_fraud.index, 
                y=day_fraud.values,
                labels={'x': 'Day of Week', 'y': 'Fraud Rate'},
                color=day_fraud.values,
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Temporal data not available in the sample dataset.")

def show_feature_distributions(data):
    """Show feature distributions tab content"""
    st.subheader("Feature Distributions by Risk Level")
    
    # Select feature to visualize
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'label_code']
    
    if numerical_cols:
        selected_feature = st.selectbox("Select Feature", numerical_cols)
        
        # Create histogram with fraud/non-fraud distinction
        fig = px.histogram(
            data,
            x=selected_feature,
            color='label_code',
            marginal="box",
            barmode='overlay',
            labels={'label_code': 'Risk Level'},
            color_discrete_map={0: 'blue', 1: 'red'},
            opacity=0.7,
            nbins=50
        )
        
        fig.update_layout(
            title=f'Distribution of {selected_feature} by Risk Level',
            xaxis_title=selected_feature,
            yaxis_title='Count',
            legend_title='Risk Level',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numerical features available for visualization.")

def show_risk_heatmap(data):
    """Show risk heatmap tab content"""
    st.subheader("Risk Heatmap")
    
    # Create temporal heatmap if data is available
    if 'hour' in data.columns and 'day_of_week' in data.columns:
        # Create pivot table of fraud rate by hour and day
        pivot = pd.pivot_table(
            data, 
            values='label_code', 
            index='hour', 
            columns='day_of_week', 
            aggfunc='mean'
        )
        
        # Reorder columns to standard week order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex(columns=day_order)
        
        # Create heatmap
        fig = px.imshow(
            pivot,
            labels=dict(x="Day of Week", y="Hour of Day", color="Fraud Rate"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="Reds",
            aspect="auto"
        )
        
        fig.update_layout(
            title='Fraud Rate by Hour and Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation Guide:**
        - Darker red cells indicate higher fraud rates
        - Use this heatmap to identify high-risk time periods
        - Consider implementing additional security measures during high-risk hours
        """)
    else:
        st.warning("Temporal data (hour, day of week) not available for heatmap visualization.")
