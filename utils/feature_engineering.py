""" 
Feature Engineering Module for Fraud Detection System

This module contains functions to engineer features for the fraud detection model.
This implementation is aligned with the trained model's expected features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for fraud detection exactly matching the trained model expectations.

    The model expects these specific features:
    - amount_foreign
    - is_foreign_transaction
    - is_high_risk_country
    - previous_fraud_flag
    - risk_score
    - rolling_std_amount
    - rolling_mean_amount
    - hours_since_last_tx
    - amount_hour

    Args:
        df (pd.DataFrame): Input dataframe with raw transaction data

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure all required columns exist (even if empty)
    required_columns = [
        'is_foreign_transaction', 
        'is_high_risk_country', 
        'previous_fraud_flag',
        'risk_score'
    ]

    for col in required_columns:
        if col not in result_df.columns:
            result_df[col] = 0

    # Convert categorical risk indicators to numeric if they're strings
    for col in ['is_foreign_transaction', 'is_high_risk_country', 'previous_fraud_flag']:
        if col in result_df.columns and result_df[col].dtype == 'object':
            result_df[col] = (
                result_df[col]
                .map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0})
                .fillna(0)
                .astype(int)
            )

    # Process transaction time if available
    if 'transaction_time' in result_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(result_df['transaction_time']):
            result_df['transaction_time'] = pd.to_datetime(
                result_df['transaction_time'], errors='coerce'
            )
        result_df['hour'] = result_df['transaction_time'].dt.hour
    else:
        result_df['hour'] = 12  # Default to noon

    # Standardize column names
    if 'transaction_amount' in result_df.columns and 'amount' not in result_df.columns:
        result_df['amount'] = result_df['transaction_amount']
    elif 'amount' not in result_df.columns:
        result_df['amount'] = 0

    # Amount-based features
    result_df['amount_foreign'] = result_df['amount'] * result_df.get('is_foreign_transaction', 0)
    result_df['amount_hour']    = result_df['amount'] * result_df['hour']

    # Temporal user‐based features
    user_col = 'customer_id' if 'customer_id' in result_df.columns else 'user_id' if 'user_id' in result_df.columns else None
    if user_col and 'transaction_time' in result_df.columns:
        result_df = result_df.sort_values([user_col, 'transaction_time'])
        grouped = result_df.groupby(user_col)
        result_df['hours_since_last_tx'] = grouped['transaction_time'].diff().dt.total_seconds() / 3600
        result_df['rolling_mean_amount'] = grouped['amount'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        result_df['rolling_std_amount']  = grouped['amount'].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    else:
        result_df['hours_since_last_tx']  = 0
        result_df['rolling_mean_amount'] = result_df['amount']
        result_df['rolling_std_amount']  = 0

    # Default or compute risk_score if missing
    if 'risk_score' not in result_df.columns:
        factors = []
        if 'is_foreign_transaction' in result_df: factors.append(result_df['is_foreign_transaction'])
        if 'is_high_risk_country' in result_df: factors.append(result_df['is_high_risk_country'])
        if 'previous_fraud_flag' in result_df: factors.append(result_df['previous_fraud_flag'] * 2)
        result_df['risk_score'] = sum(factors) / len(factors) if factors else 0

    # Final cleanup
    result_df = result_df.fillna(0)
    required_features = [
        'amount_foreign',
        'is_foreign_transaction',
        'is_high_risk_country',
        'previous_fraud_flag',
        'risk_score',
        'rolling_std_amount',
        'rolling_mean_amount',
        'hours_since_last_tx',
        'amount_hour'
    ]
    return result_df[required_features].copy()
