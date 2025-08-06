"""
Unit tests for feature engineering module
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to sys.path to ensure we can import from utils
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the feature engineering function
from utils.feature_engineering import engineer_features

class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering"""

    def setUp(self):
        """Set up test data"""
        # Create a sample transaction dataframe
        self.test_data = pd.DataFrame({
            'transaction_amount': [100.0, 500.0, 1200.0],
            'transaction_time': [
                pd.to_datetime('2023-05-15 10:30:00'),
                pd.to_datetime('2023-05-16 14:45:00'),
                pd.to_datetime('2023-05-17 22:15:00')
            ],
            'customer_id': ['CUST001', 'CUST002', 'CUST001'],
            'is_foreign_transaction': [0, 1, 0],
            'is_high_risk_country': [0, 1, 0],
            'previous_fraud_flag': [0, 0, 1]
        })

    def test_engineer_features_output(self):
        """Test that feature engineering produces expected output format"""
        result = engineer_features(self.test_data)
        
        # Check if it returns a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if expected engineered features are present based on the actual implementation
        expected_features = [
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
        
        for feature in expected_features:
            with self.subTest(feature=feature):
                self.assertIn(feature, result.columns)

    def test_datetime_features(self):
        """Test that datetime features are correctly extracted"""
        result = engineer_features(self.test_data)
        
        self.assertIn('amount_hour', result.columns)
        self.assertTrue(all(result['amount_hour'] >= 0))  # Should be non-negative

    def test_missing_columns_handling(self):
        """Test handling of data with missing columns"""
        # Create data without some expected columns
        minimal_data = pd.DataFrame({
            'transaction_amount': [100.0],
            'customer_id': ['CUST001']
        })
        
        # Should not raise an exception
        result = engineer_features(minimal_data)
        
        # Check if it added default values
        self.assertIn('is_foreign_transaction', result.columns)
        self.assertIn('is_high_risk_country', result.columns)
        
    def test_amount_foreign_calculation(self):
        """Test that amount_foreign is properly calculated"""
        result = engineer_features(self.test_data)
        
        # Check if amount_foreign feature was created
        self.assertIn('amount_foreign', result.columns)
        
        # Verify that amount_foreign exists - the actual values depend on implementation
        # but we know the first transaction should have 0 as it's not foreign
        self.assertEqual(result.iloc[0]['amount_foreign'], 0)          # 100.0 * 0 (not foreign)


if __name__ == '__main__':
    unittest.main()
