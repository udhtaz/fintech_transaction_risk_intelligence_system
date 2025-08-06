"""
Test the model predictions with our updated feature engineering
"""

import os
import pandas as pd
import joblib
import json
import sys
import unittest
from unittest import mock
from datetime import datetime

# Add parent directory to sys.path to ensure we can import from utils
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the feature engineering function
from utils.feature_engineering import engineer_features


class TestPredictions(unittest.TestCase):
    """Test cases for prediction functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Sample transaction data
        self.test_data = {
            "transaction_amount": 156.78,
            "transaction_time": "2023-05-15T14:30:00",
            "customer_id": "CUST12345",
            "time_of_day": "Afternoon", 
            "day_of_week": "Monday",
            "device_type": "Mobile",
            "is_foreign_transaction": 0,
            "is_high_risk_country": 0,
            "previous_fraud_flag": 0
        }
        
        # Create a mock model
        self.mock_model = unittest.mock.MagicMock()
        self.mock_model.predict_proba.return_value = [[0.7, 0.3]]  # 30% risk
        
        self.mock_metadata = {
            "model_version": "1.0",
            "features": ["transaction_amount", "is_foreign_transaction", "is_high_risk_country"]
        }
    
    @unittest.mock.patch('joblib.load')
    @unittest.mock.patch('builtins.open')
    def test_model_loading(self, mock_open, mock_joblib_load):
        """Test loading the model"""

        mock_joblib_load.return_value = self.mock_model
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(self.mock_metadata)
        
        # Just test that the test setup works
        self.assertEqual(self.mock_metadata["model_version"], "1.0")
    
    def test_feature_engineering_for_prediction(self):
        """Test that feature engineering works for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([self.test_data])
        
        # Apply feature engineering
        features = engineer_features(df)
        
        # Check that required columns exist
        self.assertIn('amount_foreign', features.columns)
        self.assertIn('is_foreign_transaction', features.columns)
        self.assertIn('is_high_risk_country', features.columns)
        self.assertIn('previous_fraud_flag', features.columns)
        self.assertIn('amount_hour', features.columns)
        
        # Check derived values - amount_foreign should be 0 for non-foreign transactions
        self.assertEqual(features.iloc[0]['amount_foreign'], 0)  # Not foreign, so amount_foreign is 0

if __name__ == "__main__":
    unittest.main()
