"""
Unit tests for the dashboard components
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path to ensure we can import from dashboard
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import dashboard utilities
from dashboard.utils.prediction import predict_transaction, get_risk_level

class TestDashboardComponents(unittest.TestCase):
    """Test cases for dashboard components"""

    def setUp(self):
        """Set up test data"""
        # Sample transaction data
        self.transaction_data = {
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
        
        # Mock model and metadata
        self.mock_model = MagicMock()
        self.mock_model.predict_proba.return_value = [[0.7, 0.3]]  # 30% risk
        
        self.mock_metadata = {
            "model_version": "1.0",
            "features": ["transaction_amount", "is_foreign_transaction", "is_high_risk_country"]
        }

    def test_risk_level_determination(self):
        """Test risk level determination function"""
        # Test different risk levels
        low_risk_level, low_risk_color = get_risk_level(0.2)
        med_risk_level, med_risk_color = get_risk_level(0.5)
        high_risk_level, high_risk_color = get_risk_level(0.8)
        
        # Verify the risk levels and colors
        self.assertEqual(low_risk_level, "Low Risk")
        self.assertEqual(low_risk_color, "green")
        
        self.assertEqual(med_risk_level, "Medium Risk")
        self.assertEqual(med_risk_color, "orange")
        
        self.assertEqual(high_risk_level, "High Risk")
        self.assertEqual(high_risk_color, "red")
        
    def test_model_preparation(self):
        """Test model setup in the dashboard"""
        # Create a mock DataFrame
        test_df = pd.DataFrame({
            'transaction_amount': [100.0],
            'is_foreign_transaction': [0]
        })
        
        # Set up model mock
        model_mock = MagicMock()
        model_mock.predict_proba.return_value = [[0.7, 0.3]]
        
        # Verify model mock works as expected
        self.assertEqual(model_mock.predict_proba(test_df)[0][1], 0.3)
        
    def test_predict_transaction_function(self):
        """Test the predict_transaction function"""
        # Create a sample transaction
        sample_tx = {
            "transaction_amount": 156.78,
            "transaction_time": "2023-05-15T14:30:00",
            "customer_id": "CUST12345",
            "is_foreign_transaction": 0,
            "is_high_risk_country": 0,
            "previous_fraud_flag": 0
        }
        
        # Mock the model
        model_mock = MagicMock()
        model_mock.predict_proba.return_value = np.array([[0.7, 0.3]])
        
        # Call predict_transaction
        result = predict_transaction(model_mock, sample_tx)
        
        # Verify the result
        self.assertEqual(result[0], 0.3)
        self.assertTrue(model_mock.predict_proba.called)


if __name__ == '__main__':
    unittest.main()
