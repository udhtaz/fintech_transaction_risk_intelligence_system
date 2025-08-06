"""
Unit tests for the API endpoints
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add parent directory to sys.path to ensure we can import from api
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the FastAPI app
from api.main import app

class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints"""

    def setUp(self):
        """Set up test client"""
        self.client = TestClient(app)
        
        # Sample valid transaction
        self.valid_transaction = {
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
        
        # Sample invalid transaction (missing required field)
        self.invalid_transaction = {
            "customer_id": "CUST12345"
        }

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Check basic response structure
        self.assertIsInstance(response.json(), dict)
        
    def test_model_info_endpoint(self):
        """Test the model info endpoint"""
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # The API response structure has model_version nested in model_info
        self.assertIn("model_info", data)
        self.assertIn("model_version", data["model_info"])
        self.assertIn("features", data)
        
    def test_predict_endpoint_valid(self):
        """Test the prediction endpoint with valid data"""
        response = self.client.post("/predict", json=self.valid_transaction)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("explanation", data)
        
    def test_predict_endpoint_invalid(self):
        """Test the prediction endpoint with invalid data"""
        response = self.client.post("/predict", json=self.invalid_transaction)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
    def test_batch_predict_endpoint(self):
        """Test the batch prediction endpoint"""
        batch_data = {"transactions": [self.valid_transaction, self.valid_transaction]}
        response = self.client.post("/predict/batch", json=batch_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["predictions"]), 2)


if __name__ == '__main__':
    unittest.main()
