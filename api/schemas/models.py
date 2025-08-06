"""
Pydantic schemas for the Fintech Transaction Risk Intelligence API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

class Transaction(BaseModel):
    """Schema for a single transaction to evaluate"""
    # Required fields by the model
    transaction_amount: float = Field(..., description="Transaction amount (required)")
    is_foreign_transaction: int = Field(0, description="Whether transaction is international (1) or domestic (0)")
    is_high_risk_country: int = Field(0, description="Whether transaction is from a high-risk country (1) or not (0)")
    previous_fraud_flag: int = Field(0, description="Whether user had previous fraudulent activity (1) or not (0)")
    risk_score: Optional[float] = Field(None, description="Pre-computed risk score (if available)")
    
    # Important fields for temporal features
    transaction_time: Optional[str] = Field(None, description="Transaction timestamp (ISO format)")
    customer_id: Optional[str] = Field(None, description="Customer identifier (used for temporal features)")
    
    # Additional contextual fields
    time_of_day: Optional[str] = Field(None, description="Time of day category (Morning, Afternoon, Evening, Night)")
    day_of_week: Optional[str] = Field(None, description="Day of week")
    device_type: Optional[str] = Field(None, description="Device used for transaction")
    merchant_category: Optional[str] = Field(None, description="Category of the merchant")
    
    # Legacy field support (for backward compatibility)
    amount: Optional[float] = Field(None, description="Transaction amount (legacy field)")
    user_id: Optional[str] = Field(None, description="User identifier (legacy field)")
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class BatchTransactions(BaseModel):
    """Schema for batch processing of multiple transactions"""
    transactions: List[Transaction] = Field(..., description="List of transactions to evaluate")

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    status: str = Field(..., description="Status of the prediction request")
    prediction: int = Field(..., description="Binary prediction (1=fraudulent, 0=legitimate)")
    probability: float = Field(..., description="Probability of the transaction being fraudulent")
    explanation: Dict[str, Any] = Field(..., description="Explanation of the prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    status: str = Field(..., description="Status of the batch prediction request")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
class ModelInfoResponse(BaseModel):
    """Schema for model information response"""
    status: str = Field(..., description="Status of the request")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    features: List[str] = Field(..., description="Features used by the model")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
