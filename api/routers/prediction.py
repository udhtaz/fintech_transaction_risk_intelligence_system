"""
Prediction endpoints for the API
"""

from fastapi import APIRouter, HTTPException
import time
from typing import Dict, List
from ..schemas.models import Transaction, BatchTransactions, PredictionResponse, BatchPredictionResponse
from ..helpers.prediction import make_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Make a prediction for a single transaction
    
    Returns the fraud probability and explanation
    """
    try:
        # Convert Pydantic model to dict
        transaction_data = transaction.dict()
        
        # Make prediction
        result = make_prediction(transaction_data)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchTransactions):
    """
    Make predictions for a batch of transactions
    
    Returns predictions for all transactions in the batch
    """
    start_time = time.time()
    
    try:
        results = []
        
        for transaction in batch.transactions:
            # Convert Pydantic model to dict
            transaction_data = transaction.dict()
            
            # Make prediction
            result = make_prediction(transaction_data)
            results.append(result)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "predictions": results,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
