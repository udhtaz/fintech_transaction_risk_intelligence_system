"""
Health check and root endpoints for the API
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])

@router.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Fintech Transaction Risk Intelligence API is running"}
