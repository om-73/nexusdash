from fastapi import APIRouter, HTTPException
import os
import sys

# Adjust path to find ai_engine (assuming standard structure)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from ai_engine import process_ai_query
from ...core import state
from ...schemas.actions import AIProcessRequest

router = APIRouter()

@router.post("/process")
def process_query(request: AIProcessRequest):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        return process_ai_query(df, request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
