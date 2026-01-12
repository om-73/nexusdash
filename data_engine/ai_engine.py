import pandas as pd
import numpy as np
import re
import os

def suggest_model_config(df: pd.DataFrame, goal: str):
    """
    Heuristic-based AI engine to suggest model configuration.
    Falls back to logic if OpenAI key is missing.
    """
    
    # 1. Detect Potential Target from Goal
    # goal: "Predict price", "Classify churn", etc.
    goal_lower = goal.lower()
    cols = df.columns.tolist()
    target_col = None
    
    # Simple keyword match
    for col in cols:
        if col.lower() in goal_lower:
            target_col = col
            break
            
    # Fallback: Detect most "target-like" column (last column usually)
    if not target_col:
        target_col = cols[-1]
        
    # 2. Analyze Target
    problem_type = "regression"
    if target_col in df.columns:
        series = df[target_col]
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 20 and series.dtype != 'float':
                 problem_type = "classification"
        else:
            problem_type = "classification"
            
    # 3. Select Features (All others)
    features = [c for c in cols if c != target_col]
        
    # 4. Suggest Algorithm
    algorithm = "rf" # Random Forest is generally best default
    
    return {
        "target_column": target_col,
        "problem_type": problem_type,
        "algorithm": algorithm,
        "feature_columns": features,
        "explanation": f"Based on your goal '{goal}', we selected '{target_col}' as the target. Since it appears to be {'continuous' if problem_type=='regression' else 'categorical'}, we recommend a {problem_type} model."
    }
