from fastapi import APIRouter, HTTPException
import os
import sys
import pandas as pd
from typing import Dict, Any

from ...core import state
from ...schemas.actions import AutoAgentRunRequest, AutoAgentSampleRequest, AutoAgentPlanRequest
from ...services.agent_service import run_autonomous_agent, generate_sample_dataset
from ...services.utils import get_df_summary

router = APIRouter()

@router.post("/run")
def execute_agent(request: AutoAgentRunRequest):
    df = state.get_active_df()
    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail="No dataset is currently loaded. Please load a dataset or click 'Load Demo Dataset' in the agent studio."
        )

    try:
        result = run_autonomous_agent(
            df=df,
            goal=request.goal or "Full Auto-Pilot: Profile, Clean, Engineer Features & Train Champion Model",
            target_column=request.target_column,
            problem_type=request.problem_type,
            feature_engineering=request.feature_engineering if request.feature_engineering is not None else True,
            outlier_handling=request.outlier_handling if request.outlier_handling is not None else True
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Autonomous Agent Execution Failed: {str(e)}")

@router.post("/sample")
def load_sample_and_run(request: AutoAgentSampleRequest):
    try:
        sample_df = generate_sample_dataset(request.dataset_name)
        state.set_active_df(sample_df)
        state.reset_state(f"sample_{request.dataset_name}.csv")
        state.action_history.append(f"Loaded Sample: {request.dataset_name}")

        if request.auto_run:
            target_map = {
                "churn": "Churn",
                "housing": "SalePrice",
                "retention": "Attrition"
            }
            target_col = target_map.get(request.dataset_name, None)
            goal_text = request.goal or f"Autonomous AutoML on {request.dataset_name.capitalize()} dataset"
            result = run_autonomous_agent(
                df=sample_df,
                goal=goal_text,
                target_column=target_col,
                problem_type="auto"
            )
            return result
        else:
            return {
                "message": f"Sample dataset '{request.dataset_name}' loaded successfully",
                "summary": get_df_summary(sample_df)
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(e)}")

@router.post("/plan")
def generate_agent_plan(request: AutoAgentPlanRequest):
    df = state.get_active_df()
    if df is None or df.empty:
        return {
            "ready": False,
            "message": "No active dataset loaded.",
            "recommended_targets": [],
            "plan_steps": []
        }

    # Identify potential targets
    potential_targets = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 10 or pd.api.types.is_numeric_dtype(df[col]):
            potential_targets.append({
                "column": col,
                "type": "classification" if (n_unique <= 10 or not pd.api.types.is_numeric_dtype(df[col])) else "regression",
                "unique_values": n_unique
            })

    target_selected = request.target_column or (potential_targets[0]["column"] if potential_targets else df.columns[-1])
    target_info = next((t for t in potential_targets if t["column"] == target_selected), {
        "column": target_selected,
        "type": "classification" if df[target_selected].nunique() <= 10 else "regression"
    })

    return {
        "ready": True,
        "active_dataset": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "null_cells": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        },
        "target": target_info,
        "recommended_targets": potential_targets[:8],
        "stages": [
            {"step": 1, "name": "Dataset Ingestion & Deep Profiling", "tool": "DataProfiler"},
            {"step": 2, "name": "Self-Healing Data Quality & Cleaning", "tool": "AutoCleanEngine"},
            {"step": 3, "name": "Feature Engineering & Interactions", "tool": "FeatureSynthesizer"},
            {"step": 4, "name": "AutoML Model Tournament", "tool": "ModelArena"},
            {"step": 5, "name": "Explainability & Feature Drivers", "tool": "DriverAttributionEngine"},
            {"step": 6, "name": "Executive Synthesis & Instant Inference", "tool": "BriefSynthesizer"}
        ]
    }
