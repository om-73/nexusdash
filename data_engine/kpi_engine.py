import pandas as pd
import numpy as np

def detect_kpis(df: pd.DataFrame):
    """
    Heuristic-based KPI detection.
    Returns a list of dicts: { "label": str, "value": any, "type": str, "trend": float|None }
    """
    kpis = []
    
    # 1. SUMMATION METRICS (Money, Counts)
    # Keywords indicating things we usually sum up
    sum_keywords = ['sales', 'revenue', 'profit', 'amount', 'cost', 'price', 'total']
    
    # 2. AVERAGE METRICS (Ratings, Scores)
    avg_keywords = ['rating', 'score', 'satisfaction', 'percent', 'rate', 'density']

    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Track used columns to avoid duplicate KPIs if we have complex logic later
    used_cols = set()

    for col in numeric_cols:
        col_lower = col.lower()
        
        # Check for Summation
        if any(k in col_lower for k in sum_keywords):
            if col not in used_cols:
                total_val = df[col].sum()
                kpis.append({
                    "id": f"sum_{col}",
                    "label": f"Total {col}",
                    "value": total_val,
                    "format": "currency" if any(c in col_lower for c in ['price', 'cost', 'revenue', 'sales', 'profit']) else "number"
                })
                used_cols.add(col)
                
        # Check for Average
        elif any(k in col_lower for k in avg_keywords):
            if col not in used_cols:
                avg_val = df[col].mean()
                kpis.append({
                    "id": f"avg_{col}",
                    "label": f"Avg {col}",
                    "value": avg_val,
                    "format": "percent" if 'percent' in col_lower else "number"
                })
                used_cols.add(col)

    # 3. CATEGORICAL TOP COUNTS (if we have few KPIs)
    # If we didn't find many numeric KPIs, look for top categories
    if len(kpis) < 4:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df[col].nunique() < 50: # Only low cardinality
                top_val = df[col].mode()
                if not top_val.empty:
                    top_name = top_val[0]
                    count = df[df[col] == top_name].shape[0]
                    kpis.append({
                        "id": f"top_{col}",
                        "label": f"Top {col}",
                        "value": f"{top_name} ({count})",
                        "format": "text"
                    })
                    if len(kpis) >= 4: break

    return kpis[:6] # Return top 6 detected
