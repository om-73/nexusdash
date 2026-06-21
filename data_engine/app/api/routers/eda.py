from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from ...core import state
from ...schemas.actions import KPICalculateRequest
from ...services.utils import get_df_summary

router = APIRouter()

# Re-implement kpi_engine.py dependency
# Assuming kpi_engine.py is in data_engine root, adjust import path
import sys
import os
import traceback
# Ensure the root data_engine directory is in sys.path to find kpi_engine
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from kpi_engine import detect_kpis

@router.get("/eda")
def get_eda():
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        df = df.copy()
        numeric_df = df.select_dtypes(include=np.number)
        
        # 1. Descriptive Statistics
        description = df.describe(include='all').to_dict()
        for k, v in description.items():
            for sub_k, sub_v in v.items():
                if pd.isna(sub_v): description[k][sub_k] = None
        
        # 2. Correlation
        correlation = []
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    correlation.append({
                        "x": col1,
                        "y": col2,
                        "value": float(corr_matrix.loc[col1, col2])
                    })
            
        # 3. Distributions
        distributions = {}
        if not numeric_df.empty:
            for col in numeric_df.columns:
                try:
                    data = numeric_df[col].dropna()
                    if len(data) > 0:
                        counts, bins = np.histogram(data, bins=10)
                        distributions[col] = [{"range": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(counts[i])} for i in range(len(counts))]
                except: pass
        
        # 4. Categorical Counts
        categorical_counts = {}
        cat_df = df.select_dtypes(include=['object', 'category'])
        for col in cat_df.columns:
            try:
                counts = df[col].value_counts().head(10).to_dict()
                categorical_counts[col] = [{"name": str(k), "count": int(v)} for k, v in counts.items()]
            except: pass
            
        # 5. Scatter Data
        scatter_data = []
        if not numeric_df.empty and len(numeric_df.columns) >= 2:
            sample_size = min(500, len(numeric_df))
            sampled_df = numeric_df.sample(n=sample_size, random_state=42).replace({np.nan: None})
            
            # Explicitly convert to dict and handle numpy types
            records = sampled_df.to_dict(orient="records")
            scatter_data = []
            for row in records:
                processed_row = {}
                for k, v in row.items():
                    if hasattr(v, 'item'): processed_row[k] = v.item()
                    elif pd.isna(v): processed_row[k] = None
                    else: processed_row[k] = v
                scatter_data.append(processed_row)
 
        # 6. Box Plot Data
        box_plot_data = {}
        if not numeric_df.empty:
            for col in numeric_df.columns:
                try:
                    s = numeric_df[col].dropna()
                    if not s.empty:
                        q1, median, q3 = s.quantile(0.25), s.median(), s.quantile(0.75)
                        box_plot_data[col] = {
                            "min": float(s.min()),
                            "q1": float(q1),
                            "median": float(median),
                            "q3": float(q3),
                            "max": float(s.max())
                        }
                except: pass
        
        return {
            "description": description,
            "correlation": correlation,
            "distributions": distributions,
            "categorical_counts": categorical_counts,
            "scatter_data": scatter_data,
            "box_plot_data": box_plot_data
        }

    except Exception as e:
        error_msg = f"EDA Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eda/summary")
def get_eda_summary():
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        # Import the new AI Engine relative to this file
        # Path: data_engine/app/api/routers/eda.py -> data_engine/ai_engine.py
        # We need to adjust sys.path or use absolute import if packaged
        
        # DYNAMIC IMPORT
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
        from ai_engine import AIAnalyst
        
        analyst = AIAnalyst()
        result = analyst.analyze_dataset(df)
        
        # Map to frontend expected format
        # Frontend expects {"insights": [list of strings]}
        # Our AI engine returns {"summary", "insights", "recommendations"}
        # We can combine them or just send insights
        
        combined_insights = []
        combined_insights.append(f"📝 {result['summary']}")
        combined_insights.extend([f"💡 {i}" for i in result['insights']])
        if result.get('recommendations'):
            combined_insights.extend([f"🚀 {r}" for r in result['recommendations']])
            
        return {"insights": combined_insights}

    except Exception as e:
        print(f"AI Summary Error: {e}")
        # Fallback to simple logic if everything fails
        return {"insights": [f"Analysis Error: {str(e)}"]}

@router.get("/quality")
def get_quality():
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        missing_score = 100 - (total_missing / total_cells * 100) if total_cells > 0 else 0
        
        total_rows = len(df)
        duplicates = df.duplicated().sum()
        duplicate_score = 100 - (duplicates / total_rows * 100) if total_rows > 0 else 0
        
        total_cols = len(df.columns)
        complete_cols = len(df.columns[df.notnull().all()])
        completeness_score = (complete_cols / total_cols * 100) if total_cols > 0 else 0
        
        final_score = (missing_score * 0.4) + (duplicate_score * 0.3) + (completeness_score * 0.3)
        
        column_profile = {}
        for col in df.columns:
            column_profile[col] = {
                "missing": int(df[col].isnull().sum()),
                "missing_pct": float(df[col].isnull().mean() * 100),
                "type": str(df[col].dtype)
            }
            
        return {
            "score": round(final_score, 1),
            "metrics": {
                "missing_cells_pct": round(total_missing / total_cells * 100, 1) if total_cells > 0 else 0,
                "duplicate_rows_pct": round(duplicates / total_rows * 100, 1) if total_rows > 0 else 0,
                "complete_columns_pct": round(completeness_score, 1)
            },
            "column_profile": column_profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kpi")
def get_kpis():
    df = state.get_active_df()
    if df is None: return []
    try:
        return detect_kpis(df)
    except Exception as e:
        print(f"KPI Error: {e}")
        return []

@router.post("/kpi/calculate")
def calculate_kpi(request: KPICalculateRequest):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    col, op = request.column, request.operation
    if col not in df.columns: raise HTTPException(status_code=400, detail=f"Column {col} not found")
    
    try:
        val = 0
        if op == "sum": val = df[col].sum()
        elif op == "mean": val = df[col].mean()
        elif op == "count": val = df[col].count()
        elif op == "min": val = df[col].min()
        elif op == "max": val = df[col].max()
        elif op == "unique": val = df[col].nunique()
        
        if hasattr(val, 'item'): val = val.item()
        
        return {"value": val, "label": f"{op.capitalize()} of {col}", "type": "number"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to calculate: {str(e)}")

@router.get("/quantile")
def get_quantile(column: str, q: float = 0.95):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    if column not in df.columns: raise HTTPException(status_code=400, detail="Column not found")
    try:
        val = df[column].quantile(q)
        return {"column": column, "quantile": q, "value": float(val)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend_charts")
def recommend_charts():
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        recommendations = []
        
        # 1. Distribution of Numerical Variables
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            try:
                data = df[col].dropna()
                counts, bins = np.histogram(data, bins=10)
                chart_data = [{"range": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(counts[i])} for i in range(len(counts))]
                
                recommendations.append({
                    "id": "rec_dist",
                    "type": "distribution",
                    "title": f"Distribution of {col}",
                    "chartType": "bar",
                    "description": f"See the spread of values for {col}",
                    "x": "range",
                    "y": "count",
                    "data": chart_data
                })
            except: pass
            
        # 2. Categorical Counts
        if len(cat_cols) > 0:
            col = cat_cols[0]
            try:
                counts = df[col].value_counts().head(10).to_dict()
                chart_data = [{"name": str(k), "count": int(v)} for k, v in counts.items()]
                
                recommendations.append({
                    "id": "rec_cat",
                    "type": "categorical",
                    "title": f"Top Categories in {col}",
                    "chartType": "bar",
                    "description": f"Frequency of top categories in {col}",
                    "x": "name",
                    "y": "count",
                    "data": chart_data
                })
            except: pass
            
        # 3. Correlation
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            try:
                sample = df[[col1, col2]].sample(n=min(200, len(df)), random_state=42).replace({np.nan: None})
                chart_data = sample.to_dict(orient="records")
                
                recommendations.append({
                    "id": "rec_corr",
                    "type": "correlation",
                    "title": f"{col1} vs {col2}",
                    "chartType": "scatter",
                    "description": f"Correlation relationship between {col1} and {col2}",
                    "x": col1,
                    "y": col2,
                    "data": chart_data
                })
            except: pass
            
        # 4. Comparison
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            cat_col = cat_cols[0]
            num_col = numeric_cols[0]
            try:
                grouped = df.groupby(cat_col)[num_col].mean().head(10).sort_values(ascending=False)
                chart_data = [{"name": str(k), "value": float(v)} for k, v in grouped.items()]
                
                recommendations.append({
                    "id": "rec_comp",
                    "type": "comparison",
                    "title": f"Average {num_col} by {cat_col}",
                    "chartType": "bar",
                    "description": f"Compare average {num_col} across top {cat_col} groups",
                    "x": "name",
                    "y": "value",
                    "data": chart_data
                })
            except: pass

        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
