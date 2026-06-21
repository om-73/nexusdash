import pandas as pd
import os
from datetime import datetime
from ..core.config import SNAPSHOTS_DIR

def get_df_summary(df: pd.DataFrame):
    # OPTIMIZATION: Only clean/replace NaNs for the preview (first 50 rows)
    
    # 1. Get Preview Data (Head only)
    preview_df = df.head(50).copy()
    
    # 2. Replace NaNs only in preview for JSON serialization
    preview_df = preview_df.astype(object)
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    
    # 3. Optimize Missing Value Calculation
    if len(df) > 300000:
        sample_size = min(100000, len(df))
        sample = df.sample(n=sample_size, random_state=42)
        # Convert to standard Python int for JSON serialization
        missing_counts = {k: int(v) for k, v in (sample.isnull().mean() * len(df)).items()}
    else:
        # Convert to standard Python int for JSON serialization
        missing_counts = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}

    # Final cleanup of preview to ensure EVERYTHING is serializable
    # Some objects might still be numpy types even with .astype(object)
    preview_data = preview_df.to_dict(orient="records")
    def serialize_val(v):
        if hasattr(v, 'item'): return v.item()
        if pd.isna(v): return None
        return v
    
    preview_data = [{k: serialize_val(v) for k, v in row.items()} for row in preview_data]

    return {
        "message": "Success",
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "preview": preview_data,
        "missing_values": missing_counts
    }

def save_snapshot(df: pd.DataFrame, dataset_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{dataset_id}_{timestamp}.parquet"
    path = os.path.join(SNAPSHOTS_DIR, filename)
    df.to_parquet(path, index=False)
    return path
