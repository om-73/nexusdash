from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import pandas as pd
import numpy as np
import os
import uvicorn
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

# ML Imports
# ML Imports (Lazy loaded in functions to save memory)
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
# ... (Moved to train_model)

# NLP / Query
# NLP / Query
import re
import hashlib
from kpi_engine import detect_kpis
from ai_engine import suggest_model_config
from datetime import datetime
import sys
import contextlib
import joblib
from fastapi.responses import FileResponse

# Masks

# Masks
import hashlib

# AI
import openai


# Masks
import hashlib


# Connectors
import requests
import json
try:
    import snowflake.connector
except ImportError:
    snowflake = None
    


try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
except ImportError:
    bigquery = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Metadata Store
from metadata_store import MetadataStore
metadata_store = MetadataStore()

SNAPSHOTS_DIR = "snapshots"
if not os.path.exists(SNAPSHOTS_DIR):
    os.makedirs(SNAPSHOTS_DIR)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/model/download")
def download_model():
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No trained model found")
    return FileResponse(model_path, media_type='application/octet-stream', filename="model.pkl")

# State
active_df = None
active_file_path = None
history_stack = [] # Stack of DataFrames
redo_stack = [] # Stack of DataFrames
action_history = [] # List of strings describing actions
redo_action_history = [] # Stack of redoable actions

# --- Models ---
class FileLoadRequest(BaseModel):
    file_path: str
    file_type: str = "csv"

class CleanRequest(BaseModel):
    operation: str
    columns: Optional[list] = None
    value: Optional[str] = None
    strategy: Optional[str] = None
    rename_map: Optional[dict] = None

class TrainRequest(BaseModel):
    problem_type: str # regression, classification, clustering
    algorithm: str # linear, logistic, dt, rf, kmeans
    target_column: Optional[str] = None
    feature_columns: List[str]
    params: Optional[Dict[str, Any]] = {}
    params: Optional[Dict[str, Any]] = {}
    test_size: float = 0.2

class DatabaseConnectRequest(BaseModel):
    db_type: str # postgresql, mongodb, snowflake, bigquery, redshift, api
    connection_string: Optional[str] = None
    query: Optional[str] = None # For SQL
    collection: Optional[str] = None # For MongoDB
    limit: int = 1000
    # Snowflake / Redshift specific
    account: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema_name: Optional[str] = None
    role: Optional[str] = None
    # BigQuery specific
    project_id: Optional[str] = None
    credentials_json: Optional[str] = None # Stringified JSON
    # API specific
    api_url: Optional[str] = None
    method: Optional[str] = "GET"
    headers: Optional[Dict[str, str]] = {}
    json_body: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str

class AIConfigRequest(BaseModel):
    goal: str

class RegisterDatasetRequest(BaseModel):
    name: str # e.g. "Monthly Sales"
    source_type: str # csv, snowflake...

class ContractRequest(BaseModel):
    dataset_id: str
    contract: Dict[str, Any]

class ValidateContractRequest(BaseModel):
    dataset_id: str

class MaskRequest(BaseModel):
    columns: List[str]
    strategy: str # redact, hash, partial

class QueryBuilderRequest(BaseModel):
    select: Optional[List[str]] = None
    filters: Optional[List[Dict[str, Any]]] = None # {col, op, val}
    groupby: Optional[List[str]] = None
    aggregates: Optional[Dict[str, str]] = None # {col: 'sum'}
    sort: Optional[Dict[str, str]] = None # {col: 'asc'}
    limit: int = 1000

class FeatureRegisterRequest(BaseModel):
    name: str
    description: str
    version: str
    logic_code: str # Snippet

class NotebookRequest(BaseModel):
    code: str

class DriversRequest(BaseModel):
    target_column: str

class DashboardRequest(BaseModel):
    id: Optional[str] = None
    name: str
    layout: List[Dict[str, Any]]

# --- PII Helper ---

# --- PII Helper ---
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "phone": r'\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
}


# --- Helpers ---
def get_df_summary(df):
    # OPTIMIZATION: Only clean/replace NaNs for the preview (first 50 rows)
    # df.where returns a full copy which is slow for large datasets
    
    # 1. Get Preview Data (Head only)
    preview_df = df.head(50).copy()
    
    # 2. Replace NaNs only in preview for JSON serialization
    # Must cast to object first to ensure None is accepted and not back-converted to NaN
    preview_df = preview_df.astype(object)
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    
    # 3. Optimize Missing Value Calculation
    # For large datasets (>300k rows), estimate missing values from a sample to speed up response
    if len(df) > 300000:
        # Sample ~100k rows or 20% whichever is smaller but at least 50k
        sample_size = min(100000, len(df))
        sample = df.sample(n=sample_size, random_state=42)
        # Calculate ratio and scale back to total
        missing_counts = (sample.isnull().mean() * len(df)).astype(int).to_dict()
    else:
        missing_counts = df.isnull().sum().to_dict()

    return {
        "message": "Success",
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "preview": preview_df.to_dict(orient="records"),
        "missing_values": missing_counts
    }

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Data Engine is running"}


def save_snapshot(df: pd.DataFrame, dataset_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{dataset_id}_{timestamp}.parquet"
    path = os.path.join(SNAPSHOTS_DIR, filename)
    df.to_parquet(path, index=False)
    return path

# State persistence
@app.get("/state")
def get_state():
    global active_df
    if active_df is None:
        return None # Return None/null if no data
    try:
        return get_df_summary(active_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load Data endpoint
@app.post("/load")
def load_data(request: FileLoadRequest):
    global active_df, active_file_path
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        if request.file_type == "csv":
            # Auto-detect delimiter using csv.Sniffer (Faster than pandas python engine)
            try:
                import csv
                with open(request.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(1024)
                    dialect = csv.Sniffer().sniff(sample)
                    sep = dialect.delimiter
                
                # Use C-engine (default) with detected separator
                df = pd.read_csv(request.file_path, sep=sep)
            except Exception as e:
                print(f"Sniffer failed, falling back to default: {e}")
                df = pd.read_csv(request.file_path) # Default pandas inference
        elif request.file_type in ["xlsx", "excel"]:
            df = pd.read_excel(request.file_path)
        else:
             raise HTTPException(status_code=400, detail="Unsupported file type")
        
        active_df = df
        active_file_path = request.file_path
        history_stack.clear() # Reset stacks on new load
        redo_stack.clear()
        action_history = ["Loaded Data"]
        redo_action_history.clear()
        
        # Auto-register temporary ID for snapshotting if not registered yet?
        # Actually register_dataset is usually called AFTER load by frontend.
        # But we can assume a session-based ID or just wait for registration.
        # For now, let's create a temp ID based on filepath hash so we can track history immediately.
        temp_id = hashlib.md5(request.file_path.encode()).hexdigest()
        
        # Save initial snapshot
        snap_path = save_snapshot(df, temp_id)
        # Log run (we might not have a Formal Name yet, but we have an ID)
        # MetadataStore will require 'datasets' entry for FK constraint?
        # Yes, so let's auto-register a "Draft" dataset
        metadata_store.register_dataset(temp_id, os.path.basename(request.file_path), "upload")
        metadata_store.log_run(temp_id, df, snap_path)
        
        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean")
def clean_data(request: CleanRequest):
    global active_df
    if active_df is None:
        print("WARNING: clean_data called but active_df is None. Session likely reset.")
        raise HTTPException(status_code=400, detail="Session reset. Please reload your dataset.")
    try:
        # PUSH HISTORY BEFORE CHANGE
        history_stack.append(active_df.copy())
        redo_stack.clear() # Clear redo on new action
        redo_action_history.clear()

        action_desc = f"Performed {request.operation}"
        
        df = active_df.copy()
        if request.operation == "dropna":
            if request.columns: df.dropna(subset=request.columns, inplace=True)
            else: df.dropna(inplace=True)
        elif request.operation == "fillna":
            if request.value is not None: df.fillna(request.value, inplace=True)
            elif request.strategy:
                numeric_cols = df.select_dtypes(include=np.number).columns
                if request.strategy == "mean": df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif request.strategy == "median": df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif request.strategy == "mode":
                     for col in df.columns: 
                        m = df[col].mode()
                        if not m.empty:
                            df[col] = df[col].fillna(m[0])
        elif request.operation == "drop_duplicates": df.drop_duplicates(inplace=True)
        elif request.operation == "drop_columns": 
            if request.columns: df.drop(columns=request.columns, inplace=True)
        elif request.operation == "rename_columns":
            if request.rename_map: df.rename(columns=request.rename_map, inplace=True)
            
        elif request.operation == "remove_outliers":
            # Simple IQR based outlier removal for numeric columns
            if request.columns:
                for col in request.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            
        elif request.operation == "encode_columns":
            if request.columns:
                for col in request.columns:
            elif request.strategy == "label":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                    elif request.strategy == "onehot":
                        df = pd.get_dummies(df, columns=[col], prefix=col)

        elif request.operation == "normalize":
            if request.columns:
                from sklearn.preprocessing import MinMaxScaler, StandardScaler
                scaler = None
                if request.strategy == "minmax": scaler = MinMaxScaler()
                elif request.strategy == "standard": scaler = StandardScaler()
                
                if scaler:
                    df[request.columns] = scaler.fit_transform(df[request.columns])
        
        active_df = df
        
        # Append detailed description based on operation
        if request.operation == "dropna": action_desc = "Dropped Missing Values"
        elif request.operation == "fillna": action_desc = f"Filled Missing Values ({request.value or request.strategy})"
        elif request.operation == "drop_duplicates": action_desc = "Dropped Duplicates"
        elif request.operation == "drop_columns": action_desc = f"Dropped Columns: {request.columns}"
        elif request.operation == "remove_outliers": action_desc = "Removed Outliers"
        elif request.operation == "encode_columns": action_desc = f"Encoded Columns ({request.strategy})"
        elif request.operation == "normalize": action_desc = f"Normalized Columns ({request.strategy})"
        
        action_history.append(action_desc)
        
        # Snapshotting
        if active_file_path:
            dataset_id = hashlib.md5(active_file_path.encode()).hexdigest()
            snap_path = save_snapshot(df, dataset_id)
            metadata_store.log_run(dataset_id, df, snap_path)
        
        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kpi")
def get_kpis():
    global active_df
    if active_df is None: return [] # Return empty if no data
    try:
        return detect_kpis(active_df)
    except Exception as e:
        print(f"KPI Extract Error: {e}")
        return []

@app.get("/eda")
def get_eda():
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        df = active_df.copy()
        numeric_df = df.select_dtypes(include=np.number)
        
        # 1. Descriptive Statistics
        description = df.describe(include='all').to_dict()
        for k, v in description.items():
            for sub_k, sub_v in v.items():
                if pd.isna(sub_v): description[k][sub_k] = None
        
        # 2. Correlation Matrix (Heatmap)
        correlation = {}
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            # Format for Recharts/Heatmap: list of objects { x: col1, y: col2, value: 0.8 }
            correlation_data = []
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    correlation_data.append({
                        "x": col1,
                        "y": col2,
                        "value": float(corr_matrix.loc[col1, col2])
                    })
            correlation = correlation_data
            
        # 3. Distributions (Histograms)
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
            
        # 5. Scatter Plot Data (Sampled)
        # We perform downsampling to avoid sending massive JSON
        scatter_data = []
        if not numeric_df.empty and len(numeric_df.columns) >= 2:
            sample_size = min(500, len(numeric_df))
            sampled_df = numeric_df.sample(n=sample_size, random_state=42).replace({np.nan: None})
            scatter_data = sampled_df.to_dict(orient="records")

        # 6. Box Plot Data (For Outlier Visualization)
        box_plot_data = {}
        if not numeric_df.empty:
            for col in numeric_df.columns:
                try:
                    s = numeric_df[col].dropna()
                    if not s.empty:
                        q1 = s.quantile(0.25)
                        median = s.median()
                        q3 = s.quantile(0.75)
                        iqr = q3 - q1
                        min_val = s.min()
                        max_val = s.max()
                        # Outliers for box plot (points outside whiskers)
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = s[(s < lower_bound) | (s > upper_bound)].tolist()
                        
                        q3 = s.quantile(0.75)
                        iqr = q3 - q1
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
        print(f"EDA Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/drivers")
def analyze_drivers(request: DriversRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    target = request.target_column
    if target not in active_df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found")
        
    try:
        df = active_df.copy()
        
        # 1. Preprocessing
        # Drop rows with missing target
        df = df.dropna(subset=[target])
        
        # Separate Features and Target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Handle Missing Values in Features (Simple Imputation)
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                mode_val = X[col].mode()
                if not mode_val.empty:
                    X[col] = X[col].fillna(mode_val[0])
                else:
                    X[col] = X[col].fillna("Unknown")

        # Encode Categorical Features
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                X[col] = pd.factorize(X[col])[0]
                
        # Determine Problem Type (Regression or Classification)
        is_regression = False
        if pd.api.types.is_numeric_dtype(y):
            if pd.api.types.is_float_dtype(y):
                is_regression = True
            elif len(y.unique()) > 20:
                is_regression = True
                
        # Lazy Import for analyzing drivers
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        # Fit Model
        importances = []
        if is_regression:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)
        else:
            if not pd.api.types.is_numeric_dtype(y):
                y = pd.factorize(y)[0]
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
        # Extract Importances
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            feature_names = X.columns
            
            data = [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, imps)]
            data.sort(key=lambda x: x['importance'], reverse=True)
            importances = data
            
        return {
            "target": target,
            "problem_type": "Regression" if is_regression else "Classification",
            "drivers": importances
        }
            
    except Exception as e:
        print(f"Driver Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model(request: TrainRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        df = active_df.copy()
        
        # Lazy Imports for Training
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
        from sklearn.svm import SVR, SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.cluster import KMeans
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

        # Preprocessing: Drop NaNs in selected columns
        # Ensure we don't have duplicate columns in the dataframe itself
        df = df.loc[:, ~df.columns.duplicated()]
        
        cols_to_use = list(set(request.feature_columns + ([request.target_column] if request.target_column else [])))
        df = df[cols_to_use].dropna()
        
        if df.empty: raise ValueError("Resulting dataset is empty after dropping NaNs")

        # Encode categorical features
        # Simple approach: Label Encode everything object-like
        le_dict = {}
        for col in request.feature_columns:
            if col in df.columns and df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le # Store if needed for inference (omitted for MVP)

        X = df[request.feature_columns]
        
        if request.problem_type == "clustering":
            if request.algorithm == "kmeans":
                n_clusters = int(request.params.get("n_clusters", 3))
                model = KMeans(n_clusters=n_clusters)
                model.fit(X)
                labels = model.labels_
                # Metrics (Silhouette is expensive, just return cluster counts)
                unique, counts = np.unique(labels, return_counts=True)
                return {
                    "metrics": {"inertia": float(model.inertia_)},
                    "clusters": dict(zip([str(u) for u in unique], [int(c) for c in counts]))
                }
        
        # For Regression/Classification
        y = df[request.target_column]
        if y.dtype == 'object': # Encode Target if categorical
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=42)
        
        # Helper to train a single model
        def train_single(algo, p_type):
            m = None
            if p_type == "regression":
                if algo == "linear": m = LinearRegression()
                elif algo == "ridge": m = Ridge()
                elif algo == "lasso": m = Lasso()
                elif algo == "dt": m = DecisionTreeRegressor()
                elif algo == "rf": m = RandomForestRegressor()
                elif algo == "svr": m = SVR()
                elif algo == "gbr": m = GradientBoostingRegressor()
                elif algo == "ada": m = AdaBoostRegressor()
            elif p_type == "classification":
                if algo == "logistic": m = LogisticRegression(max_iter=1000)
                elif algo == "dt": m = DecisionTreeClassifier()
                elif algo == "rf": m = RandomForestClassifier()
                elif algo == "knn": m = KNeighborsClassifier()
                elif algo == "nb": m = GaussianNB()
                elif algo == "svm": m = SVC(probability=True)
                elif algo == "gbc": m = GradientBoostingClassifier()
                elif algo == "ada": m = AdaBoostClassifier()
            
            if m:
                m.fit(X_train, y_train)
                preds = m.predict(X_test)
                mets = {}
                if p_type == "regression":
                    mets = {
                        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                        "r2": float(r2_score(y_test, preds))
                    }
                else:
                    mets = {
                        "accuracy": float(accuracy_score(y_test, preds)),
                        "f1_weighted": float(f1_score(y_test, preds, average='weighted'))
                    }
                return m, preds, mets

        if request.algorithm == "all":
            results = []
            if request.problem_type == "regression":
                algos = ["linear", "ridge", "lasso", "dt", "rf", "svr", "gbr", "ada"]
            else:
                algos = ["logistic", "knn", "nb", "dt", "rf", "svm", "gbc", "ada"]
            
            best_score = -float('inf')
            best_algo = None
            best_model_data = None
            
            for algo in algos:
                try:
                    m, preds, mets = train_single(algo, request.problem_type)
                    
                    score = mets.get("r2") if request.problem_type == "regression" else mets.get("accuracy")
                    if score > best_score:
                        best_score = score
                        best_algo = algo
                        best_model_data = (m, preds, mets)
                        
                    results.append({
                        "algorithm": algo,
                        "metrics": mets
                    })
                except Exception as e:
                    print(f"Failed {algo}: {e}")
            
            # Use best model for feature importance details
            if best_model_data:
                model, y_pred, metrics = best_model_data
                
                # Save Best Model
                if not os.path.exists("models"): os.makedirs("models")
                joblib.dump(model, "models/model.pkl")
                
                return {
                    "metrics": metrics,
                    "predictions_preview": y_pred[:10].tolist(),
                    "actual_preview": y_test[:10].tolist(),
                    "comparison": results,
                    "best_algorithm": best_algo,
                    "download_available": True
                }

            else:
                raise HTTPException(status_code=500, detail="All models failed to train.")

        # Single Model Path
        model, y_pred, metrics = train_single(request.algorithm, request.problem_type)
            
        # Feature Importance (Tree based)
        importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, col in enumerate(request.feature_columns):
                importance[col] = float(model.feature_importances_[i])
        elif hasattr(model, 'coef_'):
             # Linear cases (coef_ might be 2d for multiclass)
             if hasattr(model.coef_, 'shape') and len(model.coef_.shape) > 1:
                  # Take mean absolute coef for multiclass
                  coefs = np.mean(np.abs(model.coef_), axis=0)
             else:
                  coefs = np.abs(model.coef_)
             for i, col in enumerate(request.feature_columns):
                 if i < len(coefs):
                    importance[col] = float(coefs[i])

        return {
            "metrics": metrics,
            "feature_importance": importance,
            "predictions_preview": y_pred[:10].tolist(),
            "actual_preview": y_test[:10].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
def export_data():
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        stream = io.StringIO()
        active_df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=cleaned_data.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/connect_db")
def connect_db(request: DatabaseConnectRequest):
    global active_df, active_file_path
    try:
        df = pd.DataFrame()
        source_name = request.db_type

        # 1. PostgreSQL / Redshift
        if request.db_type in ["postgresql", "redshift"]:
            if not request.query:
                raise HTTPException(status_code=400, detail="Query is required for SQL databases")
            
            # Simple Read-only check
            forbidden = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE ", "GRANT ", "REVOKE "]
            if any(cmd in request.query.upper() for cmd in forbidden):
                 raise HTTPException(status_code=400, detail="Only SELECT queries are allowed (Read-only mode)")

            # If using individual fields for Redshift/Postgres could be added here, 
            # but usually connection_string is standard.
            if not request.connection_string:
                 raise HTTPException(status_code=400, detail="Connection string is required")
            
            engine = create_engine(request.connection_string)
            with engine.connect() as connection:
                df = pd.read_sql(request.query, connection)

        # 2. MongoDB
        elif request.db_type == "mongodb":
            if not request.collection:
                 raise HTTPException(status_code=400, detail="Collection is required for MongoDB")
            if not request.connection_string:
                 raise HTTPException(status_code=400, detail="Connection string is required")
                 
            client = MongoClient(request.connection_string)
            # Parse DB name from string or default? 
            # PyMongo client.get_database() uses the one in URI if present.
            db = client.get_database() 
            collection = db[request.collection]
            cursor = collection.find().limit(request.limit)
            df = pd.DataFrame(list(cursor))
            if '_id' in df.columns: df.drop(columns=['_id'], inplace=True)

        # 3. Snowflake
        elif request.db_type == "snowflake":
            if not snowflake:
                raise HTTPException(status_code=500, detail="Snowflake driver not installed")
            if not request.query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            # Simple Read-only check
            forbidden = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE ", "GRANT ", "REVOKE "]
            if any(cmd in request.query.upper() for cmd in forbidden):
                 raise HTTPException(status_code=400, detail="Only SELECT queries are allowed (Read-only mode)")
            
            ctx = snowflake.connector.connect(
                user=request.user,
                password=request.password,
                account=request.account,
                warehouse=request.warehouse,
                database=request.database,
                schema=request.schema_name,
                role=request.role
            )
            cur = ctx.cursor()
            try:
                cur.execute(request.query)
                df = cur.fetch_pandas_all()
            finally:
                cur.close()
                ctx.close()

        # 4. Google BigQuery
        elif request.db_type == "bigquery":
            if not bigquery:
                raise HTTPException(status_code=500, detail="BigQuery driver not installed")
            if not request.query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            # Simple Read-only check
            forbidden = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE ", "GRANT ", "REVOKE "]
            if any(cmd in request.query.upper() for cmd in forbidden):
                 raise HTTPException(status_code=400, detail="Only SELECT queries are allowed (Read-only mode)")

            if not request.credentials_json:
                raise HTTPException(status_code=400, detail="Service Account JSON is required")

            # Parse credentials
            try:
                service_account_info = json.loads(request.credentials_json)
                credentials = service_account.Credentials.from_service_account_info(service_account_info)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid Credentials JSON: {str(e)}")

            client = bigquery.Client(credentials=credentials, project=request.project_id)
            df = client.query(request.query).to_dataframe()

        # 5. REST API
        elif request.db_type == "api":
            if not request.api_url:
                raise HTTPException(status_code=400, detail="API URL is required")
            
            method = request.method.upper() if request.method else "GET"
            headers = request.headers or {}
            
            if method == "GET":
                response = requests.get(request.api_url, headers=headers)
            elif method == "POST":
                response = requests.post(request.api_url, headers=headers, json=request.json_body)
            else:
                raise HTTPException(status_code=400, detail="Unsupported HTTP method")

            if response.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"API Error {response.status_code}: {response.text}")

            try:
                data = response.json()
                # Attempt to find list in response
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Heuristic: find the first key that is a list of objects
                    found_list = False
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            df = pd.DataFrame(value)
                            found_list = True
                            break
                    if not found_list:
                        # Just normalize single dict
                        df = pd.json_normalize(data)
                else:
                    raise ValueError("API response is not JSON object or list")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse API response: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Unsupported database type")

        if df.empty:
             raise HTTPException(status_code=400, detail="Query returned no data")

        active_df = df
        active_file_path = f"{request.db_type}://connection"
        history_stack.clear()
        redo_stack.clear()
        action_history = [f"Connected to {source_name}"]
        redo_action_history.clear()
        return get_df_summary(df)
        
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/pipeline")
def get_pipeline():
    global action_history
    return {"steps": action_history}
    
@app.post("/undo")
def undo_action():
    global active_df, history_stack, redo_stack, action_history, redo_action_history
    if not history_stack:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    try:
        # Move current to redo
        redo_stack.append(active_df)
        if action_history: redo_action_history.append(action_history.pop())
        
        # Pop from history to active
        active_df = history_stack.pop()
        return get_df_summary(active_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/redo")
def redo_action():
    global active_df, history_stack, redo_stack, action_history, redo_action_history
    if not redo_stack:
        raise HTTPException(status_code=400, detail="Nothing to redo")
    try:
        # Move active to history
        history_stack.append(active_df)
        if redo_action_history: action_history.append(redo_action_history.pop())
        
        # Pop from redo to active
        active_df = redo_stack.pop()
        return get_df_summary(active_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quality")
def get_quality():
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        df = active_df.copy()
        
        # 1. Missing Values Score (Weighted 40%)
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        missing_score = 100 - (total_missing / total_cells * 100) if total_cells > 0 else 0
        
        # 2. Duplicate Rows Score (Weighted 30%)
        total_rows = len(df)
        duplicates = df.duplicated().sum()
        duplicate_score = 100 - (duplicates / total_rows * 100) if total_rows > 0 else 0
        
        # 3. Completeness (Columns with no missing values) (Weighted 30%)
        total_cols = len(df.columns)
        complete_cols = len(df.columns[df.notnull().all()])
        completeness_score = (complete_cols / total_cols * 100) if total_cols > 0 else 0
        
        # Detailed Column Profile
        column_profile = {}
        for col in df.columns:
            column_profile[col] = {
                "missing": int(df[col].isnull().sum()),
                "missing_pct": float(df[col].isnull().mean() * 100),
                "unique": int(df[col].nunique()),
                "type": str(df[col].dtype)
            }

        final_score = (missing_score * 0.4) + (duplicate_score * 0.3) + (completeness_score * 0.3)
        
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

@app.get("/eda/summary")
def get_eda_summary():
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        df = active_df.copy()
        insights = []
        
        # 1. Shape
        insights.append(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # 2. Missing Values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if not missing_cols.empty:
            insights.append(f"Found missing values in {len(missing_cols)} columns: " + ", ".join(missing_cols.index.tolist()) + ".")
        else:
            insights.append("No missing values found in the dataset.")
            
        # 3. Duplicates
        dups = df.duplicated().sum()
        if dups > 0:
            insights.append(f"Identified {dups} duplicate rows.")
        
        # 4. Data Types
        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        insights.append(f"Columns breakdown: {len(numeric_cols)} numerical, {len(cat_cols)} categorical.")
        
        # 5. Outliers (Simple Z-score check for first few numeric cols)
        outlier_notes = []
        for col in numeric_cols[:5]: # Check first 5 to save time
             if df[col].nunique() <= 1: continue
             if df[col].std() == 0: continue
             z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
             outliers = np.sum(z_scores > 3)
             if outliers > 0:
                 outlier_notes.append(f"{col} ({outliers})")
        if outlier_notes:
            insights.append(f"Potential outliers detected in: {', '.join(outlier_notes)}.")
            
        # 6. Correlations
        if len(numeric_cols) > 1:
            corr_mat = df[numeric_cols].corr().abs()
            # Select upper triangle
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
            if high_corr:
                insights.append(f"High correlation detected in columns: {', '.join(high_corr)}.")

        return {"insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend_charts")
def recommend_charts():
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        df = active_df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        recommendations = []
        
        # 1. Distribution of Numerical Variables (Histogram/Bar)
        # Suggest the first numeric column
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
            
        # 2. Categorical Counts (Bar)
        # Suggest the first categorical column
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
            
        # 3. Correlation (Scatter)
        # Suggest top 2 numeric columns
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
            
        # 4. Comparison (Bar with Group)
        # Average of Num Col by Cat Col
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            cat_col = cat_cols[0]
            num_col = numeric_cols[0]
            try:
                # Group by cat, mean of num
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

@app.post("/query_ml")
def query_ml(request: QueryRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        query = request.query.lower()
        df = active_df.copy()
        
        # Heuristic Parser
        clean_query = re.sub(r'show\s+|rows\s+|where\s+|find\s+|data\s+|records\s+', '', query, flags=re.IGNORECASE).strip()
        
        col_map = {c.lower(): c for c in df.columns}
        
        processed_query = clean_query
        for l_col, a_col in col_map.items():
            if l_col in processed_query:
                processed_query = processed_query.replace(l_col, f"`{a_col}`")
        
        processed_query = processed_query.replace(" equals ", " == ").replace(" equal ", " == ").replace("=", "==")
        processed_query = processed_query.replace("===", "==")
        
        try:
            result_df = df.query(processed_query)
            return get_df_summary(result_df)
        except Exception as q_err:
            # Fallback for simple keyword search
            mask = np.column_stack([df[col].astype(str).str.contains(clean_query, case=False, na=False) for col in df.columns])
            result_df = df.loc[mask.any(axis=1)]
            return get_df_summary(result_df)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not understand query: " + str(e))

# --- Observability Endpoints ---

@app.post("/register_dataset")
def register_dataset(request: RegisterDatasetRequest):
    global active_df
    # Generate ID based on name or use UUID (simple hash for now)
    dataset_id = hashlib.md5(request.name.encode()).hexdigest()
    metadata_store.register_dataset(dataset_id, request.name, request.source_type)
    
    if active_df is not None:
        metadata_store.log_run(dataset_id, active_df)
        
    return {"message": "Dataset registered", "dataset_id": dataset_id}

@app.get("/check_health")
def check_health(dataset_id: str):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        df = active_df
        history = metadata_store.get_run_history(dataset_id)
        
        # 1. Volume Anomaly (Z-Score or Moving Average)
        volume_status = "stable"
        current_rows = len(df)
        if len(history) > 1: # history[0] is THIS run if logged, so look at history[1:] if we just logged it? 
            # Actually get_run_history returns all. Let's assume user calls this AFTER loading.
            # But the 'log_run' happens at registration or specific checkpoints.
            # Let's verify history for Previous runs.
            prev_counts = [row[0] for row in history if row[0] != current_rows] # Filter current if present
            if prev_counts:
                avg = np.mean(prev_counts)
                std = np.std(prev_counts)
                if std > 0 and abs(current_rows - avg) > (3 * std):
                    volume_status = "anomaly"
                elif std == 0 and abs(current_rows - avg) > 0:
                     # Exact match expected but diff found
                     volume_status = "changed" # e.g. 100 vs 1000
        
        # 2. Schema Drift
        schema_status = "stable"
        if len(history) > 0:
            # Check latest history entry that isn't potentially current (logic simplified)
            last_schema_hash = history[0][2]
            # Re-calc current hash
            current_schema = str(sorted([(c, str(t)) for c, t in df.dtypes.items()]))
            current_hash = hashlib.md5(current_schema.encode()).hexdigest()
            
            if last_schema_hash != current_hash and len(history) > 1:
                 # Logic bit fuzzy if we just logged it. Assuming check_health pulls history including current? 
                 # Let's compare vs history[1] if history[0] is recent?
                 # Simplified: Compare current DF calc vs History[0]. If same, good. 
                 pass # Actually if we haven't logged current run yet, history[0] is PREVIOUS run.
            
            # Use 'log_run' carefully. For now, assume check_health compares Active DF to History[0]
            if history[0][2] != current_hash:
                schema_status = "drift_detected"

        # 3. Freshness (Check for date columns)
        freshness_score = "unknown"
        days_since_last = None
        
        # Heuristic: Find first column with 'date' or 'time' in name or datetime type
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
        
        if date_col:
            try:
                # Convert if needed
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                     s = pd.to_datetime(df[date_col], errors='coerce')
                else:
                     s = df[date_col]
                
                last_date = s.max()
                if pd.notnull(last_date):
                    delta = datetime.now() - last_date
                    days_since_last = delta.days
                    if days_since_last < 1: freshness_score = "fresh"
                    elif days_since_last < 7: freshness_score = "ok"
                    else: freshness_score = "stale"
            except: pass

        return {
            "volume_status": volume_status,
            "current_rows": current_rows,
            "schema_status": schema_status,
            "freshness": freshness_score,
            "days_since_last_entry": days_since_last
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contracts")
def save_contract(request: ContractRequest):
    try:
        metadata_store.save_contract(request.dataset_id, request.contract)
        return {"message": "Contract saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_contract")
def validate_contract(request: ValidateContractRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    contract = metadata_store.get_contract(request.dataset_id)
    if not contract: raise HTTPException(status_code=404, detail="No contract found")
    
    df = active_df
    violations = []
    
    # Check Columns
    expected_cols = contract.get("columns", {})
    for col_name, rules in expected_cols.items():
        if col_name not in df.columns:
            violations.append(f"Missing column: {col_name}")
            continue
            
        # Type Check (Simple)
        if "type" in rules:
            target_type = rules["type"]
            actual_type = str(df[col_name].dtype)
            # Rough mapping
            if target_type == "int" and "int" not in actual_type:
                 violations.append(f"Column {col_name} expected int, got {actual_type}")
            elif target_type == "string" and "object" not in actual_type and "string" not in actual_type:
                 violations.append(f"Column {col_name} expected string, got {actual_type}")

        # Constraints
        if rules.get("unique") == True:
            if not df[col_name].is_unique:
                violations.append(f"Column {col_name} contains duplicates")
        
        if rules.get("nullable") == False:
            if df[col_name].isnull().any():
                violations.append(f"Column {col_name} contains nulls")

    return {
        "status": "passed" if not violations else "failed",
        "violations": violations
    }

# --- Compliance Endpoints ---

@app.get("/scan_pii")
def scan_for_pii():
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    df = active_df
    pii_report = {}
    
    # Only scan object (string) columns
    string_cols = df.select_dtypes(include=['object']).columns
    
    # Sample for performance (first 1000 rows)
    sample_df = df[string_cols].head(1000)
    
    for col in string_cols:
        detected_types = set()
        # Concatenate column content to speed up search (simple large string check)
        # or iterate patterns against series
        series_str = sample_df[col].astype(str).str.cat(sep=" ")
        
        for pii_type, pattern in PII_PATTERNS.items():
            if re.search(pattern, series_str):
                detected_types.add(pii_type)
        
        if detected_types:
            pii_report[col] = list(detected_types)
            
    return {"pii_detected": pii_report}

@app.post("/mask_data")
def mask_data(request: MaskRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        # PUSH HISTORY BEFORE CHANGE
        history_stack.append(active_df.copy())
        redo_stack.clear()
        
        df = active_df.copy()
        for col in request.columns:
            if col not in df.columns: continue
            
            if request.strategy == "redact":
                df[col] = "*****"
            elif request.strategy == "hash":
                # Apply SHA256
                df[col] = df[col].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest() if x else x)
            elif request.strategy == "partial":
                # Show last 4 chars
                def mask_partial(val):
                    s = str(val)
                    if len(s) <= 4: return s
                    return "*" * (len(s) - 4) + s[-4:]
                df[col] = df[col].apply(mask_partial)
        
        active_df = df
        action_history.append(f"Masked columns {request.columns} ({request.strategy})")
        return get_df_summary(df)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- AI & Intelligence Endpoints ---

class AIChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None # Or use env var
    model: str = "gpt-4o-mini"

class DriverAnalysisRequest(BaseModel):
    target_column: str
    feature_columns: Optional[List[str]] = None # If None, use all others

@app.post("/ai/chat")
def ai_chat(request: AIChatRequest):
    global active_df
    try:
        if request.api_key:
             openai.api_key = request.api_key
        elif os.getenv("OPENAI_API_KEY"):
             openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
             # Fallback if no key: Simulate AI
             return {
                 "response": "I see you're asking about the data. (AI features require an API KEY). However, I can tell you this dataset has " + str(len(active_df) if active_df is not None else 0) + " rows.",
                 "actions": []
             }

        context = ""
        if active_df is not None:
             context = f"""
             Dataset Context:
             Rows: {len(active_df)}
             Columns: {', '.join(active_df.columns)}
             Dtypes: {active_df.dtypes.to_dict()}
             First 3 rows: {active_df.head(3).to_dict(orient='records')}
             """
        
        system_prompt = "You are an expert Data Analyst Copilot. Answer the user's question about the dataset. Be concise. If the user asks for code, provide Python/Pandas code."
        
        messages = [
            {"role": "system", "content": system_prompt + context},
            {"role": "user", "content": request.message}
        ]
        
        response = openai.chat.completions.create(
            model=request.model,
            messages=messages
        )
        
        return {
            "response": response.choices[0].message.content,
            "actions": [] # Could parse response for structured actions later
        }

    except Exception as e:
        # Graceful fallback
        print(f"AI Error: {e}")
        return {"response": f"AI Error: {str(e)}. Please check your API Key.", "actions": []}

@app.post("/analyze_drivers")
def analyze_driver_factors(request: DriverAnalysisRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        df = active_df.copy()
        target = request.target_column
        
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column {target} not found")

        # Prepare X and y
        y = df[target]
        
        # Select potential drivers (numeric + categorical)
        potential_drivers = request.feature_columns if request.feature_columns else [c for c in df.columns if c != target]
        
        # Simple preprocessing (Drop NaNs, Encode)
        temp_df = df[potential_drivers + [target]].dropna()
        if temp_df.empty: raise ValueError("Not enough data after dropping NaNs")
        
        y_clean = temp_df[target]
        X_clean = temp_df[potential_drivers]
        
        # Auto-encode objects
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object':
                 le = LabelEncoder()
                 X_clean[col] = le.fit_transform(X_clean[col].astype(str))
        
        # Train Random Forest to find importance
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        if pd.api.types.is_object_dtype(y_clean):
             model = RandomForestClassifier(n_estimators=50, random_state=42)
             le_y = LabelEncoder()
             y_clean = le_y.fit_transform(y_clean.astype(str))
             
        model.fit(X_clean, y_clean)
        
        # Get Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        drivers = []
        for i in range(min(5, len(importances))):
            col_idx = indices[i]
            drivers.append({
                "feature": X_clean.columns[col_idx],
                "importance": float(importances[col_idx]),
                "description": f"Contributes {round(importances[col_idx]*100, 1)}% to variance"
            })
            
        return {"drivers": drivers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/configure")
def configure_ai(request: AIConfigRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    return suggest_model_config(active_df, request.goal)

# --- Analytics & Engineering Endpoints ---

@app.post("/query/builder")
def query_builder(request: QueryBuilderRequest):
    global active_df
    if active_df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        df = active_df.copy()
        
        # 1. Filters
        if request.filters:
            for f in request.filters:
                col = f.get("col")
                op = f.get("op")
                val = f.get("val")
                if col in df.columns:
                    if op == "==": df = df[df[col] == val]
                    elif op == "!=": df = df[df[col] != val]
                    elif op == ">": df = df[df[col] > val]
                    elif op == "<": df = df[df[col] < val]
                    elif op == ">=": df = df[df[col] >= val]
                    elif op == "<=": df = df[df[col] <= val]
                    elif op == "contains": df = df[df[col].astype(str).str.contains(str(val), case=False)]
        
        # 2. Group By & Aggregate
        if request.groupby:
            if not request.aggregates:
                # Default count if no agg specified
                df = df.groupby(request.groupby).size().reset_index(name='count')
            else:
                agg_map = {}
                for col, func in request.aggregates.items():
                    agg_map[col] = func
                df = df.groupby(request.groupby).agg(agg_map).reset_index()
                
        # 3. Select (if not grouped)
        elif request.select:
            df = df[request.select]
            
        # 4. Sort
        if request.sort:
            for col, direction in request.sort.items():
                if col in df.columns:
                    df = df.sort_values(by=col, ascending=(direction.lower() == 'asc'))
        
        # 5. Limit
        df = df.head(request.limit)
        
        # Generate Pseudo-SQL
        sql = "SELECT " + (", ".join(request.select) if request.select else "*")
        sql += " FROM current_data"
        if request.filters:
            clauses = []
            for f in request.filters:
                val_str = f"'{f['val']}'" if isinstance(f['val'], str) else str(f['val'])
                clauses.append(f"{f['col']} {f['op']} {val_str}")
            sql += " WHERE " + " AND ".join(clauses)
        if request.groupby:
            sql += " GROUP BY " + ", ".join(request.groupby)
        if request.sort:
            sql += " ORDER BY " + ", ".join([f"{k} {v}" for k,v in request.sort.items()])
        sql += f" LIMIT {request.limit}"
        
        return {
             "summary": get_df_summary(df),
             "generated_sql": sql
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/register")
def register_feature(request: FeatureRegisterRequest):
    try:
        fid = metadata_store.register_feature(request.name, request.description, request.version, request.logic_code)
        return {"message": "Feature registered", "feature_id": fid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
def get_all_features():
    try:
        return {"features": metadata_store.get_features()}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/notebook/execute")
def execute_notebook_code(request: NotebookRequest):
    global active_df
    try:
        # Capture stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
             # Define local scope with df
             local_scope = {"df": active_df, "pd": pd, "np": np}
             exec(request.code, {}, local_scope)
             
             # If user modified df, update global??
             # For safety/complexity, let's NOT update global active_df unless explicitly requested
             # But usually notebooks mutate state. 
             if "df" in local_scope and isinstance(local_scope["df"], pd.DataFrame):
                 # OPTIONAL: update active_df if code was "df = ..."
                 # active_df = local_scope["df"] 
                 pass
        
        output = f.getvalue()
        return {"output": output}
    except Exception as e:
        return {"output": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# --- Time Travel & History ---

@app.get("/history/snapshots/{dataset_id}")
def get_snapshot_history(dataset_id: str):
    history = metadata_store.get_run_history(dataset_id, limit=50)
    # History format: row_count, col_count, schema_hash, timestamp, snapshot_path
    return [
        {
            "row_count": r[0],
            "col_count": r[1],
            "timestamp": r[3],
            "snapshot_path": r[4],
            "id": i # simple index as ID for frontend
        } 
        for i, r in enumerate(history)
    ]

class RevertRequest(BaseModel):
    snapshot_path: str

@app.post("/history/revert")
def revert_to_snapshot(request: RevertRequest):
    global active_df
    if not os.path.exists(request.snapshot_path):
         raise HTTPException(status_code=404, detail="Snapshot file not found")
    
    try:
        df = pd.read_parquet(request.snapshot_path)
        active_df = df
        
        # We should probably log this revert as a new run/state?
        # Or just reset state. Let's log it as "Reverted to..."
        if active_file_path:
             dataset_id = hashlib.md5(active_file_path.encode()).hexdigest()
             metadata_store.log_run(dataset_id, df, request.snapshot_path) # Points to same snapshot? Or new one?
             # Better to point to same if we want to save space, but logically it's a new point in time.
             # Let's save a NEW snapshot to preserve linear history of "reverting"
             new_snap = save_snapshot(df, dataset_id)
             metadata_store.log_run(dataset_id, df, new_snap)

        history_stack.clear() 
        redo_stack.clear()
        
        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DiffRequest(BaseModel):
    snapshot_path_a: str
    snapshot_path_b: str # usually current or another snapshot

@app.post("/history/diff")
def diff_snapshots(request: DiffRequest):
    if not os.path.exists(request.snapshot_path_a) or not os.path.exists(request.snapshot_path_b):
         raise HTTPException(status_code=404, detail="Snapshot file not found")
    
    try:
        df_a = pd.read_parquet(request.snapshot_path_a) # e.g. Older
        df_b = pd.read_parquet(request.snapshot_path_b) # e.g. Newer
        
        # Simple diff logic
        # 1. Schema Change
        added_cols = list(set(df_b.columns) - set(df_a.columns))
        removed_cols = list(set(df_a.columns) - set(df_b.columns))
        
        # 2. Row Counts
        rows_added = len(df_b) - len(df_a)
        
        # 3. Value Diffs (expensive for large data, maybe just summary)
        # minimal diff summary
        
        return {
            "schema_diff": {
                "added_columns": added_cols,
                "removed_columns": removed_cols
            },
            "rows_diff": rows_added,
            "summary": f"Row change: {rows_added}. Cols added: {len(added_cols)}, removed: {len(removed_cols)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboards")
def list_dashboards():
    try:
        return metadata_store.list_dashboards()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboards/{dashboard_id}")
def get_dashboard(dashboard_id: str):
    try:
        dash = metadata_store.get_dashboard(dashboard_id)
        if not dash:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return dash
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboards/save")
def save_dashboard(request: DashboardRequest):
    try:
        # Generate ID if new
        dash_id = request.id or hashlib.md5(f"{request.name}{datetime.now()}".encode()).hexdigest()
        metadata_store.save_dashboard(dash_id, request.name, request.layout)
        return {"id": dash_id, "message": "Dashboard saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
