from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import numpy as np
import os
import hashlib
import io
from fastapi.responses import StreamingResponse

from ...core import state
from ...schemas.actions import FileLoadRequest, CleanRequest, FeatureBuildRequest, FeatureEngineerRequest
from ...services.utils import get_df_summary, save_snapshot
from ...core.config import SNAPSHOTS_DIR

# We need access to metadata_store. Ideally inject it, but for now import global
# Since metadata_store was in the parent directory, we might need to adjust imports or move it.
# For this refactor, let's assume we can import it or reproduce it. 
# The original main.py imported it from `metadata_store`.
import sys
import traceback
# Ensure the root data_engine directory is in sys.path to find metadata_store
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from metadata_store import MetadataStore

metadata_store = MetadataStore()

router = APIRouter()

@router.get("/state")
def get_state():
    df = state.get_active_df()
    if df is None:
        return None 
    try:
        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load")
def load_data(request: FileLoadRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        if request.file_type == "csv":
            # Auto-detect delimiter logic
            sep = ','
            try:
                import csv
                # Try reading with utf-8 first
                encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
                sample = ""
                for enc in encodings:
                    try:
                        with open(request.file_path, 'r', encoding=enc) as f:
                            sample = f.read(2048)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if not sample:
                    with open(request.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                         sample = f.read(2048)

                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
            except Exception as e:
                print(f"Sniffer failed, falling back to default sep: {e}")
            
            try:
                try:
                    df = pd.read_csv(request.file_path, sep=sep)
                except UnicodeDecodeError:
                    print("UTF-8 read failed, trying latin1")
                    df = pd.read_csv(request.file_path, sep=sep, encoding='latin1')
                except Exception:
                     try:
                         df = pd.read_csv(request.file_path)
                     except UnicodeDecodeError:
                         df = pd.read_csv(request.file_path, encoding='latin1')

            except Exception as e:
                print(f"All CSV load attempts failed, trying engine='python'")
                df = pd.read_csv(request.file_path, sep=sep, engine='python', encoding_errors='replace')
                
            if len(df.columns) == 1 and sep == ',':
                print("Detected single column with comma separator, retrying with tab/auto...")
                try:
                    df_tab = pd.read_csv(request.file_path, sep='\t')
                    if len(df_tab.columns) > 1:
                        df = df_tab
                    else:
                        df_auto = pd.read_csv(request.file_path, sep=None, engine='python')
                        if len(df_auto.columns) > 1:
                            df = df_auto
                except Exception as e:
                    print(f"Fallback retry failed: {e}")

        elif request.file_type in ["xlsx", "excel"]:
            df = pd.read_excel(request.file_path)
        elif request.file_type == "zip":
            # Pandas can natively read `.zip` containing a single CSV
            print(f"Detected zip file, attempting to load via pandas compression='zip'")
            try:
                df = pd.read_csv(request.file_path, compression='zip')
            except Exception as e:
                import zipfile
                print(f"Pandas native zip read failed: {e}. Attempting manual extraction.")
                with zipfile.ZipFile(request.file_path, 'r') as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        raise HTTPException(status_code=400, detail="No CSV file found inside the zip archive.")
                    with z.open(csv_files[0]) as f:
                        df = pd.read_csv(f)
        else:
             raise HTTPException(status_code=400, detail="Unsupported file type")
        
        state.set_active_df(df)
        state.reset_state(request.file_path)
        state.action_history.append("Loaded Data")
        
        temp_id = hashlib.md5(request.file_path.encode()).hexdigest()
        snap_path = save_snapshot(df, temp_id)
        metadata_store.register_dataset(temp_id, os.path.basename(request.file_path), "upload")
        metadata_store.log_run(temp_id, df, snap_path)
        
        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean")
def clean_data(request: CleanRequest):
    df = state.get_active_df()
    if df is None:
        raise HTTPException(status_code=400, detail="Session reset. Please reload your dataset.")
    try:
        # PUSH HISTORY
        state.push_to_history(df.copy())
        state.redo_stack.clear()
        state.redo_action_history.clear()

        action_desc = f"Performed {request.operation}"
        
        df = df.copy() # Work on copy
        
        # --- Cleaning Logic ---
        if request.operation == "dropna":
            if request.columns: df.dropna(subset=request.columns, inplace=True)
            else: df.dropna(inplace=True)
            action_desc = "Dropped Missing Values"
            
        elif request.operation == "fillna":
            if request.value is not None:
                if request.columns:
                    for col in request.columns:
                        try:
                            df[col] = df[col].fillna(request.value)
                        except Exception:
                            df[col] = df[col].astype(object).fillna(request.value)
                else:
                    df.fillna(request.value, inplace=True)
                action_desc = f"Filled Missing Values ({request.value})"
            elif request.strategy:
                target_cols = request.columns if request.columns else df.columns
                numeric_cols = df[target_cols].select_dtypes(include=np.number).columns
                
                if request.strategy == "mean":
                    if not numeric_cols.empty:
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif request.strategy == "median":
                    if not numeric_cols.empty:
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif request.strategy == "mode":
                    for col in target_cols:
                        try:
                            m = df[col].mode()
                            if not m.empty: df[col] = df[col].fillna(m[0])
                        except Exception: continue
                action_desc = f"Filled Missing Values ({request.strategy})"

        elif request.operation == "drop_duplicates": 
            df.drop_duplicates(inplace=True)
            action_desc = "Dropped Duplicates"
            
        elif request.operation == "drop_columns": 
            if request.columns: df.drop(columns=request.columns, inplace=True)
            action_desc = f"Dropped Columns: {request.columns}"
            
        elif request.operation == "rename_columns":
            if request.rename_map: df.rename(columns=request.rename_map, inplace=True)
            
        elif request.operation == "remove_outliers":
            if request.columns:
                for col in request.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            action_desc = "Removed Outliers"
            
        elif request.operation == "encode_columns":
            if request.columns:
                for col in request.columns:
                    if request.strategy == "label":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                    elif request.strategy == "onehot":
                        df = pd.get_dummies(df, columns=[col], prefix=col)
            action_desc = f"Encoded Columns ({request.strategy})"

        elif request.operation == "normalize":
            if request.columns:
                from sklearn.preprocessing import MinMaxScaler, StandardScaler
                scaler = None
                if request.strategy == "minmax": scaler = MinMaxScaler()
                elif request.strategy == "standard": scaler = StandardScaler()
                
                if scaler:
                    df[request.columns] = scaler.fit_transform(df[request.columns])
            action_desc = f"Normalized Columns ({request.strategy})"
        
        state.set_active_df(df)
        state.action_history.append(action_desc)
        
        # Snapshotting
        if state.active_file_path:
            dataset_id = hashlib.md5(state.active_file_path.encode()).hexdigest()
            snap_path = save_snapshot(df, dataset_id)
            metadata_store.log_run(dataset_id, df, snap_path)
        
        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/undo")
def undo_action():
    if not state.history_stack:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    try:
        active_df = state.get_active_df()
        state.redo_stack.append(active_df)
        if state.action_history: state.redo_action_history.append(state.action_history.pop())
        
        state.set_active_df(state.history_stack.pop())
        return get_df_summary(state.get_active_df())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/redo")
def redo_action():
    if not state.redo_stack:
        raise HTTPException(status_code=400, detail="Nothing to redo")
    try:
        active_df = state.get_active_df()
        state.push_to_history(active_df)
        if state.redo_action_history: state.action_history.append(state.redo_action_history.pop())
        
        state.set_active_df(state.redo_stack.pop())
        return get_df_summary(state.get_active_df())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export")
def export_data():
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=cleaned_data.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature/add")
def add_feature(request: FeatureBuildRequest):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        state.push_to_history(df.copy())
        df = df.copy()
        
        # Preprocess expression: wrap columns with spaces in backticks so df.eval works seamlessly
        expression = request.expression
        sorted_cols = sorted(df.columns, key=len, reverse=True)
        for col in sorted_cols:
            if ' ' in col:
                backticked = f"`{col}`"
                if col in expression and backticked not in expression:
                    expression = expression.replace(col, backticked)

        try:
            df[request.name] = df.eval(expression)
            if df[request.name].dtype == 'bool':
                df[request.name] = df[request.name].astype(int)
        except Exception as eval_err:
            raise HTTPException(status_code=400, detail=f"Invalid formula: {str(eval_err)}")
            
        state.set_active_df(df)
        state.action_history.append(f"Created feature '{request.name}' using '{request.expression}'")
        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline")
def get_pipeline():
    return {"steps": state.action_history}

@router.post("/feature/engineer")
def engineer_features(request: FeatureEngineerRequest):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        state.push_to_history(df.copy())
        df = df.copy()
        
        for feature in request.features:
            name = feature.get('name')
            ftype = feature.get('type')
            config = feature.get('config', {})
            
            if not name or name in df.columns:
                continue # Skip invalid or duplicate names
                
            if ftype == 'threshold':
                col = config.get('column')
                op = config.get('operator')
                thresh = config.get('threshold')
                if col in df.columns and thresh is not None:
                    if op == '>': df[name] = (df[col] > thresh).astype(int)
                    elif op == '>=': df[name] = (df[col] >= thresh).astype(int)
                    elif op == '<': df[name] = (df[col] < thresh).astype(int)
                    elif op == '<=': df[name] = (df[col] <= thresh).astype(int)
                    elif op == '==': df[name] = (df[col] == thresh).astype(int)
            
            elif ftype == 'quantile':
                col = config.get('column')
                op = config.get('operator')
                q_val = config.get('quantile')
                if col in df.columns and q_val is not None:
                    thresh = df[col].quantile(float(q_val))
                    if op == '>': df[name] = (df[col] > thresh).astype(int)
                    elif op == '>=': df[name] = (df[col] >= thresh).astype(int)
                    elif op == '<': df[name] = (df[col] < thresh).astype(int)
                    elif op == '<=': df[name] = (df[col] <= thresh).astype(int)
            
            elif ftype == 'conditional':
                col1 = config.get('col1')
                col2 = config.get('col2')
                logic = config.get('logic')
                if col1 in df.columns and col2 in df.columns:
                    # Creating a simple binary flag based on AND/OR logic
                    # Assumes columns are boolean/binary. If not, this is a naïve truthiness check
                    if logic == 'AND': df[name] = ((df[col1].astype(bool)) & (df[col2].astype(bool))).astype(int)
                    elif logic == 'OR': df[name] = ((df[col1].astype(bool)) | (df[col2].astype(bool))).astype(int)
            
            elif ftype == 'binning':
                col = config.get('column')
                bins = config.get('bins')
                if col in df.columns and bins:
                    try:
                        df[name] = pd.qcut(df[col], q=int(bins), labels=False, duplicates='drop')
                    except Exception:
                        df[name] = pd.cut(df[col], bins=int(bins), labels=False)
                        
            elif ftype == 'interaction':
                col1 = config.get('col1')
                col2 = config.get('col2')
                op = config.get('operator')
                if col1 in df.columns and col2 in df.columns:
                    if op == 'multiply': df[name] = df[col1] * df[col2]
                    elif op == 'divide':
                        # Avoid div by zero
                        df[name] = np.where(df[col2] == 0, 0, df[col1] / df[col2])
                    elif op == 'add': df[name] = df[col1] + df[col2]
                    elif op == 'subtract': df[name] = df[col1] - df[col2]
                    
            elif ftype == 'encode':
                col = config.get('column')
                strategy = config.get('strategy')
                if col in df.columns:
                    if strategy == 'onehot':
                        # Get dummies, drop original column
                        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)
                    elif strategy == 'label':
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df[name] = le.fit_transform(df[col].astype(str))
                        
            elif ftype == 'select':
                strategy = config.get('strategy')
                if strategy == 'variance':
                    # Drop zero variance columns
                    from sklearn.feature_selection import VarianceThreshold
                    numeric_cols = df.select_dtypes(include=np.number).columns
                    if not numeric_cols.empty:
                        vt = VarianceThreshold(threshold=0)
                        vt.fit(df[numeric_cols])
                        cols_to_drop = [c for i, c in enumerate(numeric_cols) if not vt.get_support()[i]]
                        if cols_to_drop:
                            df.drop(columns=cols_to_drop, inplace=True)
                elif strategy == 'correlation':
                    thresh = float(config.get('threshold', 0.8))
                    numeric_cols = df.select_dtypes(include=np.number).columns
                    if not numeric_cols.empty:
                        corr_matrix = df[numeric_cols].corr().abs()
                        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > thresh)]
                        if to_drop:
                            df.drop(columns=to_drop, inplace=True)
                            
            elif ftype == 'pca':
                strategy = config.get('strategy')
                n_components = config.get('n_components')
                numeric_cols = df.select_dtypes(include=np.number).columns
                if not numeric_cols.empty and len(numeric_cols) > 1:
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    # PCA requires scaling
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
                    
                    # strategy can be "components" (int) or "variance" (float)
                    if strategy == 'variance':
                        # float target
                        n_components = float(n_components) if n_components else 0.95
                    else:
                        # int target
                        n_components = int(n_components) if n_components else 2
                        
                    # Cap n_components to min(n_samples, n_features)
                    max_components = min(df.shape[0], len(numeric_cols))
                    if isinstance(n_components, int) and n_components > max_components:
                        n_components = max_components
                    
                    pca = PCA(n_components=n_components)
                    pca_results = pca.fit_transform(scaled_data)
                    
                    actual_components = pca_results.shape[1]
                    
                    # Erase old numeric columns safely, keep categoricals
                    df.drop(columns=numeric_cols, inplace=True)
                    
                    # Merge new components back in
                    for i in range(actual_components):
                        df[f'{name}_{i+1}'] = pca_results[:, i]

        state.set_active_df(df)
        state.action_history.append(f"Engineered {len(request.features)} new features")
        
        # Save snapshot
        if state.active_file_path:
            import hashlib
            from ...services.utils import save_snapshot
            dataset_id = hashlib.md5(state.active_file_path.encode()).hexdigest()
            snap_path = save_snapshot(df, dataset_id)
            metadata_store.log_run(dataset_id, df, snap_path)

        return get_df_summary(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uuid

@router.post("/dashboards/save")
def save_dashboard(payload: dict):
    db_id = str(uuid.uuid4())
    name = payload.get("name", "Untitled Dashboard")
    layout = payload.get("layout", [])
    try:
        metadata_store.save_dashboard(db_id, name, layout)
        return {"id": db_id, "message": "Dashboard saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboards")
def list_dashboards():
    try:
        return metadata_store.list_dashboards()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboards/{dashboard_id}")
def get_dashboard(dashboard_id: str):
    try:
        dash = metadata_store.get_dashboard(dashboard_id)
        if not dash:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return dash
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
