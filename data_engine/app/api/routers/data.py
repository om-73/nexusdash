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

def _read_single_file_to_df(file_path: str, file_type: str = "csv") -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        file_type = "excel"
    elif ext == ".zip":
        file_type = "zip"

    if file_type == "csv":
        sep = ','
        try:
            import csv
            encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
            sample = ""
            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        sample = f.read(2048)
                    break
                except UnicodeDecodeError:
                    continue
            if not sample:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(2048)
            dialect = csv.Sniffer().sniff(sample)
            sep = dialect.delimiter
        except Exception as e:
            print(f"Sniffer failed, falling back to default sep: {e}")

        try:
            try:
                df = pd.read_csv(file_path, sep=sep)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep=sep, encoding='latin1')
            except Exception:
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
        except Exception:
            df = pd.read_csv(file_path, sep=sep, engine='python', encoding_errors='replace')

        if len(df.columns) == 1 and sep == ',':
            try:
                df_tab = pd.read_csv(file_path, sep='\t')
                if len(df_tab.columns) > 1:
                    df = df_tab
                else:
                    df_auto = pd.read_csv(file_path, sep=None, engine='python')
                    if len(df_auto.columns) > 1:
                        df = df_auto
            except Exception as e:
                print(f"Fallback retry failed: {e}")
        return df

    elif file_type in ["xlsx", "excel"]:
        return pd.read_excel(file_path)
    elif file_type == "zip":
        try:
            return pd.read_csv(file_path, compression='zip')
        except Exception:
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise HTTPException(status_code=400, detail="No CSV file found inside the zip archive.")
                with z.open(csv_files[0]) as f:
                    return pd.read_csv(f)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

@router.post("/load")
def load_data(request: FileLoadRequest):
    try:
        # Multi-file load / combine
        if request.file_paths and len(request.file_paths) > 1:
            dfs = []
            valid_paths = []
            for fp in request.file_paths:
                if os.path.exists(fp):
                    loaded_df = _read_single_file_to_df(fp, request.file_type)
                    # Clean and strip column names to avoid whitespace mismatch
                    loaded_df.columns = [str(c).strip() for c in loaded_df.columns]
                    dfs.append(loaded_df)
                    valid_paths.append(fp)
            if not dfs:
                raise HTTPException(status_code=404, detail="None of the specified files could be found.")

            combine_mode = (request.combine_method or "auto").lower()
            join_key = request.join_key.strip() if request.join_key else None

            # Calculate schema overlap between dataframes
            all_col_sets = [set(d.columns) for d in dfs]
            shared_all = set.intersection(*all_col_sets)
            pairwise_overlap = all_col_sets[0].intersection(all_col_sets[1])
            min_cols = min(len(all_col_sets[0]), len(all_col_sets[1]))
            overlap_ratio = len(pairwise_overlap) / max(1, min_cols)

            df = None

            # -------------------------------------------------------------
            # Mode 1: Explicit or Auto Relational Key Join ("merge")
            # -------------------------------------------------------------
            target_key = join_key
            if not target_key and combine_mode in ["auto", "merge"] and shared_all:
                for cand in shared_all:
                    c_low = cand.lower()
                    if any(k in c_low for k in ["id", "key", "code", "account", "user", "username", "email", "platform", "date"]):
                        target_key = cand
                        break
                if not target_key and len(shared_all) == 1:
                    target_key = list(shared_all)[0]

            if combine_mode == "merge" or (combine_mode == "auto" and target_key):
                if target_key:
                    df = dfs[0].copy()
                    for idx, next_df in enumerate(dfs[1:], 1):
                        next_copy = next_df.copy()
                        k1 = next((c for c in df.columns if c.lower() == target_key.lower()), None)
                        k2 = next((c for c in next_copy.columns if c.lower() == target_key.lower()), None)
                        if k1 and k2:
                            df[k1] = df[k1].astype(str).str.strip()
                            next_copy[k2] = next_copy[k2].astype(str).str.strip()
                            df = pd.merge(df, next_copy, left_on=k1, right_on=k2, how="outer", suffixes=("", f"_{idx+1}"))
                            if k1 != k2 and k2 in df.columns:
                                df.drop(columns=[k2], inplace=True)
                        else:
                            df = pd.concat([df.reset_index(drop=True), next_copy.reset_index(drop=True)], axis=1)
                else:
                    combine_mode = "columns"

            # -------------------------------------------------------------
            # Mode 2: Horizontal Side-by-Side Merge ("columns", "side_by_side")
            # Also selected by "auto" when schemas are disjoint (overlap_ratio < 0.4)
            # -------------------------------------------------------------
            if df is None and (combine_mode in ["columns", "side_by_side", "horizontal"] or (combine_mode == "auto" and overlap_ratio < 0.4)):
                renamed_dfs = []
                seen_cols = set()
                for idx, d in enumerate(dfs):
                    d_copy = d.reset_index(drop=True).copy()
                    new_cols = []
                    for c in d_copy.columns:
                        if c in seen_cols:
                            new_c = f"{c}_{idx+1}"
                            new_cols.append(new_c)
                            seen_cols.add(new_c)
                        else:
                            new_cols.append(c)
                            seen_cols.add(c)
                    d_copy.columns = new_cols
                    renamed_dfs.append(d_copy)

                max_len = max(len(d) for d in renamed_dfs)
                aligned_dfs = []
                for d in renamed_dfs:
                    if len(d) < max_len:
                        repeat_times = (max_len // len(d)) + 1
                        padded = pd.concat([d] * repeat_times, ignore_index=True).iloc[:max_len]
                        aligned_dfs.append(padded)
                    else:
                        aligned_dfs.append(d)

                df = pd.concat(aligned_dfs, axis=1)

            # -------------------------------------------------------------
            # Mode 3: Vertical Row Union ("concat", "rows", "vertical")
            # Used when schemas match (overlap_ratio >= 0.4)
            # -------------------------------------------------------------
            if df is None:
                df = pd.concat(dfs, ignore_index=True)

            # Ensure strictly unique column names across all combined dataframes
            if df.columns.duplicated().any():
                cols = pd.Series(df.columns)
                for dup in df.columns[df.columns.duplicated()].unique():
                    dup_indices = cols[cols == dup].index.tolist()
                    for i, dup_idx in enumerate(dup_indices[1:], start=2):
                        cols[dup_idx] = f"{dup}_{i}"
                df.columns = cols

            primary_path = valid_paths[0]
            state.set_active_df(df)
            state.reset_state(primary_path)
            state.action_history.append(f"Loaded and combined {len(dfs)} datasets")

            temp_id = hashlib.md5("".join(valid_paths).encode()).hexdigest()
            snap_path = save_snapshot(df, temp_id)
            metadata_store.register_dataset(temp_id, f"Combined ({len(dfs)} files)", "upload")
            metadata_store.log_run(temp_id, df, snap_path)

            summary = get_df_summary(df)
            summary["is_combined"] = True
            summary["combined_files_count"] = len(dfs)
            return summary

        # Single file load
        file_path = request.file_path or (request.file_paths[0] if request.file_paths else None)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        df = _read_single_file_to_df(file_path, request.file_type)
        if df.columns.duplicated().any():
            cols = pd.Series(df.columns)
            for dup in df.columns[df.columns.duplicated()].unique():
                dup_indices = cols[cols == dup].index.tolist()
                for i, dup_idx in enumerate(dup_indices[1:], start=2):
                    cols[dup_idx] = f"{dup}_{i}"
            df.columns = cols

        state.set_active_df(df)
        state.reset_state(file_path)
        state.action_history.append("Loaded Data")

        temp_id = hashlib.md5(file_path.encode()).hexdigest()
        snap_path = save_snapshot(df, temp_id)
        metadata_store.register_dataset(temp_id, os.path.basename(file_path), "upload")
        metadata_store.log_run(temp_id, df, snap_path)

        return get_df_summary(df)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
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
