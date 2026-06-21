from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os
import joblib
import json
import sys
import traceback
# Ensure the root data_engine directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from ...core import state
from ...schemas.actions import TrainRequest, PredictRequest, DriversRequest, AIConfigRequest

router = APIRouter()

@router.post("/train")
def train_model(request: TrainRequest):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    
    # Lazy imports to save memory
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, RidgeClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    
    try:
        df = df.copy()
        
        # 1. Parse and coerce numeric string columns (features and target) before dropping nulls.
        # This resolves cases where rating or other continuous values are loaded as string object columns.
        cols_to_check = list(request.feature_columns)
        if request.target_column not in cols_to_check:
            cols_to_check.append(request.target_column)

        import re
        def extract_first_number(val):
            if pd.isna(val) or val is None:
                return np.nan
            # Extract first float or int (e.g. '4.1 • 52 mins' -> 4.1, '4 • 34 mins' -> 4.0)
            match = re.search(r'[-+]?\d*\.\d+|\d+', str(val))
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return np.nan
            return np.nan

        for col in cols_to_check:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Attempt direct conversion first, fallback to regex digit extraction
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notna().sum() / len(df) < 0.8 if len(df) > 0 else True:
                    coerced = df[col].apply(extract_first_number)
                
                non_nan_ratio = coerced.notna().sum() / len(df) if len(df) > 0 else 0
                if col == request.target_column and request.problem_type == "regression":
                    # Only coerce target if it is actually numeric-ish (>80% numbers)
                    if non_nan_ratio > 0.8:
                        df[col] = coerced
                else:
                    if non_nan_ratio > 0.8:
                        df[col] = coerced

        # 2. Cleaning - now dropna will also drop rows with invalid coerced values (e.g. "N/A" -> NaN)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=[request.target_column] + request.feature_columns)

        if df.empty:
            raise ValueError("All rows contain missing values in the selected columns. Please clean your data first.")
        
        original_dtypes = {col: str(df[col].dtype) for col in request.feature_columns}
        
        # 3. Encoding - encode any remaining categorical feature columns
        encoders = {}
        for col in request.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        X = df[request.feature_columns]
        
        # Clustering
        if request.problem_type == "clustering":
            model = KMeans(n_clusters=3)
            model.fit(X)
            df['Cluster'] = model.labels_
            return {"clusters": df[['Cluster']].head(20).to_dict(orient='records')}

        # Regression/Classification Target Setup
        y = df[request.target_column]

        if request.problem_type == "regression" and not pd.api.types.is_numeric_dtype(y):
            # If target is still non-numeric, fallback to classification
            request.problem_type = "classification"

        if request.problem_type == "classification" and pd.api.types.is_numeric_dtype(y):
            # If target is numeric but continuous, classification will fail.
            if y.nunique() > 20: # arbitrary threshold for continuous
                raise ValueError("Target variable appears to be continuous, but problem type is classification. Please use regression or bin the target variable.")

        if request.problem_type == "classification":
            # Prevent OOM/timeouts by automatically grouping rare categories
            if y.nunique() > 50:
                top_classes = y.value_counts().nlargest(49).index
                y = y.where(y.isin(top_classes), 'Other')

        if not pd.api.types.is_numeric_dtype(y):
             le_target = LabelEncoder()
             y = le_target.fit_transform(y.astype(str))
             encoders[request.target_column] = le_target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size)
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Not enough data to split into training and testing sets.")

        # Define model mappings
        if request.problem_type == "regression":
            algo_map = {
                "linear": LinearRegression(),
                "ridge": Ridge(),
                "lasso": Lasso(),
                "dt": DecisionTreeRegressor(max_depth=10),
                "rf": RandomForestRegressor(n_estimators=20, max_depth=10),
                "gbr": GradientBoostingRegressor(n_estimators=20, max_depth=5),
                "ada": AdaBoostRegressor(n_estimators=20),
                "svr": SVR(max_iter=1000)
            }
            pretty_names = {
                "linear": "Linear Regression",
                "ridge": "Ridge Regression (L2)",
                "lasso": "Lasso Regression (L1)",
                "dt": "Decision Tree",
                "rf": "Random Forest",
                "gbr": "Gradient Boosting",
                "ada": "AdaBoost",
                "svr": "Support Vector Regression (SVR)"
            }
        else:
            algo_map = {
                "logistic": LogisticRegression(max_iter=200),
                "nb": GaussianNB(),
                "dt": DecisionTreeClassifier(max_depth=10),
                "rf": RandomForestClassifier(n_estimators=20, max_depth=10),
                "gbc": GradientBoostingClassifier(n_estimators=20, max_depth=5),
                "ada": AdaBoostClassifier(n_estimators=20),
                "knn": KNeighborsClassifier(),
                "svm": SVC(probability=True, max_iter=1000)
            }
            pretty_names = {
                "logistic": "Logistic Regression",
                "nb": "Naive Bayes (Gaussian)",
                "dt": "Decision Tree",
                "rf": "Random Forest",
                "gbc": "Gradient Boosting",
                "ada": "AdaBoost",
                "knn": "K-Nearest Neighbors (KNN)",
                "svm": "Support Vector Machine (SVM)"
            }

        # Default to all available algorithms for the task if none specified
        selected_algos = request.algorithms if request.algorithms else list(algo_map.keys())
        
        trained_models = {}
        metrics_dict = {}
        preds_dict = {}
        comparison_list = []
        
        for algo_name in selected_algos:
            if algo_name not in algo_map:
                continue
            model_inst = algo_map[algo_name]
            try:
                model_inst.fit(X_train, y_train)
                preds_val = model_inst.predict(X_test)
                
                if request.problem_type == "regression":
                    r2 = float(r2_score(y_test, preds_val))
                    mse = float(mean_squared_error(y_test, preds_val))
                    metrics = {"r2": r2, "mse": mse}
                    score = r2
                else:
                    acc = float(accuracy_score(y_test, preds_val))
                    metrics = {"accuracy": acc}
                    score = acc
                    
                trained_models[algo_name] = model_inst
                metrics_dict[algo_name] = metrics
                preds_dict[algo_name] = preds_val
                
                comparison_list.append({
                    "algorithm": pretty_names.get(algo_name, algo_name),
                    "metrics": metrics,
                    "score": score
                })
            except Exception as train_err:
                print(f"Failed to train {algo_name}: {train_err}")
                
        if not trained_models:
            raise ValueError("All selected algorithms failed to train. Please check your data features and target.")
            
        # Select best model based on score (R2 or Accuracy)
        comparison_list = sorted(comparison_list, key=lambda x: x["score"], reverse=True)
        best_algo_item = comparison_list[0]
        best_algo_key = [k for k, v in pretty_names.items() if v == best_algo_item["algorithm"]][0]
        
        model = trained_models[best_algo_key]
        preds = preds_dict[best_algo_key]
        
        # Prepare actual and predictions preview lists, decoding classification labels if encoded
        actual_list = (y_test if isinstance(y_test, pd.Series) else pd.Series(y_test)).head(10).tolist()
        preds_list = preds[:10].tolist()
        
        if request.target_column in encoders:
            try:
                le_target = encoders[request.target_column]
                actual_list = [str(x) for x in le_target.inverse_transform(actual_list)]
                preds_list = [str(x) for x in le_target.inverse_transform(preds_list)]
            except Exception as e:
                print(f"Warning: Failed to decode previews: {e}")
                
        # Clean comparison list scores for response
        for item in comparison_list:
            item.pop("score", None)
            
        # Prepare Response Data
        response_data = {
            "target_column": request.target_column,
            "problem_type": request.problem_type,
            "metrics": metrics_dict[best_algo_key],
            "best_algorithm": pretty_names[best_algo_key],
            "download_available": True,
            "actual_preview": [float(v) if isinstance(v, (np.generic, np.ndarray)) else v for v in actual_list],
            "predictions_preview": [float(v) if isinstance(v, (np.generic, np.ndarray)) else v for v in preds_list],
            "comparison": comparison_list,
            "feature_importance": {},
            "classification_report": {},
            "features": list(X.columns),
            "feature_types": original_dtypes
        }
        
        # Add classification report if classification
        if request.problem_type != "regression":
            from sklearn.metrics import classification_report
            report = classification_report(y_test, preds, output_dict=True)
            def sanitize_dict(d):
                return {k: float(v) if isinstance(v, (np.generic, np.ndarray)) else (sanitize_dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
            response_data["classification_report"] = sanitize_dict(report)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            feat_imp = {f: float(i) for f, i in zip(X.columns, imps)}
            feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            response_data["feature_importance"] = feat_imp
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coef_abs = np.mean(np.abs(coefs), axis=0)
            else:
                coef_abs = np.abs(coefs)
            feat_imp = {f: float(i) for f, i in zip(X.columns, coef_abs)}
            feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            response_data["feature_importance"] = feat_imp

        # Save Model & Metadata & Encoders
        if not os.path.exists("models"): os.makedirs("models")
        joblib.dump(model, "models/model.pkl")
        joblib.dump(encoders, "models/encoders.pkl")
        
        # Save metadata to disk for persistence
        try:
            with open("models/metadata.json", "w") as f:
                json.dump(response_data, f)
        except Exception as e:
            print(f"Warning: Failed to save metadata to disk: {e}")

        state.set_active_model(model)
        state.set_active_model_metadata(response_data)
        
        return response_data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/download")
def download_model():
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No trained model found")
    return FileResponse(model_path, media_type='application/octet-stream', filename="model.pkl")

@router.get("/model/metadata")
def get_model_metadata():
    metadata = state.get_active_model_metadata()
    if metadata is None:
        # Try loading from disk
        metadata_path = "models/metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                state.set_active_model_metadata(metadata)
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to load metadata from disk")
        
    if metadata is None:
        raise HTTPException(status_code=404, detail="No model metadata found. Please train a model first.")
    return metadata

@router.post("/model/predict")
def predict_model(request: PredictRequest):
    model = state.get_active_model()
    if model is None:
        # Try loading from disk
        model_path = "models/model.pkl"
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                state.set_active_model(model)
            except Exception:
                raise HTTPException(status_code=400, detail="Failed to load model from disk")
        else:
            raise HTTPException(status_code=400, detail="No model trained yet")
    
    try:
        metadata = state.get_active_model_metadata()
        if not metadata:
            metadata_path = "models/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
        
        # Convert inputs dict to DataFrame for prediction
        input_data = {}
        if metadata and "feature_importance" in metadata:
            # We ordered features in feature_importance but the actual input might use another order
            # Better to rely on the metadata features order if available, else fallback
            feature_cols = list(metadata.get("features", request.inputs.keys()))
            for col in feature_cols:
                val = request.inputs.get(col, 0)
                try:
                    val = float(val)
                except:
                    pass
                input_data[col] = [val]
        else:
            for k, v in request.inputs.items():
                input_data[k] = [v]

        input_df = pd.DataFrame(input_data)
        
        # Load encoders
        encoders = {}
        encoders_path = "models/encoders.pkl"
        if os.path.exists(encoders_path):
            try:
                encoders = joblib.load(encoders_path)
            except Exception:
                pass
                
        # Apply encoding logic using the saved encoders
        for col in input_df.columns:
            if col in encoders:
                try:
                    # Safely transform; if unseen label, we map to a default or 0
                    le = encoders[col]
                    known_classes = set(le.classes_)
                    
                    mapped = []
                    for val in input_df[col]:
                        val_str = str(val) if pd.api.types.is_numeric_dtype(type(val)) else val
                        if val in known_classes:
                            mapped.append(le.transform([val])[0])
                        elif val_str in known_classes:
                            mapped.append(le.transform([val_str])[0])
                        else:
                            mapped.append(0) # Fallback for unknown category
                    input_df[col] = mapped
                except Exception as e:
                    print(f"Encoder error for col {col}: {e}")
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            elif not pd.api.types.is_numeric_dtype(input_df[col]):
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        prediction = model.predict(input_df)
        
        # Handle numpy return types
        pred_val = prediction[0]
        if isinstance(pred_val, (np.generic, np.ndarray)):
            pred_val = pred_val.item()
            
        # If target column was encoded, decode the prediction back to its string representation
        if metadata and "target_column" in metadata:
            target_col = metadata["target_column"]
            if target_col in encoders:
                try:
                    le = encoders[target_col]
                    pred_idx = int(round(pred_val))
                    decoded = le.inverse_transform([pred_idx])[0]
                    pred_val = str(decoded)
                except Exception as e:
                    print(f"Failed to decode prediction target: {e}")

        return {"prediction": pred_val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
@router.post("/ai/configure")
def configure_ai(request: AIConfigRequest):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        # Dynamic import of ai_engine (as in eda.py)
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
        from ai_engine import suggest_model_config
        
        return suggest_model_config(df, request.goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/analyze/drivers")
def analyze_drivers(request: DriversRequest):
    df = state.get_active_df()
    if df is None: raise HTTPException(status_code=400, detail="No data loaded")
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        target = request.target_column
        df = df.dropna(subset=[target])
        X = df.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
        y = df[target]
        
        if X.empty:
            raise ValueError("No numeric features available to analyze drivers.")
            
        is_regression = pd.api.types.is_numeric_dtype(y)
        
        if not is_regression: 
            if y.nunique() > 50:
                raise ValueError(f"Target variable '{target}' has too many unique categories ({y.nunique()}) for driver analysis.")
            y = y.astype(str)
            model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=1)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1)
        
        model.fit(X, y)
        
        imps = model.feature_importances_
        drivers = [{"feature": f, "importance": float(i)} for f, i in zip(X.columns, imps)]
        drivers.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            "target": target,
            "problem_type": "Regression" if is_regression else "Classification",
            "drivers": drivers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
