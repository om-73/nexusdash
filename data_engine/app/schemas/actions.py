from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class PredictRequest(BaseModel):
    inputs: Dict[str, Any]

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
    algorithms: List[str] # linear, logistic, dt, rf, kmeans
    target_column: Optional[str] = None
    feature_columns: List[str]
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

class FeatureBuildRequest(BaseModel):
    name: str # New column name
    expression: str # e.g. "Sales * 0.2", "Age > 30"

class FeatureEngineerRequest(BaseModel):
    features: List[Dict[str, Any]]

class NotebookRequest(BaseModel):
    code: str

class DriversRequest(BaseModel):
    target_column: str

class DashboardRequest(BaseModel):
    id: Optional[str] = None
    name: str
    layout: List[Dict[str, Any]]

class KPICalculateRequest(BaseModel):
    column: str
    operation: str # sum, mean, count, min, max, unique

class AIProcessRequest(BaseModel):
    query: str
