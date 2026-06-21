from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routers import data, eda, ml, db

app = FastAPI(title="Data Engine API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(data.router, tags=["Data"])
app.include_router(eda.router, tags=["EDA"])
app.include_router(ml.router, tags=["ML"])
app.include_router(db.router, tags=["Database"])

@app.get("/")
@app.head("/")
def read_root():
    return {"message": "Data Engine v2 (Modular) is running"}
