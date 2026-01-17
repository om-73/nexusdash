# Data Analysis Platform (Enterprise Edition)

A comprehensive data intelligence platform designed for modern enterprises. Seamlessly connect, clean, visualize, and analyze your data using advanced AI and Machine Learning capabilities.

## 🚀 Key Features

### 1. Data Connectivity & Management
- **Universal Connectors**: Upload CSV/Excel (with robust encoding support for UTF-8, Latin1, CP1252) or connect to **Snowflake, BigQuery, Redshift, PostgreSQL, and MongoDB**.
- **Data Contracts**: Define and enforce JSON-based schema validation on ingestion.
- **Observability**: Automatic freshness checks, volume anomaly detection, and schema change alerts.
- **PII Protection**: Auto-detection and masking of sensitive info (Emails, SSNs, Credit Cards).

### 2. Smart Data Quality
- **Auto-Cleaning**: Drop missing values, fill with median/mode, remove duplicates, and handle outliers.
- **Data Quality Score**: Real-time scoring based on completeness, uniqueness, and consistency.
- **Snapshot History**: track changes with "Time Travel" undo/redo capabilities and version diffing.

### 3. Advanced Logistics & Analytics
- **Automated EDA**: Instant distribution plots, correlation heatmaps, and statistical summaries.
- **Key Drivers Analysis**: Automatically identify which factors most influence your target metric (e.g., "Why did Sales drop?").
- **Custom Dashboards**: Build and persist your own dashboard layouts with Bar, Line, and Pie charts.
- **Natural Language Query**: Ask questions like "Show customers from NY with spend > 500" in plain English.

### 4. Machine Learning Studio
- **AutoML**: Select multiple algorithms simultaneously (e.g. Linear + Random Forest) to compare performance metrics (R2, RMSE, Accuracy) side-by-side.
- **Model Training**: Train Regression, Classification, or Clustering models.
- **Model Export**: Download the best-performing model as a `.pkl` file for production deployment.
- **Feature Store**: Centralized repository for managing and sharing ML features.

### 5. Developer Tools
- **Query Builder**: Visual SQL interface for non-technical users.
- **Embedded Notebooks**: Execute Python code directly within the platform.
- **Audit Logs**: Comprehensive tracking of all user actions.

## 🛠 Tech Stack

- **Frontend**: React.js, TailwindCSS, Recharts, Lucide Icons
- **Backend Proxy**: Node.js, Express
- **Data Engine**: Python 3, FastAPI, Pandas, Scikit-learn, Joblib
- **Database**: SQLite (Metadata & Dashboard configurations)

## 📦 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-repo/data-visualizer.git
    cd data-visualizer
    ```

2.  **Prerequisites**
    - Node.js (v16+)
    - Python (v3.9+)

3.  **Start the Development Environment**
    We provide a handy script to start all services (Frontend, Backend, Data Engine):
    ```bash
    ./start-dev.sh
    ```
    This will launch:
    - Frontend: `http://localhost:5173`
    - Backend: `http://localhost:5001`
    - Data Engine: `http://localhost:8000`

## 📝 Usage Guide

1.  **Load Data**: Upload a file or connect to a database.
2.  **Explore**: Check the `EDA` tab for automated insights.
3.  **Clean**: Use the `Clean` tab to fix data quality issues.
4.  **Dashboards**: Go to `Dashboard` -> `Custom Dashboard` to build your views.
5.  **Analyze**: Use `Key Drivers` in EDA to find correlations.
6.  **Train**: Go to `Model` to train and download ML models.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---
