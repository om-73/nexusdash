<div align="center">
  <h1>🚀 NexusDash</h1>
  <p><strong>Enterprise Data Intelligence, Autonomous AutoML & Observability Platform</strong></p>
  
  <p>
    <a href="https://nexusdash-theta.vercel.app">Live Frontend (Vercel)</a> •
    <a href="https://nexusdash-4.onrender.com">Live Backend (Render)</a> •
    <a href="PROJECT_DOCUMENTATION.md">Full Architecture & Engineering Manual</a>
  </p>
</div>

---

NexusDash is a comprehensive, production-grade data intelligence and automated machine learning (AutoML) platform designed for modern enterprises. It enables data scientists, business analysts, and engineers to connect, clean, fuse, visualize, and autonomously train predictive models across disparate data sources with zero manual boilerplate.

> 📖 **Comprehensive Documentation**: For in-depth system architecture, source-code traceability, ATS resume bullets, and technical interview Q&A, refer to [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

---

## 🌟 Key Features

### 1. Autonomous AI Agent (AutoML Core v2.5)
- **1-Click Autonomous Execution**: Automated end-to-end data pipeline:
  $$\text{Deep Profiling} \longrightarrow \text{Self-Healing Cleaning} \longrightarrow \text{Feature Synthesis} \longrightarrow \text{Tournament Arena} \longrightarrow \text{Driver Attribution} \longrightarrow \text{Executive Dashboard}$$
- **AutoML Tournament Arena**: Rapidly benchmarks candidate algorithms (`RandomForest`, `GradientBoosting`, `LogisticRegression` / `Ridge`, `DecisionTree`) using an intelligent 5,000-row tournament subsampling heuristic to evaluate models in under **4 seconds**.
- **Real-Time Mission Control Stream**: Live telemetry streaming showing agent thoughts, executed tools, data quality diffs, and metric comparisons.
- **Autonomous Dashboard Blueprint Synthesis**: The AI Agent automatically synthesizes and persists an 8-widget custom dashboard (4 KPIs, 4 analytical charts) configured specifically for the target metric.

### 2. Intelligent Multi-Dataset Schema Fusion
- **Relational Key Join (`merge`)**: Automatically identifies shared primary/foreign keys across datasets and applies an outer join with suffix collision resolution.
- **Horizontal Padded Feature Merge (`columns`)**: Merges disjoint datasets side-by-side with automatic cyclic modulo padding (`(max_len // len(d)) + 1`), preserving 100% of data values with zero null-row gaps.
- **Vertical Row Union (`concat`)**: Stacks identically structured datasets seamlessly.

### 3. Custom & Adaptive 12-Column Dashboards
- **12-Column Responsive Grid**: Draggable and resizable widgets powered by `react-grid-layout` and `recharts`.
- **Zero-Height Collapse Fix**: Enforces `min-h-[170px]` and decoupled flex KPI widgets to guarantee 100% SVG render reliability.
- **Dynamic Dataset Adaptation**: Dashboards dynamically hydrate and adapt aggregations (`sum`, `mean`, `count`, `min`, `max`) when switching between active files or database connections.

### 4. Enterprise Database Connectivity & Live Sync
- **Native SQL & NoSQL Support**: Direct connections to PostgreSQL, MySQL, SQLite, and MongoDB via SQLAlchemy and PyMongo.
- **Live Query Execution**: Execute custom SQL statements or MongoDB collection queries with immediate DataFrame mapping and automatic metadata registration.

### 5. Smart Data Quality & Parquet Time-Travel
- **Self-Healing Hygiene**: Automatic median/mode imputation, IQR-based extreme outlier capping ($1.5 \times \text{IQR}$), deduplication, and target leakage stripping.
- **Parquet-Backed Snapshot History**: Fast columnar micro-snapshots with SHA-256 schema hashing for instant rollback and time-travel versioning.
- **PII Compliance Scanner**: Scans and masks sensitive Personally Identifiable Information (Emails, SSNs, Credit Cards) with one click.

---

## 🛠 Tech Stack

```
nexusdash-
├── client/           # React 19 + Vite 7 + Tailwind CSS v4 + Recharts 3
├── server/           # Node.js 18+ + Express 5 + Multer + Axios
└── data_engine/      # Python 3.11 + FastAPI + Pandas + Scikit-Learn + PyArrow + SQLite
```

- **Frontend**: React 19, Vite 7, Tailwind CSS v4, Recharts, React-Grid-Layout, Lucide Icons.
- **Backend Gateway**: Node.js 18+, Express 5, Multer (100MB streaming upload), JWT Authentication, Bcrypt, Gzip Compression.
- **Data Engine (Brain)**: Python 3.11+, FastAPI, Pandas, NumPy, Scikit-Learn, PyArrow, SQLite (`metadata.db`), SQLAlchemy, PyMongo.
- **AI / LLM Integrations**: Google Gemini 2.5 API (`@google/genai`) and Groq Cloud (LLaMA 3) with 6-second race-condition timeout fallbacks.

---

## 🚀 Local Development Setup

### Prerequisites
- **Node.js** (v18.0.0 or higher)
- **Python** (v3.9 or higher, recommended v3.11)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/om-73/nexusdash.git
cd nexusdash
```

### 2. Launch Development Environment
We provide an automated multi-process startup script:
```bash
chmod +x start-dev.sh
./start-dev.sh
```

This installs dependencies and launches:
- **Frontend SPA**: `http://localhost:5173`
- **Node.js Gateway**: `http://localhost:5001`
- **FastAPI Data Engine**: `http://localhost:8000`

---

## ☁️ Deployment Architecture

NexusDash employs a hybrid production deployment:
- **Frontend**: Hosted on **Vercel** with Vite production optimizations and route rewrites (`vercel.json`).
- **Backend Services**: Hosted on **Render** via a unified `Dockerfile`. Node.js acts as the master process supervisor, automatically spawning the FastAPI Python engine (`SPAWN_PYTHON=true`) and reverse-proxying requests internally over `http://127.0.0.1:8000`.

---

## 👤 Author & Architecture

**Omprakash Singh**  
*Full Stack Developer & Distributed Systems Engineer*

- **GitHub**: [@om-73](https://github.com/om-73)
- **LinkedIn**: [Omprakash Singh](https://linkedin.com/in/omprakash-singh-265796228)
- **Portfolio**: [folio-mu-flax.vercel.app](https://folio-mu-flax.vercel.app)

*Meta Hacker Cup 2025 Qualifier (Top 0.5% globally) • Smart India Hackathon Finalist.*

---

<div align="center">
  <i>Licensed under the MIT License</i>
</div>
