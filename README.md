<div align="center">
  <h1>🚀 NexusDash</h1>
  <p><strong>Enterprise Data Intelligence & Machine Learning Platform</strong></p>
  
  <p>
    <a href="https://nexusdash-theta.vercel.app">Live Frontend (Vercel)</a> •
    <a href="https://nexusdash-4.onrender.com">Live Backend (Render)</a>
  </p>
</div>

---

NexusDash is a comprehensive data intelligence platform designed for modern enterprises. It allows you to seamlessly connect, clean, visualize, and analyze your data using advanced AI and Machine Learning capabilities right from your browser.

## 🌟 Key Features

### 1. Data Connectivity & Management
- **Universal Connectors**: Upload CSV/Excel formats with robust encoding support or connect to external data sources.
- **Observability**: Immediate insights into your dataset with automatic summary statistics.
- **PII Protection**: Auto-detection and masking of sensitive information (Emails, SSNs, Credit Cards) during ingestion.

### 2. Smart Data Quality
- **Auto-Cleaning**: Drop missing values, fill with median/mode, remove duplicates, and handle outliers with one click.
- **Data Quality Score**: Real-time scoring based on completeness, uniqueness, and consistency.
- **Snapshot History**: Track changes with full "Time Travel" undo/redo capabilities and version diffing.

### 3. Advanced Exploratory Data Analysis (EDA)
- **Automated EDA**: Instant distribution plots, correlation heatmaps, and statistical summaries.
- **Key Drivers Analysis**: Automatically identify which factors most influence your target metrics using ensemble trees.
- **Custom Dashboards**: Build and persist your own dashboard layouts with interactive charts.
- **AI Analytics**: Natural language queries via LLM integration to ask questions about your data in plain English.

### 4. Machine Learning Studio
- **Model Training**: Easily train Regression (RandomForestRegressor) or Classification (RandomForestClassifier) models.
- **Real-time Predictions**: Make live inferences on your trained models directly through the UI.
- **Model Export**: Download the best-performing model as a `.pkl` file including its metadata for production deployment.

## 🛠 Tech Stack

NexusDash is built using a modern, decoupled but unified architecture:

- **Frontend (The Interface)**: 
  - React.js (Vite) + TailwindCSS
  - Recharts for dynamic charting + Framer Motion
- **Backend (The Proxy Router)**: 
  - Node.js + Express.js
  - Manages file uploads, external APIs, and proxies heavy lifting.
- **Data Engine (The Brain)**: 
  - Python 3.11+ + FastAPI
  - Pandas, NumPy, Scikit-Learn for heavy data processing and model training.

## 🚀 Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/om-73/nexusdash.git
   cd nexusdash
   ```

2. **Prerequisites**
   - Node.js (v18+)
   - Python (v3.9+)

3. **Start the Development Environment**
   We provide a handy script to start all services natively:
   ```bash
   ./start-dev.sh
   ```
   This will install dependencies and launch:
   - Frontend Server at `http://localhost:5173`
   - Node.js Backend at `http://localhost:5001`
   - Python Data Engine at `http://localhost:8000`

## ☁️ Deployment

NexusDash utilizes a hybrid deployment model:

- **Frontend**: Deployed seamlessly on **Vercel** with automatic Vite builds.
- **Backend Services**: Hosted on **Render** using a unified Docker environment. The Node.js Express server runs and programmatically spawns the FastAPI Python engine (`SPAWN_PYTHON=true`). Reverse proxying allows uploading files to Node and routing processing to the internal Python engine transparently.

## 👤 Author

**Omprakash Singh**  
*Full Stack Developer*

- **GitHub**: [@om-73](https://github.com/om-73)
- **LinkedIn**: [Omprakash Singh](https://linkedin.com/in/omprakash-singh-265796228)
- **Portfolio**: [folio-mu-flax.vercel.app](https://folio-mu-flax.vercel.app)

Omprakash is a Full Stack Developer with expertise in distributed systems and microservices. He is a Meta Hacker Cup 2025 Qualifier (Top 0.5% globally) and a Smart India Hackathon finalist.

---
<div align="center">
  <i>Licensed under the MIT License</i>
</div>
