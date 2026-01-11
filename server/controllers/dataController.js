const axios = require('axios');
const path = require('path');

// Helper to communicate with Python Engine
const pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';

exports.processUpload = async (req, res) => {
    try {
        const { filePath, fileType } = req.body; // Expects absolute path or relative from upload handling

        if (!filePath) {
            return res.status(400).json({ error: 'File path is required' });
        }

        // Call Python Engine
        const response = await axios.post(`${pythonEngineUrl}/load`, {
            file_path: filePath,
            file_type: fileType || 'csv'
        });

        res.json(response.data);
    } catch (error) {
        console.error('Error connecting to Data Engine:', error.message);
        if (error.response) {
            console.error('Data Engine Response:', JSON.stringify(error.response.data, null, 2));
            return res.status(error.response.status).json(error.response.data);
        }
        res.status(500).json({ error: 'Failed to process data', details: error.message });
    }
};

exports.getState = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/state`);
        res.json(response.data);
    } catch (error) {
        // If 404 or connection error, just return null (no state)
        res.json(null);
    }
};

exports.cleanData = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/clean`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error cleaning data:', error.message);
        if (error.response) {
            console.error('Data Engine Clean Error:', JSON.stringify(error.response.data, null, 2));
            return res.status(error.response.status).json(error.response.data);
        }
        res.status(500).json({ error: 'Failed to clean data', details: error.message });
    }
};

exports.getEDA = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/eda`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching EDA:', error.message);
        if (error.response) {
            return res.status(error.response.status).json(error.response.data);
        }
        res.status(500).json({ error: 'Failed to fetch EDA' });
    }
};

exports.trainModel = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/train`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error training model:', error.message);
        if (error.response) {
            return res.status(error.response.status).json(error.response.data);
        }
        res.status(500).json({ error: 'Failed to train model' });
    }
};

exports.exportData = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/export`, { responseType: 'stream' });
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', 'attachment; filename=cleaned_data.csv');
        response.data.pipe(res);
    } catch (error) {
        console.error('Error exporting data:', error.message);
        res.status(500).json({ error: 'Failed to export data' });
    }
};

exports.connectDatabase = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/connect_db`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error connecting to database:', error.message);
        const errorMessage = error.response?.data?.detail || error.message || 'Failed to connect to database';
        res.status(500).json({ error: errorMessage });
    }
};

exports.undoAction = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/undo`);
        res.json(response.data);
    } catch (error) {
        // It's okay if undo fails (e.g., stack empty), just return error message
        res.status(400).json({ error: error.response?.data?.detail || 'Nothing to undo' });
    }
};

exports.redoAction = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/redo`);
        res.json(response.data);
    } catch (error) {
        res.status(400).json({ error: error.response?.data?.detail || 'Nothing to redo' });
    }
};

exports.getDataQuality = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/quality`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching quality score:', error.message);
        res.status(500).json({ error: 'Failed to fetch quality score' });
    }
};

exports.getPipeline = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/pipeline`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch pipeline' });
    }
};

exports.getEDASummary = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/eda/summary`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching EDA summary:', error.message);
        res.status(500).json({ error: 'Failed to fetch EDA summary' });
    }
};

exports.getChartRecommendations = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/recommend_charts`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching chart recommendations:', error.message);
        res.status(500).json({ error: 'Failed to fetch chart recommendations' });
    }
};

exports.getKPIs = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/kpi`);
        res.json(response.data); // Returns list of objects
    } catch (error) {
        console.error('Error fetching KPIs:', error.message);
        res.status(500).json({ error: 'Failed to fetch KPIs' });
    }
};

exports.queryMachineLearning = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/query_ml`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error querying ML engine:', error.message);
        res.status(500).json({ error: 'Failed to process query' });
    }
};

exports.registerDataset = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/register_dataset`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to register dataset' });
    }
};

exports.checkHealth = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/check_health?dataset_id=${req.params.id}`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to check health' });
    }
};

exports.saveContract = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/contracts`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to save contract' });
    }
};

exports.validateContract = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/validate_contract`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to validate contract' });
    }
};

exports.scanPII = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/scan_pii`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to scan for PII' });
    }
};

exports.maskData = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/mask_data`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to mask data' });
    }
};

exports.aiChat = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/ai/chat`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to process AI chat' });
    }
};

exports.configureAI = async (req, res) => {
    try {
        // req.body should be { goal: "predict price" }
        const response = await axios.post(`${pythonEngineUrl}/ai/configure`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error("AI Config Error:", error.response?.data || error.message);
        res.status(500).json({ error: error.response?.data?.detail || 'Failed to configure AI' });
    }
};

exports.analyzeDrivers = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/analyze/drivers`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to analyze drivers' });
    }
};

exports.downloadModel = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/model/download`, { responseType: 'stream' });
        res.setHeader('Content-Disposition', 'attachment; filename=model.pkl');
        res.setHeader('Content-Type', 'application/octet-stream');
        response.data.pipe(res);
    } catch (error) {
        res.status(500).json({ error: 'Failed to download model' });
    }
};

exports.queryBuilder = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/query/builder`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to execute query' });
    }
};

exports.registerFeature = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/features/register`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to register feature' });
    }
};

exports.getFeatures = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/features`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch features' });
    }
};

exports.executeNotebook = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/notebook/execute`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to execute notebook code' });
    }
};

exports.getSnapshotHistory = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/history/snapshots/${req.params.id}`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch snapshot history' });
    }
};

exports.revertToSnapshot = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/history/revert`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to revert to snapshot' });
    }
};

exports.diffSnapshots = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/history/diff`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to diff snapshots' });
    }
};

exports.saveDashboard = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/dashboards/save`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to save dashboard' });
    }
};

exports.listDashboards = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/dashboards`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to list dashboards' });
    }
};

exports.getDashboard = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/dashboards/${req.params.id}`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to get dashboard' });
    }
};
