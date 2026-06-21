const axios = require('axios');
const path = require('path');
const gemini = require('../utils/gemini');

// Helper to communicate with Python Engine
const pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://127.0.0.1:8000';

exports.processUpload = async (req, res) => {
    try {
        const file_path = req.body.file_path || req.body.filePath;
        const file_type = req.body.file_type || req.body.fileType || 'csv';

        console.log('[Info] processUpload called with:', { file_path, file_type });

        if (!file_path) {
            console.error('[Error] Missing file_path in request body:', req.body);
            return res.status(400).json({ error: 'File path is required' });
        }

        // Verify file exists before sending to Python
        const fs = require('fs');
        if (!fs.existsSync(file_path)) {
            console.error('[Error] File does not exist at path:', file_path);
            return res.status(400).json({ error: `File not found at path: ${file_path}` });
        }

        console.log('[Info] File exists, sending to Python engine at:', pythonEngineUrl);

        // Call Python Engine
        const response = await axios.post(`${pythonEngineUrl}/load`, {
            file_path: file_path,
            file_type: file_type || 'csv'
        });

        console.log('[Debug] Data Engine responded with status:', response.status);
        if (response.data) {
            console.log('[Debug] Data Engine response keys:', Object.keys(response.data));
            console.log('[Debug] Data Preview rows:', response.data.preview ? response.data.preview.length : 0);
        }

        res.json(response.data);
    } catch (error) {
        console.error('[Fatal Error] processUpload failed:', error.message);
        if (error.stack) console.error(error.stack);

        if (error.response) {
            console.error('[Debug] Error from Data Engine:', error.response.status, JSON.stringify(error.response.data));
            return res.status(error.response.status).json(error.response.data);
        }

        res.status(500).json({
            error: 'Failed to process data',
            message: error.message,
            code: error.code,
            url: error.config?.url,
            dataEngineError: error.response?.data,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
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

exports.calculateKPI = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/kpi/calculate`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error calculating KPI:', error.message);
        if (error.response) {
            // Forward the upstream error details (e.g. 400 from Python)
            res.status(error.response.status).json(error.response.data);
        } else {
            res.status(500).json({ error: 'Failed to calculate KPI' });
        }
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
        const { messages, systemInstruction } = req.body;
        if (!messages || !Array.isArray(messages)) {
            return res.status(400).json({ error: 'messages array is required' });
        }
        
        const reply = await gemini.chat(messages, {
            systemInstruction: systemInstruction || "You are Nexus AI, a helpful enterprise data intelligence assistant for the NexusDash platform. Help the user analyze their datasets, understand metrics, and build machine learning workflows."
        });
        
        res.json({ reply });
    } catch (error) {
        console.error('Error in aiChat:', error);
        if (process.env.NODE_ENV === 'production') {
            return res.json({ reply: "I'm sorry, I encountered an issue connecting to my brain. Please verify that your Gemini API key is configured correctly in the environment." });
        }
        res.status(500).json({ error: 'Failed to process AI chat', details: error.message });
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

exports.getModelMetadata = async (req, res) => {
    try {
        const response = await axios.get(`${pythonEngineUrl}/model/metadata`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching model metadata:', error.message);
        if (error.response) {
            return res.status(error.response.status).json(error.response.data);
        }
        res.status(500).json({ error: 'Failed to fetch model metadata' });
    }
};

exports.predictModel = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/model/predict`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error predicting model:', error.message);
        if (error.response) {
            return res.status(error.response.status).json(error.response.data);
        }
        res.status(500).json({ error: 'Failed to predict model' });
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

exports.engineerFeature = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/feature/engineer`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error applying feature engineering:', error.message);
        if (error.response) {
            return res.status(error.response.status).json(error.response.data);
        }
        res.status(500).json({ error: 'Failed to apply feature engineering' });
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
