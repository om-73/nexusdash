const axios = require('axios');
const gemini = require('../utils/gemini');

const pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://127.0.0.1:8000';

exports.runAgent = async (req, res) => {
    try {
        console.log('[Agent Controller] Executing Autonomous Agent...', req.body);
        const response = await axios.post(`${pythonEngineUrl}/agent/run`, req.body, { timeout: 60000 });
        const agentData = response.data;

        // Optionally enrich executive brief using Gemini if API key is present
        try {
            if (process.env.GEMINI_API_KEY || process.env.GROQ_API_KEY) {
                const prompt = `You are Nexus Autonomous AI Agent. Summarize the following AutoML and data intelligence run for an executive dashboard in 2 punchy bullet points:
Champion Model: ${agentData.champion_model?.name} (${agentData.champion_model?.metric_name}: ${agentData.champion_model?.metric_value})
Target Column: ${agentData.target_column} (${agentData.problem_type})
Quality Score: ${agentData.data_diff?.before?.quality_score}% -> ${agentData.data_diff?.after?.quality_score}%
Top Drivers: ${JSON.stringify(agentData.feature_importance?.slice(0, 3))}
Keep it under 50 words.`;
                const llmSummary = await gemini.chat([{ role: 'user', content: prompt }]);
                if (llmSummary && !llmSummary.startsWith('⚠️')) {
                    agentData.llm_takeaways = llmSummary;
                }
            }
        } catch (llmErr) {
            console.warn('[Agent Controller] LLM enrichment skipped:', llmErr.message);
        }

        res.json(agentData);
    } catch (error) {
        console.error('[Agent Controller] Error executing agent:', error.response?.data || error.message);
        const status = error.response?.status || 500;
        const detail = error.response?.data?.detail || error.message || 'Failed to execute autonomous agent';
        res.status(status).json({ error: detail });
    }
};

exports.loadSample = async (req, res) => {
    try {
        console.log('[Agent Controller] Loading sample dataset and running agent:', req.body);
        const response = await axios.post(`${pythonEngineUrl}/agent/sample`, req.body, { timeout: 60000 });
        res.json(response.data);
    } catch (error) {
        console.error('[Agent Controller] Error loading sample:', error.response?.data || error.message);
        const status = error.response?.status || 500;
        const detail = error.response?.data?.detail || error.message || 'Failed to load sample dataset';
        res.status(status).json({ error: detail });
    }
};

exports.getPlan = async (req, res) => {
    try {
        const response = await axios.post(`${pythonEngineUrl}/agent/plan`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('[Agent Controller] Error getting plan:', error.message);
        res.status(500).json({ error: 'Failed to generate agent plan' });
    }
};
