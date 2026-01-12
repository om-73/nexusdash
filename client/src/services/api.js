import axios from 'axios';

// reliable runtime check for Vercel
const isVercel = window.location.hostname.includes('vercel.app');
const PROD_URL = 'https://nexusdash-2.onrender.com/api';
const API_URL = isVercel ? PROD_URL : (import.meta.env.VITE_API_URL || 'http://127.0.0.1:5001/api');

export const api = axios.create({
    baseURL: API_URL,
});

export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export const loadData = async (filePath, fileType = 'csv') => {
    const response = await api.post('/data/load', {
        filePath,
        fileType
    });
    return response.data;
};

export const cleanData = async (payload) => {
    const response = await api.post('/data/clean', payload);
    return response.data;
};

export const getEDA = async () => {
    const response = await api.get('/data/eda');
    return response.data;
};

export const trainModel = async (payload) => {
    const response = await api.post('/data/train', payload);
    return response.data;
};

export const exportData = () => {
    // For download, we can simply redirect or use window.open since it's a GET request
    window.location.href = `${API_URL}/data/export`;
};

export const connectDatabase = async (payload) => {
    const response = await api.post('/data/connect', payload);
    return response.data;
};

export const undoAction = async () => {
    const response = await api.post('/data/undo');
    return response.data;
};

export const redoAction = async () => {
    const response = await api.post('/data/redo');
    return response.data;
};

export const getDataQuality = async () => {
    const response = await api.get('/data/quality');
    return response.data;
};

export const getPipeline = async () => {
    const response = await api.get('/data/pipeline');
    return response.data;
};

export const getEDASummary = async () => {
    const response = await api.get('/data/eda-summary');
    return response.data;
};

export const getChartRecommendations = async () => {
    const response = await api.get('/data/recommend');
    return response.data;
};

export const queryMachineLearning = async (query) => {
    const response = await api.post('/data/query', { query });
    return response.data;
};

export const configureAI = async (goal) => {
    const response = await api.post('/data/ai/configure', { goal });
    return response.data;
};

export const analyzeDrivers = async (targetColumn) => {
    const response = await api.post('/data/analyze-drivers', { target_column: targetColumn });
    return response.data;
};

export const downloadModel = async () => {
    const response = await api.get('/data/model/download', { responseType: 'blob' });
    return response.data;
};



export const saveDashboard = async (name, layout) => {
    const response = await api.post('/data/dashboards/save', { name, layout });
    return response.data;
};

export const listDashboards = async () => {
    const response = await api.get('/data/dashboards');
    return response.data;
};

export const getDashboard = async (id) => {
    const response = await api.get(`/data/dashboards/${id}`);
    return response.data;
};

export const getState = async () => {
    const response = await api.get('/data/state');
    return response.data;
};

export const getKPIs = async () => {
    const response = await api.get('/data/kpi');
    return response.data;
};
