import axios from 'axios';

const isLocalHostOrNetwork = typeof window !== 'undefined' && (
    window.location.hostname === 'localhost' || 
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname === '0.0.0.0' ||
    window.location.hostname.endsWith('.local') ||
    /^192\.168\.\d+\.\d+$/.test(window.location.hostname) ||
    /^10\.\d+\.\d+\.\d+$/.test(window.location.hostname) ||
    /^172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+$/.test(window.location.hostname)
);

const getApiBaseUrl = () => {
    if (import.meta.env.VITE_API_URL) {
        return import.meta.env.VITE_API_URL;
    }
    if (typeof window === 'undefined') {
        return 'http://127.0.0.1:5001/api';
    }
    // If local dev environment
    if (isLocalHostOrNetwork) {
        // If served directly by backend on port 5001, use relative /api
        if (window.location.port === '5001') {
            return '/api';
        }
        // If accessed through Vite dev server (e.g. port 5173), direct all API calls to port 5001
        return `${window.location.protocol}//${window.location.hostname}:5001/api`;
    }
    // In production deployment (e.g. Render / Docker single-origin)
    return '/api';
};

const API_URL = getApiBaseUrl();

export const api = axios.create({
    baseURL: API_URL,
    timeout: 120000, // 2-minute default to accommodate Render free-tier cold starts
});


// Request interceptor to attach JWT token and log requests
api.interceptors.request.use(request => {
    const token = typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null;
    if (token && !request.headers['Authorization']) {
        request.headers['Authorization'] = `Bearer ${token}`;
    }
    if (import.meta.env.DEV) {
        console.log('[API Request]', request.method.toUpperCase(), request.url);
    }
    return request;
}, error => Promise.reject(error));

// Response interceptor to log errors with informative details
api.interceptors.response.use(
    response => response,
    error => {
        if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
            console.warn('[API Timeout]', error.config?.method?.toUpperCase(), error.config?.url, error.message);
        } else {
            console.error('[API Error]', error.response?.status, error.response?.statusText, error.response?.data || error.message);
        }

        // Handle Token Expiry / Invalid Token (excluding auth endpoints to prevent reload loop)
        if (error.response?.status === 401 && !error.config?.url?.includes('/auth/')) {
            const token = localStorage.getItem('token');
            if (token) {
                console.warn("Session expired or invalid. Resetting...");
                localStorage.removeItem('token');
                window.location.reload();
            }
        }
        return Promise.reject(error);
    }
);

export const uploadFile = async (fileOrFiles) => {
    const list = (Array.isArray(fileOrFiles) || fileOrFiles instanceof FileList)
        ? Array.from(fileOrFiles)
        : [fileOrFiles];

    if (!list.length || !list[0]) {
        throw new Error('No file provided to upload');
    }

    // When uploading a single file, send field 'file' for 100% compatibility with
    // both legacy multer single('file') and modern multer any() endpoints.
    // For multiple files, send 'files' with transparent sequential fallback.
    const formData = new FormData();
    if (list.length === 1) {
        formData.append('file', list[0]);
    } else {
        list.forEach((file) => {
            formData.append('files', file);
        });
    }

    try {
        const response = await api.post('/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 180000
        });
        return response.data;
    } catch (err) {
        // Fallback for multi-file upload if the running backend only accepts single 'file' field
        if (list.length > 1 && (err.response?.status === 400 && String(err.response?.data?.error).includes('Unexpected'))) {
            console.warn('[uploadFile] Backend expected single file field. Uploading sequentially...');
            const uploadedFiles = [];
            for (const file of list) {
                const singleFd = new FormData();
                singleFd.append('file', file);
                const res = await api.post('/upload', singleFd, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: 180000
                });
                uploadedFiles.push({
                    filePath: res.data.filePath,
                    filename: res.data.filename,
                    originalName: res.data.originalName || file.name,
                    size: res.data.size || file.size
                });
            }
            return {
                message: `${uploadedFiles.length} files uploaded successfully`,
                files: uploadedFiles,
                count: uploadedFiles.length,
                filePath: uploadedFiles[0].filePath,
                filename: uploadedFiles[0].filename,
                originalName: uploadedFiles[0].originalName,
                size: uploadedFiles[0].size
            };
        }
        throw err;
    }
};

export const uploadFiles = async (files) => {
    return uploadFile(files);
};

export const deleteFile = async (filePath) => {
    if (!filePath) return;
    try {
        const response = await api.post('/upload/cleanup', { filePath });
        return response.data;
    } catch (err) {
        console.warn('Failed to delete file:', err);
    }
};

export const cleanupSessionFileBeacon = (filePath) => {
    if (!filePath) return;
    try {
        const payload = JSON.stringify({ filePath });
        const blob = new Blob([payload], { type: 'application/json' });
        const url = `${API_URL}/upload/cleanup`;
        if (navigator.sendBeacon) {
            navigator.sendBeacon(url, blob);
        } else {
            fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload,
                keepalive: true
            }).catch(() => {});
        }
    } catch (e) {
        console.warn('Beacon cleanup error:', e);
    }
};

export const loadData = async (filePathOrPaths, fileType = 'csv', combineMethod = 'concat', joinKey = null) => {
    const payload = Array.isArray(filePathOrPaths)
        ? { file_paths: filePathOrPaths, file_type: fileType, combine_method: combineMethod, join_key: joinKey }
        : { file_path: filePathOrPaths, file_type: fileType };

    const response = await api.post('/data/load', payload, { timeout: 300000 });
    return response.data;
};

export const combineDatasets = async (filePaths, combineMethod = 'concat', joinKey = null) => {
    const response = await api.post('/data/combine', {
        file_paths: filePaths,
        combine_method: combineMethod,
        join_key: joinKey
    }, { timeout: 300000 });
    return response.data;
};

export const cleanData = async (payload) => {
    const response = await api.post('/data/clean', payload, { timeout: 300000 });
    return response.data;
};

export const getEDA = async () => {
    const response = await api.get('/data/eda', { timeout: 180000 });
    return response.data;
};

export const trainModel = async (payload) => {
    const response = await api.post('/data/train', payload, { timeout: 600000 });
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

export const chatWithAI = async (messages, systemInstruction) => {
    const response = await api.post('/data/ai/chat', { messages, systemInstruction });
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

export const getModelMetadata = async () => {
    const response = await api.get('/data/model/metadata');
    return response.data;
};

export const predictModel = async (inputs) => {
    const response = await api.post('/data/predict', { inputs });
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
    const response = await api.get('/data/state', { timeout: 120000 });
    return response.data;
};

export const getKPIs = async (req, res) => {
    const response = await api.get('/data/kpi');
    return response.data;
};

export const calculateKPI = async (column, operation) => {
    const response = await api.post('/data/kpi/calculate', { column, operation });
    return response.data;
};

export const addFeature = async (name, expression) => {
    const response = await api.post('/feature/add', { name, expression });
    return response.data;
};

export const getQuantile = async (column, q = 0.95) => {
    const response = await api.get('/data/quantile', { params: { column, q } });
    return response.data;
};

export const runAutonomousAgent = async (params = {}) => {
    try {
        const response = await api.post('/agent/run', params, { timeout: 600000 });
        return response.data;
    } catch (err) {
        if (err.response?.status === 404) {
            console.log('[API Fallback] Retrying via /data/agent/run...');
            const fallback = await api.post('/data/agent/run', params, { timeout: 600000 });
            return fallback.data;
        }
        throw err;
    }
};

export const loadAgentSample = async (datasetName, autoRun = true, goal = null) => {
    try {
        const response = await api.post('/agent/sample', { dataset_name: datasetName, auto_run: autoRun, goal }, { timeout: 600000 });
        return response.data;
    } catch (err) {
        if (err.response?.status === 404) {
            console.log('[API Fallback] Retrying via /data/agent/sample...');
            const fallback = await api.post('/data/agent/sample', { dataset_name: datasetName, auto_run: autoRun, goal }, { timeout: 600000 });
            return fallback.data;
        }
        throw err;
    }
};

export const getAgentPlan = async (params = {}) => {
    try {
        const response = await api.post('/agent/plan', params, { timeout: 60000 });
        return response.data;
    } catch (err) {
        if (err.response?.status === 404) {
            console.log('[API Fallback] Retrying via /data/agent/plan...');
            const fallback = await api.post('/data/agent/plan', params, { timeout: 60000 });
            return fallback.data;
        }
        throw err;
    }
};


