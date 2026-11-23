import React, { createContext, useContext, useState, useEffect } from 'react';
import { getState, deleteFile, cleanupSessionFileBeacon, loadData, combineDatasets, connectDatabase } from '../services/api';
import { useAuth } from './AuthContext';

const DataContext = createContext();

const getInitialSummary = () => {
    try {
        const cached = sessionStorage.getItem('nexus_data_summary');
        return cached ? JSON.parse(cached) : null;
    } catch (e) {
        return null;
    }
};

const getInitialPreview = () => {
    try {
        const cached = sessionStorage.getItem('nexus_data_preview');
        return cached ? JSON.parse(cached) : [];
    } catch (e) {
        return [];
    }
};

const getInitialSessionObj = (key) => {
    try {
        const cached = sessionStorage.getItem(key);
        return cached ? JSON.parse(cached) : null;
    } catch (e) {
        return null;
    }
};

export const DataProvider = ({ children }) => {
    const { isAuthenticated } = useAuth();
    const [dataSummary, setDataSummaryState] = useState(getInitialSummary);
    const [dataPreview, setDataPreviewState] = useState(getInitialPreview);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Multi-dataset state
    const [uploadedDatasets, setUploadedDatasetsState] = useState(() => getInitialSessionObj('nexus_uploaded_datasets') || []);
    const [activeDatasetId, setActiveDatasetIdState] = useState(() => sessionStorage.getItem('nexus_active_dataset_id') || null);
    const [isUnifiedMode, setIsUnifiedModeState] = useState(() => sessionStorage.getItem('nexus_is_unified_mode') === 'true');

    const setIsUnifiedMode = (val) => {
        setIsUnifiedModeState(val);
        try {
            sessionStorage.setItem('nexus_is_unified_mode', val ? 'true' : 'false');
        } catch (e) {}
    };

    const setUploadedDatasets = (datasets) => {
        setUploadedDatasetsState(datasets);
        try {
            if (datasets && datasets.length > 0) {
                sessionStorage.setItem('nexus_uploaded_datasets', JSON.stringify(datasets));
            } else {
                sessionStorage.removeItem('nexus_uploaded_datasets');
            }
        } catch (e) {}
    };

    const setActiveDatasetId = (id) => {
        setActiveDatasetIdState(id);
        try {
            if (id) sessionStorage.setItem('nexus_active_dataset_id', id);
            else sessionStorage.removeItem('nexus_active_dataset_id');
        } catch (e) {}
    };

    // Helper: Register newly uploaded files into datasets list
    const registerUploadedDatasets = (newFiles, activeSummary = null, activePreview = null) => {
        setUploadedDatasetsState(prev => {
            const list = [...prev];
            newFiles.forEach((file, index) => {
                const id = file.filename || file.filePath || `dataset-${Date.now()}-${index}`;
                const ext = (file.originalName || file.name || '').split('.').pop().toLowerCase();
                const fileType = (ext === 'xlsx' || ext === 'xls') ? 'excel' : (ext === 'zip' ? 'zip' : 'csv');

                const existingIdx = list.findIndex(d => d.filePath === file.filePath || d.name === (file.originalName || file.name));
                const entry = {
                    id,
                    name: file.originalName || file.name || 'Dataset',
                    filePath: file.filePath,
                    fileType,
                    size: file.size || 0,
                    uploadedAt: new Date().toISOString(),
                    shape: (index === 0 && activeSummary?.shape) ? activeSummary.shape : null,
                    summary: (index === 0 && activeSummary) ? activeSummary : null,
                    isActive: index === 0
                };

                if (existingIdx >= 0) {
                    list[existingIdx] = { ...list[existingIdx], ...entry };
                } else {
                    list.push(entry);
                }
            });

            if (newFiles.length > 0) {
                const primaryId = newFiles[0].filename || newFiles[0].filePath;
                setActiveDatasetId(primaryId);
                list.forEach(d => {
                    d.isActive = (d.id === primaryId);
                });
            }

            try {
                sessionStorage.setItem('nexus_uploaded_datasets', JSON.stringify(list));
            } catch (e) {}
            return list;
        });
    };

    // Helper: Switch active dataset
    const switchActiveDataset = async (datasetId) => {
        const dataset = uploadedDatasets.find(d => d.id === datasetId);
        if (!dataset) return;

        setLoading(true);
        setError(null);
        try {
            let dataRes;
            if (dataset.isDatabase && dataset.dbConfig) {
                dataRes = await connectDatabase(dataset.dbConfig);
                setActiveDatabaseConfig(dataset.dbConfig);
            } else {
                dataRes = await loadData(dataset.filePath, dataset.fileType);
            }
            setDataSummary(dataRes);
            setDataPreview(dataRes.preview || []);
            setActiveDatasetId(datasetId);

            const updated = uploadedDatasets.map(d => ({
                ...d,
                isActive: d.id === datasetId,
                shape: d.id === datasetId ? (dataRes.shape || d.shape) : d.shape,
                summary: d.id === datasetId ? dataRes : d.summary
            }));
            setUploadedDatasets(updated);

            if (dataset.filePath) {
                sessionStorage.setItem('active_session_file', dataset.filePath);
            }
            return dataRes;
        } catch (err) {
            console.error("Failed to switch dataset:", err);
            const msg = err.response?.data?.message || err.response?.data?.error || err.message || "Failed to switch active dataset";
            setError(msg);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    // Helper: Combine multiple datasets into one unified dataset
    const combineDatasetsAction = async (datasetIds, combineMethod = 'auto', joinKey = null) => {
        const selected = uploadedDatasets.filter(d => datasetIds.includes(d.id));
        if (selected.length < 2) {
            throw new Error("Select at least 2 datasets to combine");
        }

        const filePaths = selected.map(d => d.filePath);
        setLoading(true);
        setError(null);
        try {
            const dataRes = await combineDatasets(filePaths, combineMethod, joinKey);
            setDataSummary(dataRes);
            setDataPreview(dataRes.preview || []);

            const combinedName = combineMethod === 'merge' 
                ? `Joined (${selected.map(s => s.name.replace(/\.[^/.]+$/, "")).join(" ⋈ ")})`
                : combineMethod === 'columns'
                ? `Merged Features (${selected.map(s => s.name.replace(/\.[^/.]+$/, "")).join(" ∥ ")})`
                : combineMethod === 'concat'
                ? `Merged Rows (${selected.map(s => s.name.replace(/\.[^/.]+$/, "")).join(" + ")})`
                : `Combined Master (${selected.map(s => s.name.replace(/\.[^/.]+$/, "")).join(" & ")})`;

            const combinedId = `combined-${Date.now()}`;
            const combinedEntry = {
                id: combinedId,
                name: combinedName,
                filePath: filePaths[0],
                fileType: 'csv',
                size: selected.reduce((acc, s) => acc + (s.size || 0), 0),
                uploadedAt: new Date().toISOString(),
                shape: dataRes.shape || null,
                summary: dataRes,
                isActive: true,
                isCombined: true,
                combinedFilesCount: selected.length
            };

            const updated = uploadedDatasets.map(d => ({ ...d, isActive: false }));
            updated.unshift(combinedEntry);
            setUploadedDatasets(updated);
            setActiveDatasetId(combinedId);

            return dataRes;
        } catch (err) {
            console.error("Failed to combine datasets:", err);
            const msg = err.response?.data?.message || err.response?.data?.error || err.message || "Failed to combine datasets";
            setError(msg);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    // Helper: 1-Click activate All Files Together (Unified Mode)
    const useAllFilesTogether = async (combineMethod = 'auto', joinKey = null) => {
        const rawDatasets = uploadedDatasets.filter(d => !d.isCombined);
        if (rawDatasets.length === 0) {
            throw new Error("No uploaded datasets available to combine");
        }
        if (rawDatasets.length === 1) {
            setIsUnifiedMode(false);
            return switchActiveDataset(rawDatasets[0].id);
        }

        const res = await combineDatasetsAction(rawDatasets.map(d => d.id), combineMethod, joinKey);
        setIsUnifiedMode(true);
        return res;
    };

    // Helper: Use a specific file separately
    const useFileSeparately = async (datasetId) => {
        setIsUnifiedMode(false);
        return switchActiveDataset(datasetId);
    };

    // Helper: Remove dataset
    const removeUploadedDataset = async (datasetId) => {
        const dataset = uploadedDatasets.find(d => d.id === datasetId);
        if (!dataset) return;

        if (dataset.filePath && !dataset.isCombined) {
            deleteFile(dataset.filePath).catch(() => {});
        }

        const remaining = uploadedDatasets.filter(d => d.id !== datasetId);
        setUploadedDatasets(remaining);

        if (activeDatasetId === datasetId) {
            if (remaining.length > 0) {
                switchActiveDataset(remaining[0].id).catch(() => {});
            } else {
                setActiveDatasetId(null);
                setDataSummary(null);
                setDataPreview([]);
                sessionStorage.removeItem('active_session_file');
            }
        }
    };

    // Cross-page autonomous operation results
    const [cleaningResult, setCleaningResultState] = useState(() => getInitialSessionObj('nexus_cleaning_result'));
    const [featureEngineeringResult, setFeatureEngineeringResultState] = useState(() => getInitialSessionObj('nexus_fe_result'));
    const [modelResult, setModelResultState] = useState(() => getInitialSessionObj('nexus_model_result'));
    const [predictionResult, setPredictionResultState] = useState(() => getInitialSessionObj('nexus_prediction_result'));
    const [agentDashboard, setAgentDashboardState] = useState(() => getInitialSessionObj('nexus_agent_dashboard'));
    const [activeDatabaseConfig, setActiveDatabaseConfigState] = useState(() => getInitialSessionObj('nexus_active_db'));

    const setAgentDashboard = (dash) => {
        setAgentDashboardState(dash);
        try {
            if (dash) sessionStorage.setItem('nexus_agent_dashboard', JSON.stringify(dash));
            else sessionStorage.removeItem('nexus_agent_dashboard');
        } catch (e) {}
    };

    const setActiveDatabaseConfig = (config) => {
        setActiveDatabaseConfigState(config);
        try {
            if (config) sessionStorage.setItem('nexus_active_db', JSON.stringify(config));
            else sessionStorage.removeItem('nexus_active_db');
        } catch (e) {}
    };

    const refreshFromDatabase = async () => {
        if (!activeDatabaseConfig) {
            throw new Error("No active database connection configured.");
        }
        setLoading(true);
        setError(null);
        try {
            const dataRes = await connectDatabase(activeDatabaseConfig);
            setDataSummary(dataRes);
            if (dataRes.preview) setDataPreview(dataRes.preview);
            return dataRes;
        } catch (err) {
            console.error("Failed to refresh live database data:", err);
            const msg = err.response?.data?.error || err.response?.data?.detail || err.message || "Failed to refresh database";
            setError(msg);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const setCleaningResult = (res) => {
        setCleaningResultState(res);
        try {
            if (res) sessionStorage.setItem('nexus_cleaning_result', JSON.stringify(res));
            else sessionStorage.removeItem('nexus_cleaning_result');
        } catch (e) {}
    };

    const setFeatureEngineeringResult = (res) => {
        setFeatureEngineeringResultState(res);
        try {
            if (res) sessionStorage.setItem('nexus_fe_result', JSON.stringify(res));
            else sessionStorage.removeItem('nexus_fe_result');
        } catch (e) {}
    };

    const setModelResult = (res) => {
        setModelResultState(res);
        try {
            if (res) sessionStorage.setItem('nexus_model_result', JSON.stringify(res));
            else sessionStorage.removeItem('nexus_model_result');
        } catch (e) {}
    };

    const setPredictionResult = (res) => {
        setPredictionResultState(res);
        try {
            if (res) sessionStorage.setItem('nexus_prediction_result', JSON.stringify(res));
            else sessionStorage.removeItem('nexus_prediction_result');
        } catch (e) {}
    };

    // Central dispatcher: routes agent execution artifacts to their matching dedicated pages
    const syncAgentRunResults = (runData) => {
        if (!runData) return;

        // 1. Send Cleaning results to Data Clean page (/clean)
        if (runData.data_diff) {
            setCleaningResult({
                diff: runData.data_diff,
                changes: runData.data_diff.changes || [],
                quality_before: runData.data_diff.before?.quality_score,
                quality_after: runData.data_diff.after?.quality_score,
                missing_resolved: runData.data_diff.before?.missing_cells,
                duplicates_removed: runData.data_diff.before?.duplicate_rows,
                columns_before: runData.data_diff.before?.columns,
                columns_after: runData.data_diff.after?.columns,
                timestamp: new Date().toISOString()
            });
        }

        // 2. Send Feature Engineering results to Feature Engineering page (/features)
        if (runData.feature_engineering || runData.feature_importance) {
            setFeatureEngineeringResult({
                transformations: runData.feature_engineering?.transformations || [],
                total_synthetic: runData.feature_engineering?.total_synthetic || 0,
                explanation: runData.feature_engineering?.explanation || '',
                feature_importance: runData.feature_importance || [],
                timestamp: new Date().toISOString()
            });
        }

        // 3. Send Champion Model & Tournament results to Model Training page (/model)
        if (runData.champion_model || runData.tournament) {
            setModelResult({
                champion: runData.champion_model,
                tournament: runData.tournament || [],
                target_column: runData.target_column,
                problem_type: runData.problem_type,
                feature_importance: runData.feature_importance || [],
                timestamp: new Date().toISOString()
            });
        }

        // 4. Send Prediction inputs & Model Metadata to Prediction page (/predict)
        if (runData.champion_model || runData.sample_inputs) {
            setPredictionResult({
                model_name: runData.champion_model?.name,
                target_column: runData.target_column,
                problem_type: runData.problem_type,
                sample_inputs: runData.sample_inputs || {},
                metric_name: runData.champion_model?.metric_name,
                metric_value: runData.champion_model?.metric_value,
                prediction: null,
                timestamp: new Date().toISOString()
            });
        }

        // 5. Send Custom Dashboard configuration to Custom Dashboard page (/dashboard/custom)
        if (runData.custom_dashboard) {
            setAgentDashboard(runData.custom_dashboard);
        }
    };

    const setDataSummary = (data) => {
        setDataSummaryState(data);
        try {
            if (data) {
                sessionStorage.setItem('nexus_data_summary', JSON.stringify(data));
            } else {
                sessionStorage.removeItem('nexus_data_summary');
            }
        } catch (e) {}
    };

    const setDataPreview = (preview) => {
        setDataPreviewState(preview);
        try {
            if (preview && preview.length > 0) {
                sessionStorage.setItem('nexus_data_preview', JSON.stringify(preview));
            } else {
                sessionStorage.removeItem('nexus_data_preview');
            }
        } catch (e) {}
    };

    // Automatic cleanup of uploaded CSV when browser session ends
    useEffect(() => {
        const handleSessionEnd = () => {
            const activeFile = sessionStorage.getItem('active_session_file');
            if (activeFile) {
                cleanupSessionFileBeacon(activeFile);
            }
        };

        window.addEventListener('pagehide', handleSessionEnd);
        window.addEventListener('beforeunload', handleSessionEnd);

        return () => {
            window.removeEventListener('pagehide', handleSessionEnd);
            window.removeEventListener('beforeunload', handleSessionEnd);
        };
    }, []);

    // Re-hydrate state in background on mount or login
    useEffect(() => {
        if (!isAuthenticated) {
            const activeFile = sessionStorage.getItem('active_session_file');
            if (activeFile) {
                deleteFile(activeFile).catch(() => {});
                sessionStorage.removeItem('active_session_file');
            }
            setDataSummary(null);
            setDataPreview([]);
            return;
        }

        let isMounted = true;
        const fetchState = async () => {
            try {
                const res = await getState();
                if (!isMounted) return;
                if (res) {
                    setDataSummary(res);
                    if (res.preview) setDataPreview(res.preview);
                }
            } catch (err) {
                console.warn("Failed to restore state", err);
            }
        };

        fetchState();

        return () => {
            isMounted = false;
        };
    }, [isAuthenticated]);

    const refreshData = async () => {
        setLoading(true);
        try {
            const res = await getState();
            if (res) {
                setDataSummary(res);
                if (res.preview) setDataPreview(res.preview);
            }
        } catch (err) {
            console.warn("Failed to refresh state", err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <DataContext.Provider value={{
            dataSummary, setDataSummary,
            dataPreview, setDataPreview,
            loading, setLoading,
            error, setError,
            refreshData,
            // Multi-dataset management
            uploadedDatasets, setUploadedDatasets,
            activeDatasetId, setActiveDatasetId,
            isUnifiedMode, setIsUnifiedMode,
            registerUploadedDatasets,
            switchActiveDataset,
            combineDatasetsAction,
            useAllFilesTogether,
            useFileSeparately,
            removeUploadedDataset,
            // Cross-page operation results
            cleaningResult, setCleaningResult,
            featureEngineeringResult, setFeatureEngineeringResult,
            modelResult, setModelResult,
            predictionResult, setPredictionResult,
            syncAgentRunResults,
            // Agent Custom Dashboard & Database Live Sync
            agentDashboard, setAgentDashboard,
            activeDatabaseConfig, setActiveDatabaseConfig,
            refreshFromDatabase
        }}>
            {children}
        </DataContext.Provider>
    );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useData = () => {
    const context = useContext(DataContext);
    if (!context) {
        throw new Error('useData must be used within a DataProvider');
    }
    return context;
};
