import React, { useCallback, useState } from 'react';
import { 
    Upload, FileText, CheckCircle, AlertCircle, Loader, Database, X, 
    Layers, GitMerge, Trash2, ArrowRight, Table, Check, Plus, FolderKanban, Info, Sparkles
} from 'lucide-react';
import { uploadFile, loadData, connectDatabase, deleteFile, combineDatasets } from '../services/api';
import { useData } from '../context/DataContext';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../context/ToastContext';

export default function DataLoad() {
    const [dragActive, setDragActive] = useState(false);
    const [uploadingProgress, setUploadingProgress] = useState(null);
    const [selectedDatasetIds, setSelectedDatasetIds] = useState([]);
    const [showJoinModal, setShowJoinModal] = useState(false);
    const [joinKey, setJoinKey] = useState('');
    const [combineModalMode, setCombineModalMode] = useState('columns'); // 'columns' | 'merge' | 'concat'
    
    const { 
        dataSummary, setDataSummary, 
        dataPreview, setDataPreview, 
        setLoading, loading, 
        setError, error,
        uploadedDatasets,
        activeDatasetId,
        isUnifiedMode,
        useAllFilesTogether,
        useFileSeparately,
        registerUploadedDatasets,
        switchActiveDataset,
        combineDatasetsAction,
        removeUploadedDataset,
        setUploadedDatasets,
        setActiveDatasetId,
        setActiveDatabaseConfig
    } = useData();

    const rawDatasets = uploadedDatasets ? uploadedDatasets.filter(d => !d.isCombined) : [];
    const activeDataset = uploadedDatasets ? (uploadedDatasets.find(d => d.id === activeDatasetId) || uploadedDatasets[0]) : null;

    const selectedDatasets = (uploadedDatasets || []).filter(d => selectedDatasetIds.includes(d.id));
    const candidateColumns = React.useMemo(() => {
        if (selectedDatasets.length < 2) return [];
        const cols0 = selectedDatasets[0].summary?.columns || [];
        const cols1 = selectedDatasets[1].summary?.columns || [];
        const common = [];
        cols0.forEach(c0 => {
            const match = cols1.find(c1 => String(c1).toLowerCase().trim() === String(c0).toLowerCase().trim());
            if (match && !common.includes(c0)) {
                common.push(c0);
            }
        });
        return common;
    }, [selectedDatasets]);

    const navigate = useNavigate();
    const { addToast } = useToast();

    // DB Connection State
    const [showDbModal, setShowDbModal] = useState(false);
    const [dbType, setDbType] = useState('postgresql');
    const [connString, setConnString] = useState('');
    const [query, setQuery] = useState('SELECT * FROM users LIMIT 100');
    const [collection, setCollection] = useState('users');

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    }, []);

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    };

    const formatFileSize = (bytes) => {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    };

    const handleFiles = async (files) => {
        setLoading(true);
        setError(null);
        const count = files.length;
        setUploadingProgress(`Uploading ${count} file${count > 1 ? 's' : ''}...`);

        try {
            console.log(`[DataLoad] Starting upload for ${count} file(s)...`);
            const uploadRes = await uploadFile(files);
            console.log('[DataLoad] Upload successful:', uploadRes);

            const uploadedList = uploadRes.files || [
                {
                    filePath: uploadRes.filePath,
                    filename: uploadRes.filename,
                    originalName: uploadRes.originalName,
                    size: uploadRes.size
                }
            ];

            // Primary file to activate immediately
            const primary = uploadedList[0];
            setUploadingProgress(`Reading & inspecting ${primary.originalName}...`);

            const ext = primary.originalName.split('.').pop().toLowerCase();
            const fileType = (ext === 'xlsx' || ext === 'xls') ? 'excel' : (ext === 'zip' ? 'zip' : 'csv');

            const dataRes = await loadData(primary.filePath, fileType);
            console.log('[DataLoad] Primary data loaded successfully:', dataRes);

            setDataSummary(dataRes);
            setDataPreview(dataRes.preview || []);

            // Track active session file
            if (primary.filePath) {
                sessionStorage.setItem('active_session_file', primary.filePath);
            }

            // Register in multi-dataset registry
            if (registerUploadedDatasets) {
                registerUploadedDatasets(uploadedList, dataRes, dataRes.preview);
            }

            if (count === 1) {
                addToast(`'${primary.originalName}' loaded successfully!`, 'success');
            } else {
                addToast(`${count} files uploaded! '${primary.originalName}' is active.`, 'success');
            }
        } catch (err) {
            console.error('[DataLoad] Error:', err);
            const errorMsg = err.response?.data?.message ||
                err.response?.data?.error ||
                err.response?.data?.detail ||
                err.message ||
                'Failed to upload/load files';
            setError(errorMsg);
            addToast(`Upload failed: ${errorMsg}`, 'error');
        } finally {
            setLoading(false);
            setUploadingProgress(null);
        }
    };

    const handleSwitchDataset = async (datasetId) => {
        try {
            await useFileSeparately(datasetId);
            const ds = uploadedDatasets.find(d => d.id === datasetId);
            addToast(`Switched active dataset to '${ds?.name || 'Dataset'}' (Separate mode)`, 'info');
        } catch (err) {
            addToast(`Failed to switch dataset: ${err.message}`, 'error');
        }
    };

    const handleActivateAllTogether = async () => {
        try {
            await useAllFilesTogether('auto');
            addToast(`✨ All ${rawDatasets.length} files combined into an optimized master dataset!`, 'success');
        } catch (err) {
            addToast(`Failed to combine files: ${err.message}`, 'error');
        }
    };

    const toggleDatasetSelection = (id) => {
        setSelectedDatasetIds(prev => 
            prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]
        );
    };

    const handleCombineColumns = async () => {
        if (selectedDatasetIds.length < 2) {
            addToast('Please select at least 2 datasets to merge side-by-side', 'warning');
            return;
        }
        try {
            await combineDatasetsAction(selectedDatasetIds, 'columns');
            addToast(`Merged ${selectedDatasetIds.length} datasets side-by-side (all features preserved without nulls)!`, 'success');
            setSelectedDatasetIds([]);
        } catch (err) {
            addToast(`Merge failed: ${err.message}`, 'error');
        }
    };

    const handleCombineConcat = async () => {
        if (selectedDatasetIds.length < 2) {
            addToast('Please select at least 2 datasets to union rows', 'warning');
            return;
        }
        try {
            await combineDatasetsAction(selectedDatasetIds, 'concat');
            addToast(`Stacked ${selectedDatasetIds.length} datasets into a single table!`, 'success');
            setSelectedDatasetIds([]);
        } catch (err) {
            addToast(`Union failed: ${err.message}`, 'error');
        }
    };

    const handleCombineJoin = async () => {
        if (selectedDatasetIds.length < 2) {
            addToast('Please select at least 2 datasets to combine', 'warning');
            return;
        }
        try {
            if (combineModalMode === 'columns') {
                await combineDatasetsAction(selectedDatasetIds, 'columns');
                addToast(`Merged ${selectedDatasetIds.length} datasets side-by-side (all features preserved without nulls)!`, 'success');
            } else if (combineModalMode === 'concat') {
                await combineDatasetsAction(selectedDatasetIds, 'concat');
                addToast(`Stacked ${selectedDatasetIds.length} datasets into a single table!`, 'success');
            } else {
                if (!joinKey.trim()) {
                    addToast('Please specify or pick a key column to join on (e.g. Account ID, ID)', 'warning');
                    return;
                }
                await combineDatasetsAction(selectedDatasetIds, 'merge', joinKey.trim());
                addToast(`Joined datasets on '${joinKey}'!`, 'success');
            }
            setShowJoinModal(false);
            setJoinKey('');
            setSelectedDatasetIds([]);
        } catch (err) {
            addToast(`Combine failed: ${err.message}`, 'error');
        }
    };

    const handleDbConnect = async () => {
        setLoading(true);
        setError(null);
        try {
            const payload = {
                db_type: dbType,
                connection_string: connString,
                query: (dbType === 'postgresql' || dbType === 'sqlite' || dbType === 'mysql' || dbType === 'redshift') ? query : undefined,
                collection: dbType === 'mongodb' ? collection : undefined
            };

            const dataRes = await connectDatabase(payload);
            setDataSummary(dataRes);
            setDataPreview(dataRes.preview || []);
            setActiveDatabaseConfig(payload);

            const dbDataset = {
                id: `db-${dbType}-${Date.now()}`,
                filename: `Live DB (${dbType.toUpperCase()})`,
                filePath: null,
                fileType: 'database',
                isDatabase: true,
                dbType: dbType,
                dbConfig: payload,
                shape: dataRes.shape || [0, 0],
                uploadedAt: new Date().toISOString(),
                isActive: true,
                summary: dataRes
            };

            setUploadedDatasets(prev => {
                const current = prev || [];
                const updated = current.map(d => ({ ...d, isActive: false }));
                const list = [dbDataset, ...updated];
                try {
                    sessionStorage.setItem('nexus_uploaded_datasets', JSON.stringify(list));
                } catch (e) {}
                return list;
            });
            setActiveDatasetId(dbDataset.id);

            setShowDbModal(false);
            navigate('/dashboard/custom');
            addToast(`Connected to ${dbType.toUpperCase()} database successfully! Custom Dashboard adapted.`, 'success');
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.error || err.response?.data?.detail || 'Failed to connect to database');
            addToast('Failed to connect to database', 'error');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-4 md:p-8 max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex justify-between items-center flex-wrap gap-4 bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <h1 className="text-3xl font-extrabold text-slate-800">Dataset Manager</h1>
                        <span className="px-2.5 py-0.5 bg-blue-50 text-blue-700 text-xs font-bold rounded-full border border-blue-200">
                            Multi-File Support
                        </span>
                    </div>
                    <p className="text-slate-500 text-sm">
                        Upload one or more files (CSV, Excel, ZIP), switch between datasets, or merge tables seamlessly.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowDbModal(true)}
                        className="px-4 py-2 border border-slate-200 hover:border-slate-300 text-slate-700 rounded-xl font-medium text-sm flex items-center gap-2 transition-colors bg-white hover:bg-slate-50 shadow-sm"
                    >
                        <Database size={16} className="text-slate-500" />
                        Connect Database
                    </button>
                    {dataSummary && (
                        <button
                            onClick={() => navigate('/')}
                            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl font-medium text-sm flex items-center gap-2 transition-colors shadow-sm"
                        >
                            Open Dashboard
                            <ArrowRight size={16} />
                        </button>
                    )}
                </div>
            </div>

            {/* Dropzone Container */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <div
                    className={`border-2 border-dashed rounded-xl p-8 md:p-10 text-center transition-all duration-200 ${
                        dragActive ? 'border-primary bg-blue-50/50 scale-[1.01]' : 'border-slate-300 hover:border-primary hover:bg-slate-50/60'
                    }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <div className="flex flex-col items-center justify-center gap-3">
                        <div className="w-14 h-14 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center shadow-sm">
                            {loading ? <Loader className="animate-spin" size={28} /> : <Upload size={28} />}
                        </div>

                        {loading ? (
                            <div className="space-y-1">
                                <h3 className="text-lg font-bold text-slate-800">
                                    {uploadingProgress || 'Processing Datasets...'}
                                </h3>
                                <p className="text-xs text-slate-500">Inspecting schema and calculating quality metrics.</p>
                            </div>
                        ) : (
                            <>
                                <div>
                                    <h3 className="text-lg font-bold text-slate-800">Drag & Drop one or more files</h3>
                                    <p className="text-xs text-slate-500 mt-0.5">Select multiple files at once or drop them here</p>
                                </div>
                                <div className="flex items-center gap-2 my-1">
                                    <div className="h-px w-12 bg-slate-200" />
                                    <span className="text-xs uppercase text-slate-400 font-semibold tracking-wider">or</span>
                                    <div className="h-px w-12 bg-slate-200" />
                                </div>
                                <label
                                    htmlFor="file-upload"
                                    className="px-5 py-2 bg-slate-900 text-white rounded-xl text-sm font-semibold cursor-pointer hover:bg-slate-800 transition-all shadow-sm hover:shadow"
                                >
                                    Browse Multiple Files
                                </label>
                                <input
                                    id="file-upload"
                                    type="file"
                                    multiple
                                    className="hidden"
                                    onChange={handleChange}
                                    accept=".csv,.xlsx,.xls,.zip"
                                />
                                <p className="text-xs text-slate-400 mt-2">
                                    Supports CSV, Excel (.xlsx, .xls), and ZIP archives (up to 100MB per file)
                                </p>
                            </>
                        )}
                    </div>
                </div>

                {error && (
                    <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-xl flex items-center gap-3 border border-red-100 text-sm">
                        <AlertCircle size={18} className="shrink-0" />
                        <span>{error}</span>
                    </div>
                )}
            </div>

            {/* Uploaded Datasets Tray / Manager */}
            {uploadedDatasets && uploadedDatasets.length > 0 && (
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
                    <div className="flex justify-between items-center flex-wrap gap-3 pb-3 border-b border-slate-100">
                        <div className="flex items-center gap-2">
                            <FolderKanban size={20} className="text-blue-600" />
                            <h2 className="text-lg font-bold text-slate-800">Uploaded Datasets</h2>
                            <span className="px-2 py-0.5 bg-slate-100 text-slate-700 text-xs font-bold rounded-full">
                                {uploadedDatasets.length} file{uploadedDatasets.length > 1 ? 's' : ''}
                            </span>
                        </div>

                        {/* Batch Operations Bar */}
                        <div className="flex items-center gap-2 flex-wrap">
                            {selectedDatasetIds.length >= 2 && (
                                <>
                                    <button
                                        onClick={handleCombineColumns}
                                        disabled={loading}
                                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm cursor-pointer"
                                        title="Merge columns side-by-side so all features are preserved without empty rows"
                                    >
                                        <Table size={14} />
                                        Merge Features Side-by-Side
                                    </button>
                                    <button
                                        onClick={() => {
                                            setCombineModalMode('columns');
                                            setShowJoinModal(true);
                                        }}
                                        disabled={loading}
                                        className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm cursor-pointer"
                                    >
                                        <GitMerge size={14} />
                                        Combine / Join Options...
                                    </button>
                                    <button
                                        onClick={handleCombineConcat}
                                        disabled={loading}
                                        className="px-3 py-1.5 bg-slate-700 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm cursor-pointer"
                                        title="Stack rows vertically (for partitioned datasets with same columns)"
                                    >
                                        <Layers size={14} />
                                        Union Rows ({selectedDatasetIds.length})
                                    </button>
                                </>
                            )}
                            <label
                                htmlFor="file-upload"
                                className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-semibold cursor-pointer flex items-center gap-1.5 transition-colors"
                            >
                                <Plus size={14} />
                                Add More Files
                            </label>
                        </div>
                    </div>

                    {/* Operational Scope Selector: Use Together vs Use Separately */}
                    {rawDatasets.length > 1 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-slate-50 rounded-xl border border-slate-200">
                            {/* Card 1: Use All Files Together */}
                            <div className={`p-4 rounded-xl border transition-all ${
                                isUnifiedMode 
                                    ? 'bg-purple-50/80 border-purple-400 ring-2 ring-purple-300' 
                                    : 'bg-white border-slate-200 hover:border-purple-300'
                            }`}>
                                <div className="flex items-start justify-between gap-2 mb-2">
                                    <div className="flex items-center gap-2">
                                        <div className="p-2 bg-purple-100 text-purple-700 rounded-lg">
                                            <Sparkles size={18} />
                                        </div>
                                        <div>
                                            <h3 className="font-bold text-slate-800 text-sm">Use All Files Together</h3>
                                            <span className="text-[11px] text-purple-700 font-semibold">Unified Cross-Table Master Dataset</span>
                                        </div>
                                    </div>
                                    {isUnifiedMode && (
                                        <span className="px-2 py-0.5 bg-purple-600 text-white text-[10px] font-bold rounded-full uppercase tracking-wider">
                                            Active Scope
                                        </span>
                                    )}
                                </div>
                                <p className="text-xs text-slate-600 mb-3">
                                    Combines all {rawDatasets.length} uploaded files into one unified master dataset for cross-table modeling, cleaning, and autonomous analysis.
                                </p>
                                <button
                                    onClick={handleActivateAllTogether}
                                    disabled={loading || isUnifiedMode}
                                    className={`w-full py-2 px-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                                        isUnifiedMode
                                            ? 'bg-purple-600 text-white cursor-default'
                                            : 'bg-purple-600 hover:bg-purple-700 text-white shadow-sm'
                                    }`}
                                >
                                    <Sparkles size={14} />
                                    {isUnifiedMode ? '✨ Unified Master Active' : 'Activate All Together (Auto-Merge)'}
                                </button>
                            </div>

                            {/* Card 2: Use Files Separately */}
                            <div className={`p-4 rounded-xl border transition-all ${
                                !isUnifiedMode 
                                    ? 'bg-blue-50/80 border-blue-400 ring-2 ring-blue-300' 
                                    : 'bg-white border-slate-200 hover:border-blue-300'
                            }`}>
                                <div className="flex items-start justify-between gap-2 mb-2">
                                    <div className="flex items-center gap-2">
                                        <div className="p-2 bg-blue-100 text-blue-700 rounded-lg">
                                            <FolderKanban size={18} />
                                        </div>
                                        <div>
                                            <h3 className="font-bold text-slate-800 text-sm">Use Files Separately</h3>
                                            <span className="text-[11px] text-blue-700 font-semibold">Independent Dataset Focus</span>
                                        </div>
                                    </div>
                                    {!isUnifiedMode && (
                                        <span className="px-2 py-0.5 bg-blue-600 text-white text-[10px] font-bold rounded-full uppercase tracking-wider">
                                            Active Scope
                                        </span>
                                    )}
                                </div>
                                <p className="text-xs text-slate-600 mb-3">
                                    Focus on one specific file at a time. Click "Activate" on any file card below to direct all analytics and models to that file.
                                </p>
                                <div className="text-xs font-semibold text-slate-600 py-2 px-3 text-center bg-white rounded-lg border border-slate-200">
                                    {!isUnifiedMode ? `Currently active: ${activeDataset?.name || 'Individual File'}` : 'Click "Activate" on any file below to switch to separate mode'}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Dataset Cards Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {uploadedDatasets.map((dataset) => {
                            const isCurrentActive = dataset.id === activeDatasetId || dataset.isActive;
                            const isSelected = selectedDatasetIds.includes(dataset.id);

                            return (
                                <div
                                    key={dataset.id}
                                    className={`relative p-4 rounded-xl border transition-all ${
                                        isCurrentActive 
                                            ? 'border-blue-500 bg-blue-50/40 shadow-sm ring-1 ring-blue-500' 
                                            : isSelected
                                            ? 'border-indigo-300 bg-indigo-50/30'
                                            : 'border-slate-200 bg-white hover:border-slate-300 hover:shadow-xs'
                                    }`}
                                >
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="flex items-start gap-2.5 min-w-0">
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => toggleDatasetSelection(dataset.id)}
                                                className="mt-1 rounded text-primary focus:ring-primary cursor-pointer"
                                                title="Select to merge with another dataset"
                                            />
                                            <div className="p-2 bg-slate-100 text-slate-600 rounded-lg shrink-0">
                                                <FileText size={18} />
                                            </div>
                                            <div className="min-w-0">
                                                <h4 className="font-semibold text-slate-800 text-sm truncate" title={dataset.name}>
                                                    {dataset.name}
                                                </h4>
                                                <div className="flex items-center gap-2 mt-0.5 text-xs text-slate-500">
                                                    <span>{formatFileSize(dataset.size)}</span>
                                                    <span>•</span>
                                                    <span className="uppercase font-mono text-[10px] bg-slate-100 px-1.5 py-0.2 rounded">
                                                        {dataset.fileType}
                                                    </span>
                                                    {dataset.isCombined && (
                                                        <span className="text-[10px] bg-emerald-100 text-emerald-700 font-bold px-1.5 py-0.2 rounded">
                                                            Combined
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        <button
                                            onClick={() => removeUploadedDataset(dataset.id)}
                                            className="text-slate-400 hover:text-red-500 p-1 rounded-md transition-colors"
                                            title="Remove dataset"
                                        >
                                            <Trash2 size={15} />
                                        </button>
                                    </div>

                                    {/* Dataset Stats & Action */}
                                    <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
                                        <div className="text-xs text-slate-600 font-medium">
                                            {dataset.shape ? (
                                                <span>{dataset.shape[0]?.toLocaleString()} rows × {dataset.shape[1]} cols</span>
                                            ) : isCurrentActive && dataSummary?.shape ? (
                                                <span>{dataSummary.shape[0]?.toLocaleString()} rows × {dataSummary.shape[1]} cols</span>
                                            ) : (
                                                <span className="text-slate-400">Ready to inspect</span>
                                            )}
                                        </div>

                                        {isCurrentActive ? (
                                            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-emerald-100 text-emerald-800 text-[11px] font-bold rounded-full">
                                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                                Active
                                            </span>
                                        ) : (
                                            <button
                                                onClick={() => handleSwitchDataset(dataset.id)}
                                                disabled={loading}
                                                className="px-2.5 py-1 bg-white hover:bg-slate-100 text-slate-700 border border-slate-200 text-xs font-semibold rounded-lg transition-colors flex items-center gap-1 shadow-2xs"
                                            >
                                                Activate
                                            </button>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Quick Launchpad to Matching Pages */}
                    <div className="pt-3 border-t border-slate-100 flex items-center justify-between flex-wrap gap-2">
                        <div className="flex items-center gap-1.5 text-xs text-slate-500">
                            <Info size={14} className="text-blue-500 shrink-0" />
                            <span>Select multiple files using the checkboxes above to merge or join them.</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => navigate('/agent')}
                                className="px-3 py-1.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg text-xs font-bold hover:opacity-95 transition-all shadow-xs flex items-center gap-1.5"
                            >
                                Run Autonomous Agent ➔
                            </button>
                            <button
                                onClick={() => navigate('/clean')}
                                className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-semibold transition-colors"
                            >
                                Clean Data
                            </button>
                            <button
                                onClick={() => navigate('/features')}
                                className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-semibold transition-colors"
                            >
                                Feature Synthesis
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Active Dataset Preview Table */}
            {dataSummary && dataPreview && dataPreview.length > 0 && (
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                        <div className="flex items-center gap-2">
                            <Table size={20} className="text-emerald-600" />
                            <h3 className="font-bold text-slate-800 text-base">
                                Active Dataset Live Preview
                            </h3>
                            <span className="text-xs text-slate-500">
                                (Showing first {dataPreview.length} rows)
                            </span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-slate-600 font-medium">
                            <span className="bg-slate-100 px-2.5 py-1 rounded-md">
                                Total Rows: <strong>{dataSummary.shape?.[0]?.toLocaleString()}</strong>
                            </span>
                            <span className="bg-slate-100 px-2.5 py-1 rounded-md">
                                Columns: <strong>{dataSummary.shape?.[1]}</strong>
                            </span>
                        </div>
                    </div>

                    <div className="overflow-x-auto rounded-xl border border-slate-200">
                        <table className="min-w-full text-xs text-left text-slate-600">
                            <thead className="bg-slate-50 text-slate-700 font-bold uppercase text-[10px] tracking-wider border-b border-slate-200">
                                <tr>
                                    {Object.keys(dataPreview[0] || {}).map((col) => (
                                        <th key={col} className="px-4 py-2.5 whitespace-nowrap">
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {dataPreview.slice(0, 8).map((row, idx) => (
                                    <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                                        {Object.values(row).map((val, cellIdx) => (
                                            <td key={cellIdx} className="px-4 py-2 whitespace-nowrap font-mono text-[11px] text-slate-700">
                                                {val !== null && val !== undefined ? String(val) : <span className="text-slate-400 italic">null</span>}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Combine & Merge Datasets Modal */}
            {showJoinModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-xs p-4">
                    <div className="bg-white rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden border border-slate-100">
                        <div className="p-5 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                            <div className="flex items-center gap-2">
                                <GitMerge size={20} className="text-indigo-600" />
                                <h3 className="text-lg font-bold text-slate-800">Combine & Merge Datasets</h3>
                            </div>
                            <button onClick={() => setShowJoinModal(false)} className="text-slate-400 hover:text-slate-600 cursor-pointer">
                                <X size={20} />
                            </button>
                        </div>

                        <div className="p-5 space-y-4">
                            <div className="text-xs text-slate-600">
                                Choose how you want to combine <strong>{selectedDatasetIds.length} selected datasets</strong>:
                            </div>

                            {/* Mode Selector Tabs */}
                            <div className="grid grid-cols-3 gap-2 p-1 bg-slate-100 rounded-xl text-xs font-bold">
                                <button
                                    type="button"
                                    onClick={() => setCombineModalMode('columns')}
                                    className={`py-2 px-2 rounded-lg transition text-center cursor-pointer ${
                                        combineModalMode === 'columns'
                                            ? 'bg-white text-indigo-700 shadow-sm'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    Side-by-Side (Columns)
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setCombineModalMode('merge')}
                                    className={`py-2 px-2 rounded-lg transition text-center cursor-pointer ${
                                        combineModalMode === 'merge'
                                            ? 'bg-white text-indigo-700 shadow-sm'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    Key-Based Join
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setCombineModalMode('concat')}
                                    className={`py-2 px-2 rounded-lg transition text-center cursor-pointer ${
                                        combineModalMode === 'concat'
                                            ? 'bg-white text-indigo-700 shadow-sm'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    Union Rows
                                </button>
                            </div>

                            {combineModalMode === 'columns' && (
                                <div className="p-3.5 bg-blue-50 border border-blue-200 rounded-xl space-y-1 text-xs text-blue-900">
                                    <div className="font-bold flex items-center gap-1.5 text-blue-800">
                                        <Sparkles size={14} className="text-blue-600" />
                                        Recommended for Different Schemas
                                    </div>
                                    <p className="text-blue-700 leading-relaxed">
                                        Combines all features from both datasets horizontally without deleting or setting any columns to null. Every row receives data from both files.
                                    </p>
                                </div>
                            )}

                            {combineModalMode === 'merge' && (
                                <div className="space-y-3">
                                    <p className="text-xs text-slate-500">
                                        Matches records on a shared identifier column (e.g., <code>Account ID</code>, <code>id</code>, <code>username</code>).
                                    </p>
                                    <div>
                                        <label className="block text-xs font-bold text-slate-700 uppercase mb-1">
                                            Common Key Column Name
                                        </label>
                                        <input
                                            type="text"
                                            value={joinKey}
                                            onChange={(e) => setJoinKey(e.target.value)}
                                            placeholder="e.g. Account ID, Customer_ID, id"
                                            className="w-full p-2.5 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                                            autoFocus
                                        />
                                    </div>

                                    {candidateColumns.length > 0 && (
                                        <div className="flex items-center gap-1.5 flex-wrap pt-1">
                                            <span className="text-[11px] text-slate-500 font-medium">Detected matching keys:</span>
                                            {candidateColumns.map(col => (
                                                <button
                                                    key={col}
                                                    type="button"
                                                    onClick={() => setJoinKey(col)}
                                                    className={`text-[11px] px-2.5 py-1 rounded-full font-mono font-semibold transition cursor-pointer ${
                                                        joinKey.toLowerCase() === col.toLowerCase()
                                                            ? 'bg-indigo-600 text-white shadow-xs'
                                                            : 'bg-indigo-50 text-indigo-700 hover:bg-indigo-100'
                                                    }`}
                                                >
                                                    {col}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {combineModalMode === 'concat' && (
                                <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl space-y-1 text-xs text-slate-700">
                                    <div className="font-bold text-slate-800">
                                        Vertical Row Stacking
                                    </div>
                                    <p className="leading-relaxed">
                                        Stacks rows sequentially. Best when both datasets share the exact same column structure (e.g. monthly transaction partitions).
                                    </p>
                                </div>
                            )}
                        </div>

                        <div className="p-4 border-t border-slate-100 bg-slate-50 flex justify-end gap-2">
                            <button
                                onClick={() => setShowJoinModal(false)}
                                className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-200 rounded-xl text-xs cursor-pointer"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleCombineJoin}
                                disabled={loading || (combineModalMode === 'merge' && !joinKey.trim())}
                                className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold text-xs disabled:opacity-50 flex items-center gap-1.5 transition-colors shadow-sm cursor-pointer"
                            >
                                {loading && <Loader size={14} className="animate-spin" />}
                                {combineModalMode === 'columns' ? 'Merge Features Side-by-Side' : combineModalMode === 'merge' ? 'Join on Key' : 'Union Rows'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Database Modal */}
            {showDbModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-xs p-4">
                    <div className="bg-white rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden">
                        <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                            <h3 className="text-xl font-bold text-slate-800">Connect Database</h3>
                            <button onClick={() => setShowDbModal(false)} className="text-slate-400 hover:text-slate-600">
                                <X size={24} />
                            </button>
                        </div>

                        <div className="p-6 space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Database Type</label>
                                <select
                                    value={dbType}
                                    onChange={e => setDbType(e.target.value)}
                                    className="w-full p-2 border border-slate-300 rounded-lg"
                                >
                                    <option value="postgresql">PostgreSQL</option>
                                    <option value="mongodb">MongoDB</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Connection String</label>
                                <input
                                    type="text"
                                    value={connString}
                                    onChange={e => setConnString(e.target.value)}
                                    placeholder={dbType === 'postgresql' ? "postgresql://user:pass@localhost:5432/db" : "mongodb://localhost:27017/db"}
                                    className="w-full p-2 border border-slate-300 rounded-lg text-sm"
                                />
                                <p className="text-xs text-slate-500 mt-1">Include credentials if required.</p>
                            </div>

                            {dbType === 'postgresql' ? (
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">SQL Query</label>
                                    <textarea
                                        value={query}
                                        onChange={e => setQuery(e.target.value)}
                                        rows={3}
                                        className="w-full p-2 border border-slate-300 rounded-lg font-mono text-sm"
                                        placeholder="SELECT * FROM table"
                                    />
                                </div>
                            ) : (
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">Collection Name</label>
                                    <input
                                        type="text"
                                        value={collection}
                                        onChange={e => setCollection(e.target.value)}
                                        className="w-full p-2 border border-slate-300 rounded-lg text-sm"
                                        placeholder="users"
                                    />
                                </div>
                            )}
                        </div>

                        <div className="p-6 border-t border-slate-100 bg-slate-50 flex justify-end gap-3">
                            <button
                                onClick={() => setShowDbModal(false)}
                                className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-200 rounded-lg text-sm"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleDbConnect}
                                disabled={loading}
                                className="px-6 py-2 bg-primary text-white rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50 flex items-center gap-2 text-sm"
                            >
                                {loading && <Loader size={16} className="animate-spin" />}
                                {loading ? 'Connecting...' : 'Connect'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
