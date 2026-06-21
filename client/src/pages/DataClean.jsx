import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import { cleanData, exportData, undoAction, redoAction, getDataQuality, getPipeline } from '../services/api';
import { Trash2, Copy, Filter, Check, Eraser, Columns, ShieldAlert, Binary, Scaling, Download, Undo2, Redo2, Activity, Loader, Edit2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../context/ToastContext';

const ActionButton = ({ icon: Icon, label, onClick, disabled }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`flex items-center w-full gap-3 px-4 py-3 rounded-xl transition-all font-medium text-left
      ${disabled
                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                : 'bg-white hover:bg-slate-50 text-slate-700 shadow-sm border border-slate-200 hover:border-primary/50'
            }
    `}
    >
        <div className={`p-2 rounded-lg ${disabled ? 'bg-slate-200' : 'bg-blue-50 text-blue-600'}`}>
            <Icon size={20} />
        </div>
        <span>{label}</span>
    </button>
);

export default function DataClean() {
    const { dataSummary, setDataSummary, setDataPreview, setLoading, loading, setError } = useData();
    const navigate = useNavigate();
    const { addToast } = useToast();
    const [selectedColumns, setSelectedColumns] = useState([]);
    const [fillStrategy, setFillStrategy] = useState('mean');
    const [encodeStrategy, setEncodeStrategy] = useState('label');
    const [scaleStrategy, setScaleStrategy] = useState('minmax');
    const [qualityScore, setQualityScore] = useState(null);
    const [pipelineSteps, setPipelineSteps] = useState([]);
    const [renameMap, setRenameMap] = useState({ old: '', new: '' });
    const [dropnaColumn, setDropnaColumn] = useState('');
    const [fillnaColumn, setFillnaColumn] = useState('');

    React.useEffect(() => {
        if (dataSummary) {
            fetchQuality();
            fetchPipeline();
        }
    }, [dataSummary]);

    // Auto-refresh pipeline when data summary changes (cleaning action done)

    const fetchPipeline = async () => {
        try {
            const res = await getPipeline();
            setPipelineSteps(res.steps || []);
        } catch (err) {
            console.error("Failed to fetch pipeline", err);
        }
    };

    const fetchQuality = async () => {
        try {
            const res = await getDataQuality();
            setQualityScore(res);
        } catch (err) {
            console.error("Failed to fetch quality", err);
        }
    };

    // If no data, show message but keep UI visible (mostly disabled)

    const { columns = [], missing_values = {} } = dataSummary || {};

    // Helper to check if data is active
    const hasData = !!dataSummary;

    const handleClean = async (payload) => {
        console.log("Cleaning with payload:", payload);
        setLoading(true);
        try {
            const res = await cleanData(payload);
            console.log("Clean success:", res);
            setDataSummary(res);
            setDataPreview(res.preview);
            addToast('Data cleaned successfully!', 'success');
        } catch (err) {
            console.error("Clean error:", err);
            setError('Failed to clean data');
            addToast('Failed to clean data', 'error');
        } finally {
            setLoading(false);
        }
    };

    const handleUndo = async () => {
        setLoading(true);
        try {
            const res = await undoAction();
            setDataSummary(res);
            setDataPreview(res.preview);
            addToast('Undo successful', 'success');
        } catch (err) {
            console.warn("Undo failed", err);
            addToast('Undo failed', 'error');
        } finally {
            setLoading(false);
        }
    };

    const handleRedo = async () => {
        setLoading(true);
        try {
            const res = await redoAction();
            setDataSummary(res);
            setDataPreview(res.preview);
            addToast('Redo successful', 'success');
        } catch (err) {
            console.warn("Redo failed", err);
            addToast('Redo failed', 'error');
        } finally {
            setLoading(false);
        }
    };

    const toggleColumn = (col) => {
        if (selectedColumns.includes(col)) {
            setSelectedColumns(selectedColumns.filter(c => c !== col));
        } else {
            setSelectedColumns([...selectedColumns, col]);
        }
    };

    return (
        <div className="p-4 md:p-8 max-w-7xl mx-auto flex flex-col md:flex-row gap-6 md:gap-8 relative min-h-screen">
            {!hasData && (
                <div className="absolute inset-0 z-10 bg-white/80 backdrop-blur-sm flex items-center justify-center rounded-2xl">
                    <div className="bg-white p-8 rounded-2xl shadow-xl text-center border border-slate-200 max-w-md">
                        <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Activity size={32} />
                        </div>
                        <h2 className="text-xl font-bold text-slate-800 mb-2">No Data Loaded</h2>
                        <p className="text-slate-500 mb-6">Please upload a dataset to start cleaning and transforming your data.</p>
                        <button
                            onClick={() => navigate('/load')}
                            className="w-full py-3 bg-primary text-white rounded-xl font-medium hover:bg-primary/90 transition-colors shadow-lg shadow-blue-500/20"
                        >
                            Go to Upload
                        </button>
                    </div>
                </div>
            )}

            {/* Sidebar Actions */}
            <div className={`w-full md:w-72 lg:w-80 flex-shrink-0 space-y-6 md:h-[calc(100vh-100px)] md:sticky md:top-20 md:overflow-y-auto pr-2 ${!hasData ? 'blur-sm pointer-events-none opacity-50' : ''}`}>
                <div>
                    <h2 className="text-xl font-bold text-slate-800 mb-2">Cleaning Tools</h2>
                    <p className="text-sm text-slate-500 mb-4">Apply transformations to your dataset.</p>
                </div>

                {/* Quality Score Badge */}
                {qualityScore && (
                    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm mb-4">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-slate-700 flex items-center gap-2">
                                <Activity size={16} className="text-blue-500" /> Data Health
                            </span>
                            <span className={`text-lg font-bold ${qualityScore.score >= 80 ? 'text-green-600' :
                                qualityScore.score >= 50 ? 'text-orange-500' : 'text-red-600'
                                }`}>
                                {qualityScore.score}%
                            </span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-2 mb-3">
                            <div
                                className={`h-2 rounded-full transition-all duration-500 ${qualityScore.score >= 80 ? 'bg-green-500' :
                                    qualityScore.score >= 50 ? 'bg-orange-400' : 'bg-red-500'
                                    }`}
                                style={{ width: `${qualityScore.score}%` }}
                            ></div>
                        </div>
                        <div className="text-xs text-slate-500 space-y-1">
                            <div className="flex justify-between">
                                <span>Missing Cells:</span>
                                <span>{qualityScore.metrics.missing_cells_pct}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Duplicates:</span>
                                <span>{qualityScore.metrics.duplicate_rows_pct}%</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Column Selection - MOVED TO TOP */}
                <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-2 opacity-10">
                        <Columns size={48} />
                    </div>
                    <h3 className="font-bold text-slate-800 mb-3 flex items-center gap-2 relative z-10">
                        <Columns size={18} className="text-primary" /> Select Columns
                    </h3>
                    <p className="text-xs text-slate-500 mb-2 relative z-10">Select columns to apply actions to.</p>
                    <div className="max-h-40 overflow-y-auto mb-3 space-y-2 border border-slate-200 rounded p-2 bg-slate-50 relative z-10">
                        {columns.map(col => (
                            <label key={col} className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer hover:text-slate-900">
                                <input
                                    type="checkbox"
                                    checked={selectedColumns.includes(col)}
                                    onChange={() => toggleColumn(col)}
                                    className="rounded text-primary focus:ring-primary"
                                />
                                {col}
                            </label>
                        ))}
                    </div>
                    <div className="flex gap-2 relative z-10">
                        <button
                            onClick={() => setSelectedColumns(columns)}
                            className="text-xs text-primary hover:underline font-medium"
                        >
                            Select All
                        </button>
                        <button
                            onClick={() => setSelectedColumns([])}
                            className="text-xs text-slate-500 hover:underline"
                        >
                            Clear Selection
                        </button>
                    </div>

                    <div className="mt-3 relative z-10 border-t border-slate-100 pt-2">
                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Quick Actions</p>
                        <div className="grid grid-cols-2 gap-2">
                            <button
                                onClick={() => {
                                    handleClean({ operation: 'drop_columns', columns: selectedColumns });
                                    setSelectedColumns([]);
                                }}
                                disabled={selectedColumns.length === 0 || loading}
                                className="flex items-center justify-center gap-1.5 px-3 py-2 bg-red-50 text-red-600 rounded-lg text-xs font-medium hover:bg-red-100 disabled:opacity-50 transition-colors"
                            >
                                <Eraser size={14} /> Drop Selected
                            </button>
                            <button
                                onClick={() => handleClean({ operation: 'drop_duplicates' })}
                                disabled={loading}
                                className="flex items-center justify-center gap-1.5 px-3 py-2 bg-slate-100 text-slate-600 rounded-lg text-xs font-medium hover:bg-slate-200 disabled:opacity-50 transition-colors"
                            >
                                <Copy size={14} /> Drop Dupes
                            </button>
                        </div>
                    </div>
                </div>

                <div className="space-y-3">
                    {/* Accordion Item: Missing Values */}
                    <div className="border border-slate-200 rounded-xl bg-white overflow-hidden">
                        <button className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-slate-100 transition-colors text-left" onClick={() => document.getElementById('section-missing').classList.toggle('hidden')}>
                            <span className="font-semibold text-slate-700 flex items-center gap-2"><Filter size={18} className="text-blue-500" /> Missing Data</span>
                            <span className="text-slate-400 text-xs">▼</span>
                        </button>
                        <div id="section-missing" className="p-4 space-y-4">
                            {/* Drop Nulls */}
                            <div>
                                <label className="block text-xs text-slate-500 mb-1">Remove Empty Rows</label>
                                <ActionButton
                                    icon={Trash2}
                                    label={`Drop Nulls ${selectedColumns.length > 0 ? `(${selectedColumns.length} Cols)` : '(All)'}`}
                                    onClick={() => handleClean({
                                        operation: 'dropna',
                                        columns: selectedColumns
                                    })}
                                    disabled={loading}
                                />
                            </div>

                            {/* Fill Missing */}
                            <div className="pt-2 border-t border-slate-100">
                                <label className="block text-xs text-slate-500 mb-2">Fill Missing Values</label>
                                <div className="flex gap-2 mb-2">
                                    <select
                                        className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-sm outline-none focus:border-primary"
                                        value={fillStrategy}
                                        onChange={(e) => setFillStrategy(e.target.value)}
                                    >
                                        <option value="mean">Mean (Avg)</option>
                                        <option value="median">Median (Mid)</option>
                                        <option value="mode">Mode (Common)</option>
                                    </select>
                                </div>
                                <ActionButton
                                    icon={Check}
                                    label={`Fill with ${fillStrategy} ${selectedColumns.length > 0 ? `(${selectedColumns.length} Cols)` : '(All)'}`}
                                    onClick={() => handleClean({
                                        operation: 'fillna',
                                        strategy: fillStrategy,
                                        columns: selectedColumns
                                    })}
                                    disabled={loading}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Accordion Item: Structure */}
                    <div className="border border-slate-200 rounded-xl bg-white overflow-hidden">
                        <button className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-slate-100 transition-colors text-left" onClick={() => document.getElementById('section-structure').classList.toggle('hidden')}>
                            <span className="font-semibold text-slate-700 flex items-center gap-2"><Edit2 size={18} className="text-orange-500" /> Structure & Rename</span>
                            <span className="text-slate-400 text-xs">▼</span>
                        </button>
                        <div id="section-structure" className="p-4 hidden">
                            <div className="space-y-2 mb-3">
                                <label className="block text-xs text-slate-500">Rename Column</label>
                                <select
                                    className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                    value={renameMap.old}
                                    onChange={(e) => setRenameMap({ ...renameMap, old: e.target.value })}
                                >
                                    <option value="">Select Column...</option>
                                    {columns.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                                <input
                                    type="text"
                                    placeholder="New Name"
                                    className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                    value={renameMap.new}
                                    onChange={(e) => setRenameMap({ ...renameMap, new: e.target.value })}
                                />
                                <ActionButton
                                    icon={Check}
                                    label="Rename"
                                    onClick={() => {
                                        if (!renameMap.old || !renameMap.new) return;
                                        handleClean({
                                            operation: 'rename_columns',
                                            rename_map: { [renameMap.old]: renameMap.new }
                                        });
                                        setRenameMap({ old: '', new: '' });
                                    }}
                                    disabled={loading || !renameMap.old || !renameMap.new}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Accordion Item: Advanced */}
                    <div className="border border-slate-200 rounded-xl bg-white overflow-hidden">
                        <button className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-slate-100 transition-colors text-left" onClick={() => document.getElementById('section-advanced').classList.toggle('hidden')}>
                            <span className="font-semibold text-slate-700 flex items-center gap-2"><ShieldAlert size={18} className="text-purple-500" /> Advanced</span>
                            <span className="text-slate-400 text-xs">▼</span>
                        </button>
                        <div id="section-advanced" className="p-4 space-y-4 hidden">
                            <div>
                                <h4 className="text-xs font-bold text-slate-600 mb-2 flex items-center gap-1"><ShieldAlert size={12} /> Outliers</h4>
                                <ActionButton
                                    icon={ShieldAlert}
                                    label={`Remove Outliers (IQR) ${selectedColumns.length > 0 ? `(${selectedColumns.length})` : '(All Num)'}`}
                                    onClick={() => handleClean({ operation: 'remove_outliers', columns: selectedColumns })}
                                    disabled={loading}
                                />
                            </div>

                            <div className="pt-2 border-t border-slate-100">
                                <h4 className="text-xs font-bold text-slate-600 mb-2 flex items-center gap-1"><Binary size={12} /> Encoding</h4>
                                <select
                                    className="w-full p-2 mb-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                    value={encodeStrategy}
                                    onChange={(e) => setEncodeStrategy(e.target.value)}
                                >
                                    <option value="label">Label Encoding</option>
                                    <option value="onehot">One-Hot Encoding</option>
                                </select>
                                <ActionButton
                                    icon={Binary}
                                    label={`Encode ${selectedColumns.length > 0 ? `(${selectedColumns.length})` : '(Selection Req)'}`}
                                    onClick={() => {
                                        handleClean({ operation: 'encode_columns', columns: selectedColumns, strategy: encodeStrategy });
                                        setSelectedColumns([]);
                                    }}
                                    disabled={loading || selectedColumns.length === 0}
                                />
                            </div>

                            <div className="pt-2 border-t border-slate-100">
                                <h4 className="text-xs font-bold text-slate-600 mb-2 flex items-center gap-1"><Scaling size={12} /> Normalization</h4>
                                <select
                                    className="w-full p-2 mb-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                    value={scaleStrategy}
                                    onChange={(e) => setScaleStrategy(e.target.value)}
                                >
                                    <option value="minmax">MinMax Scaling (0-1)</option>
                                    <option value="standard">Standard Scaling (Z)</option>
                                </select>
                                <ActionButton
                                    icon={Scaling}
                                    label={`Scale ${selectedColumns.length > 0 ? `(${selectedColumns.length})` : '(Selection Req)'}`}
                                    onClick={() => {
                                        handleClean({ operation: 'normalize', columns: selectedColumns, strategy: scaleStrategy });
                                        setSelectedColumns([]);
                                    }}
                                    disabled={loading || selectedColumns.length === 0}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                <ActionButton
                    icon={Download}
                    label="Export Cleaned CSV"
                    onClick={exportData}
                />

                <div className="flex gap-2">
                    <button
                        onClick={handleUndo}
                        disabled={loading}
                        className="flex-1 flex items-center justify-center gap-2 p-3 bg-white border border-slate-200 rounded-xl text-slate-600 hover:bg-slate-50 hover:text-slate-800 disabled:opacity-50"
                        title="Undo Last Action"
                    >
                        <Undo2 size={20} /> Undo
                    </button>
                    <button
                        onClick={handleRedo}
                        disabled={loading}
                        className="flex-1 flex items-center justify-center gap-2 p-3 bg-white border border-slate-200 rounded-xl text-slate-600 hover:bg-slate-50 hover:text-slate-800 disabled:opacity-50"
                        title="Redo Action"
                    >
                        <Redo2 size={20} /> Redo
                    </button>
                </div>
            </div>

            {/* Main Content: Data Summary/Preview */}
            <div className={`flex-1 relative ${!hasData ? 'blur-sm pointer-events-none opacity-50' : ''}`}>
                {loading && (
                    <div className="absolute inset-0 z-40 bg-white/50 backdrop-blur-sm flex items-center justify-center rounded-xl">
                        <div className="bg-white p-4 rounded-xl shadow-lg flex items-center gap-3 border border-slate-100">
                            <Loader className="animate-spin text-primary" size={24} />
                            <span className="font-medium text-slate-700">Processing...</span>
                        </div>
                    </div>
                )}
                <h2 className="text-xl font-bold text-slate-800 mb-4">Current Data State</h2>

                {/* Missing Values Chart/List */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6"> {/* Increased cols for better layout */}
                    {Object.entries(missing_values).map(([col, count]) => {
                        if (count === 0) return null;
                        return (
                            <div key={col} className="bg-red-50 p-3 rounded-lg border border-red-100 flex justify-between items-center">
                                <span className="font-medium text-red-800 text-sm truncate w-24" title={col}>{col}</span>
                                <span className="bg-red-200 text-red-800 text-xs px-2 py-1 rounded-full">{count} missing</span>
                            </div>
                        )
                    })}
                    {Object.values(missing_values).every(v => v === 0) && (
                        <div className="col-span-full p-4 bg-green-50 text-green-700 rounded-xl flex items-center gap-2 border border-green-100">
                            <CheckCircleIcon size={20} />
                            <span>No missing values found! Great job.</span>
                        </div>
                    )}
                </div>

                {/* Simple Table Preview (Reuse form Dashboard or abstract) */}
                {/* For now just show a simple JSON dump of stats for space efficiency or a similar table */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                    <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50 flex justify-between">
                        <h3 className="font-bold text-slate-800">Preview</h3>
                        <span className="text-sm text-slate-500">Showing first 50 rows</span>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr>
                                    {columns.map((col) => (
                                        <th key={col} className="px-6 py-3 bg-slate-50 text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-200 whitespace-nowrap">
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {(useData().dataPreview || []).map((row, i) => (
                                    <tr key={i} className="hover:bg-slate-50/50 transition-colors">
                                        {columns.map((col) => (
                                            <td key={col} className="px-6 py-3 text-sm text-slate-600 whitespace-nowrap max-w-xs truncate">
                                                {row[col] !== null ? String(row[col]) : <span className="text-slate-300 italic">null</span>}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div >
    );
}

const CheckCircleIcon = ({ size }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
);
