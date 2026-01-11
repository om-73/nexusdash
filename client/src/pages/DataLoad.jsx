import React, { useCallback, useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader, Database, X } from 'lucide-react';
import { uploadFile, loadData, connectDatabase } from '../services/api';
import { useData } from '../context/DataContext';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../context/ToastContext';

export default function DataLoad() {
    const [dragActive, setDragActive] = useState(false);
    const { setDataSummary, setDataPreview, setLoading, loading, setError, error } = useData();
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
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = async (file) => {
        setLoading(true);
        setError(null);
        try {
            // 1. Upload File
            const uploadRes = await uploadFile(file);

            // 2. Load Data Preview
            const dataRes = await loadData(uploadRes.filePath, 'csv'); // Assume CSV for now

            setDataSummary(dataRes);
            setDataPreview(dataRes.preview);

            navigate('/');
            addToast('File uploaded successfully', 'success');
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.message || err.message || 'Failed to upload/load file');
            addToast('Failed to upload file', 'error');
        } finally {
            setLoading(false);
        }
    };

    const handleDbConnect = async () => {
        setLoading(true);
        setError(null);
        try {
            const payload = {
                db_type: dbType,
                connection_string: connString,
                query: dbType === 'postgresql' ? query : undefined,
                collection: dbType === 'mongodb' ? collection : undefined
            };

            const dataRes = await connectDatabase(payload);
            setDataSummary(dataRes);
            setDataPreview(dataRes.preview);
            setShowDbModal(false);
            setShowDbModal(false);
            navigate('/');
            addToast('Connected to database successfully', 'success');
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.error || err.response?.data?.detail || 'Failed to connect to database');
            addToast('Failed to connect to database', 'error');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-8 max-w-4xl mx-auto">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold mb-2 text-slate-800">Connect Data</h1>
                    <p className="text-slate-500">Upload your dataset (CSV/Excel) or connect to a database.</p>
                </div>
                <button
                    onClick={() => setShowDbModal(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 font-medium text-slate-700 shadow-sm"
                >
                    <Database size={18} /> Connect Database
                </button>
            </div>

            <div
                className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-200 
          ${dragActive ? 'border-primary bg-primary/5' : 'border-slate-300 hover:border-primary/50'}
          ${loading ? 'opacity-50 pointer-events-none' : ''}
        `}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleChange}
                />

                <div className="flex flex-col items-center justify-center gap-4">
                    <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mb-2">
                        {loading ? <Loader className="animate-spin" size={32} /> : <Upload size={32} />}
                    </div>

                    {loading ? (
                        <div>
                            <h3 className="text-xl font-bold text-slate-700">Processing Data...</h3>
                            <p className="text-slate-500">Please wait while we inspect your file.</p>
                        </div>
                    ) : (
                        <>
                            <h3 className="text-xl font-bold text-slate-700">Drag & Drop your file here</h3>
                            <p className="text-slate-500">or</p>
                            <label
                                htmlFor="file-upload"
                                className="px-6 py-2 bg-slate-900 text-white rounded-lg font-medium cursor-pointer hover:bg-slate-800 transition-colors"
                            >
                                Browse Files
                            </label>
                            <p className="text-xs text-slate-400 mt-4">Supports CSV, Excel (max 50MB)</p>
                        </>
                    )}
                </div>
            </div>

            {/* Database Modal */}
            {showDbModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-sm">
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
                                    className="w-full p-2 border border-slate-300 rounded-lg"
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
                                        className="w-full p-2 border border-slate-300 rounded-lg"
                                        placeholder="users"
                                    />
                                </div>
                            )}

                        </div>

                        <div className="p-6 border-t border-slate-100 bg-slate-50 flex justify-end gap-3">
                            <button
                                onClick={() => setShowDbModal(false)}
                                className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-200 rounded-lg"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleDbConnect}
                                disabled={loading}
                                className="px-6 py-2 bg-primary text-white rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50 flex items-center gap-2"
                            >
                                {loading && <Loader size={16} className="animate-spin" />}
                                {loading ? 'Connecting...' : 'Connect'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {error && (
                <div className="mt-6 p-4 bg-red-50 text-red-600 rounded-xl flex items-center gap-3 border border-red-100">
                    <AlertCircle size={20} />
                    <span>{error}</span>
                </div>
            )}
        </div>
    );
}
