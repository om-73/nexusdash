import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useData } from '../context/DataContext';
import { trainModel, configureAI, downloadModel } from '../services/api';
import { Play, CheckCircle, BarChart2, Sparkles, Loader, Download, Award, Zap, ArrowRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Model() {
    const { dataSummary, modelResult } = useData();
    const navigate = useNavigate();
    const [config, setConfig] = useState({
        problem_type: 'regression',
        target_column: '',
        feature_columns: [],
        algorithms: ['linear']
    });
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [aiGoal, setAiGoal] = useState('');
    const [aiLoading, setAiLoading] = useState(false);

    useEffect(() => {
        if (modelResult) {
            if (modelResult.target_column) {
                setConfig(prev => ({
                    ...prev,
                    target_column: modelResult.target_column,
                    problem_type: modelResult.problem_type || 'regression'
                }));
            }
        }
    }, [modelResult]);

    if (!dataSummary) {
        return <div className="p-8 text-center text-slate-500">Please load data first.</div>;
    }

    const { columns } = dataSummary;

    const handleAIConfigure = async (e) => {
        e.preventDefault();
        if (!aiGoal.trim()) return;
        setAiLoading(true);
        try {
            const res = await configureAI(aiGoal);

            if (res.error) {
                alert(`AI Error: ${res.error}`);
                return;
            }

            if (res.target_column) {
                setConfig(prev => ({
                    ...prev,
                    target_column: res.target_column,
                    problem_type: res.problem_type?.toLowerCase() || 'regression',
                    algorithm: res.algorithm?.toLowerCase() === 'random forest' ? 'rf' : 'linear'
                }));
            }
            alert(`AI Suggestion: ${res.reasoning}`);
        } catch (err) {
            console.error("AI Config error:", err);
            alert("Failed to auto-configure. Check console.");
        } finally {
            setAiLoading(false);
        }
    };

    const handleFeatureToggle = (col) => {
        const current = config.feature_columns;
        if (current.includes(col)) {
            setConfig({ ...config, feature_columns: current.filter(c => c !== col) });
        } else {
            setConfig({ ...config, feature_columns: [...current, col] });
        }
    };

    const handleTrain = async () => {
        if (!config.target_column || config.feature_columns.length === 0) {
            setError("Please select a target and at least one feature.");
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const res = await trainModel(config);
            setResults(res);
            if (res.problem_type && res.problem_type !== config.problem_type) {
                setConfig(prev => ({ ...prev, problem_type: res.problem_type }));
            }
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.error || err.response?.data?.detail || err.message || "Training failed");
        } finally {
            setLoading(false);
        }
    };

    const handleDownloadModel = async () => {
        try {
            const blob = await downloadModel();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'model.pkl';
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            console.error("Download failed", err);
            alert("Failed to download model");
        }
    };

    return (
        <div className="p-8 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Configuration Panel */}
            <div className="lg:col-span-1 space-y-6">

                {/* AI Assistant Section */}
                <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-6 text-white shadow-lg">
                    <h2 className="text-sm font-bold mb-2 flex items-center">
                        <Sparkles className="mr-2 text-yellow-300" size={16} />
                        Auto-Configure
                    </h2>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={aiGoal}
                            onChange={e => setAiGoal(e.target.value)}
                            placeholder="e.g. Predict price"
                            className="flex-1 w-full px-3 py-2 rounded-lg text-slate-900 border-none text-sm focus:ring-2 focus:ring-yellow-300 outline-none"
                        />
                        <button
                            onClick={handleAIConfigure}
                            disabled={aiLoading}
                            className="bg-white/20 hover:bg-white/30 text-white font-medium px-3 py-2 rounded-lg transition-all flex items-center justify-center disabled:opacity-50"
                        >
                            {aiLoading ? <Loader className="animate-spin" size={16} /> : <Sparkles size={16} />}
                        </button>
                    </div>
                </div>

                <div>
                    <h1 className="text-3xl font-bold text-slate-800 mb-2">Machine Learning</h1>
                    <p className="text-slate-500">Train predictive models.</p>
                </div>

                <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Problem Type</label>
                        <select
                            className="w-full p-2 border border-slate-300 rounded-lg"
                            value={config.problem_type}
                            onChange={(e) => setConfig({
                                ...config,
                                problem_type: e.target.value,
                                algorithms: e.target.value === 'regression' ? ['linear'] : ['logistic']
                            })}
                        >
                            <option value="regression">Regression (Predict Number)</option>
                            <option value="classification">Classification (Predict Category)</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Algorithms (Select Multiple)</label>
                        <div className="border border-slate-300 rounded-lg p-3 max-h-60 overflow-y-auto bg-white space-y-4">
                            {config.problem_type === 'regression' ? (
                                <>
                                    <div>
                                        <p className="text-xs font-bold text-slate-400 uppercase mb-1">Linear Models</p>
                                        <div className="space-y-1">
                                            {['linear', 'ridge', 'lasso'].map(algo => (
                                                <label key={algo} className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                                                    <input
                                                        type="checkbox"
                                                        checked={config.algorithms.includes(algo)}
                                                        onChange={() => {
                                                            const current = config.algorithms;
                                                            if (current.includes(algo)) setConfig({ ...config, algorithms: current.filter(a => a !== algo) });
                                                            else setConfig({ ...config, algorithms: [...current, algo] });
                                                        }}
                                                        className="rounded text-primary focus:ring-primary"
                                                    />
                                                    <span className="text-sm">
                                                        {algo === 'linear' && 'Linear Regression'}
                                                        {algo === 'ridge' && 'Ridge Regression (L2)'}
                                                        {algo === 'lasso' && 'Lasso Regression (L1)'}
                                                    </span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-slate-400 uppercase mb-1">Tree Based</p>
                                        <div className="space-y-1">
                                            {['dt', 'rf', 'gbr', 'ada'].map(algo => (
                                                <label key={algo} className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                                                    <input
                                                        type="checkbox"
                                                        checked={config.algorithms.includes(algo)}
                                                        onChange={() => {
                                                            const current = config.algorithms;
                                                            if (current.includes(algo)) setConfig({ ...config, algorithms: current.filter(a => a !== algo) });
                                                            else setConfig({ ...config, algorithms: [...current, algo] });
                                                        }}
                                                        className="rounded text-primary focus:ring-primary"
                                                    />
                                                    <span className="text-sm">
                                                        {algo === 'dt' && 'Decision Tree'}
                                                        {algo === 'rf' && 'Random Forest'}
                                                        {algo === 'gbr' && 'Gradient Boosting'}
                                                        {algo === 'ada' && 'AdaBoost'}
                                                    </span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-slate-400 uppercase mb-1">Other</p>
                                        <div className="space-y-1">
                                            <label className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                                                <input
                                                    type="checkbox"
                                                    checked={config.algorithms.includes('svr')}
                                                    onChange={() => {
                                                        const current = config.algorithms;
                                                        if (current.includes('svr')) setConfig({ ...config, algorithms: current.filter(a => a !== 'svr') });
                                                        else setConfig({ ...config, algorithms: [...current, 'svr'] });
                                                    }}
                                                    className="rounded text-primary focus:ring-primary"
                                                />
                                                <span className="text-sm">Support Vector Regression (SVR)</span>
                                            </label>
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div>
                                        <p className="text-xs font-bold text-slate-400 uppercase mb-1">Linear/Probabilistic</p>
                                        <div className="space-y-1">
                                            {['logistic', 'nb'].map(algo => (
                                                <label key={algo} className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                                                    <input
                                                        type="checkbox"
                                                        checked={config.algorithms.includes(algo)}
                                                        onChange={() => {
                                                            const current = config.algorithms;
                                                            if (current.includes(algo)) setConfig({ ...config, algorithms: current.filter(a => a !== algo) });
                                                            else setConfig({ ...config, algorithms: [...current, algo] });
                                                        }}
                                                        className="rounded text-primary focus:ring-primary"
                                                    />
                                                    <span className="text-sm">
                                                        {algo === 'logistic' && 'Logistic Regression'}
                                                        {algo === 'nb' && 'Naive Bayes (Gaussian)'}
                                                    </span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-slate-400 uppercase mb-1">Tree Based</p>
                                        <div className="space-y-1">
                                            {['dt', 'rf', 'gbc', 'ada'].map(algo => (
                                                <label key={algo} className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                                                    <input
                                                        type="checkbox"
                                                        checked={config.algorithms.includes(algo)}
                                                        onChange={() => {
                                                            const current = config.algorithms;
                                                            if (current.includes(algo)) setConfig({ ...config, algorithms: current.filter(a => a !== algo) });
                                                            else setConfig({ ...config, algorithms: [...current, algo] });
                                                        }}
                                                        className="rounded text-primary focus:ring-primary"
                                                    />
                                                    <span className="text-sm">
                                                        {algo === 'dt' && 'Decision Tree'}
                                                        {algo === 'rf' && 'Random Forest'}
                                                        {algo === 'gbc' && 'Gradient Boosting'}
                                                        {algo === 'ada' && 'AdaBoost'}
                                                    </span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-slate-400 uppercase mb-1">Distance/Margin</p>
                                        <div className="space-y-1">
                                            {['knn', 'svm'].map(algo => (
                                                <label key={algo} className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                                                    <input
                                                        type="checkbox"
                                                        checked={config.algorithms.includes(algo)}
                                                        onChange={() => {
                                                            const current = config.algorithms;
                                                            if (current.includes(algo)) setConfig({ ...config, algorithms: current.filter(a => a !== algo) });
                                                            else setConfig({ ...config, algorithms: [...current, algo] });
                                                        }}
                                                        className="rounded text-primary focus:ring-primary"
                                                    />
                                                    <span className="text-sm">
                                                        {algo === 'knn' && 'K-Nearest Neighbors (KNN)'}
                                                        {algo === 'svm' && 'Support Vector Machine (SVM)'}
                                                    </span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Target Variable (y)</label>
                        <select
                            className="w-full p-2 border border-slate-300 rounded-lg"
                            value={config.target_column}
                            onChange={(e) => setConfig({ ...config, target_column: e.target.value })}
                        >
                            <option value="">Select Target...</option>
                            {columns.map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Features (X)</label>
                        <div className="max-h-48 overflow-y-auto border border-slate-200 p-2 rounded-lg bg-slate-50 space-y-1">
                            {columns.map(col => (
                                col !== config.target_column && (
                                    <label key={col} className="flex items-center gap-2 text-sm cursor-pointer p-1 hover:bg-slate-100 rounded">
                                        <input
                                            type="checkbox"
                                            checked={config.feature_columns.includes(col)}
                                            onChange={() => handleFeatureToggle(col)}
                                            className="rounded text-primary"
                                        />
                                        {col}
                                    </label>
                                )
                            ))}
                        </div>
                    </div>

                    <button
                        onClick={handleTrain}
                        disabled={loading}
                        className={`w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 text-white
              ${loading ? 'bg-slate-400' : 'bg-primary hover:bg-blue-600 shadow-lg shadow-primary/25'}
            `}
                    >
                        {loading ? 'Training...' : <><Play size={20} /> Train Model</>}
                    </button>

                    {error && <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm">{error}</div>}
                </div>
            </div>

            {/* Results Panel */}
            <div className="lg:col-span-2">
                {!results && !modelResult?.champion ? (
                    <div className="h-full flex items-center justify-center border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/50 p-12 text-center text-slate-400">
                        <div>
                            <BarChart2 size={48} className="mx-auto mb-4 opacity-50" />
                            <p>Configure and train a model to see results here.</p>
                        </div>
                    </div>
                ) : !results && modelResult?.champion ? (
                    <div className="space-y-6">
                        {/* Synced Champion Model Banner */}
                        <div className="bg-gradient-to-br from-indigo-900 to-slate-900 text-white p-6 rounded-2xl shadow-lg border border-indigo-800/40 space-y-4">
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-indigo-800/60 pb-3">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-amber-400/20 text-amber-400 border border-amber-400/30 flex items-center justify-center">
                                        <Award size={22} />
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <h3 className="font-bold text-base">Champion Model Synced from Autonomous Agent</h3>
                                            <span className="bg-emerald-500/20 text-emerald-300 text-[10px] font-mono px-2 py-0.5 rounded-full border border-emerald-500/30">
                                                Active Champion
                                            </span>
                                        </div>
                                        <p className="text-xs text-indigo-200 mt-0.5">
                                            Algorithm: <span className="font-bold text-white">{modelResult.champion.name}</span> • Target: <code className="text-amber-300 font-mono">{modelResult.target_column}</code>
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={handleDownloadModel}
                                        className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold py-2 px-3 rounded-lg flex items-center gap-1.5 transition"
                                    >
                                        <Download size={14} /> Download .pkl
                                    </button>
                                    <button
                                        onClick={() => navigate('/predict')}
                                        className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold py-2 px-3 rounded-lg flex items-center gap-1.5 transition"
                                    >
                                        <Zap size={14} /> Test in Predictions
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                                <div className="bg-slate-800/80 rounded-xl p-3 border border-slate-700">
                                    <span className="text-[10px] uppercase font-bold text-slate-400">{modelResult.champion.metric_name || 'Primary Metric'}</span>
                                    <p className="text-2xl font-black text-emerald-400 mt-0.5">{modelResult.champion.metric_value}</p>
                                </div>
                                {modelResult.champion.secondary_metric_name && (
                                    <div className="bg-slate-800/80 rounded-xl p-3 border border-slate-700">
                                        <span className="text-[10px] uppercase font-bold text-slate-400">{modelResult.champion.secondary_metric_name}</span>
                                        <p className="text-2xl font-black text-white mt-0.5">{modelResult.champion.secondary_metric_value}</p>
                                    </div>
                                )}
                                <div className="bg-slate-800/80 rounded-xl p-3 border border-slate-700">
                                    <span className="text-[10px] uppercase font-bold text-slate-400">Inference Latency</span>
                                    <p className="text-2xl font-black text-indigo-300 mt-0.5">{modelResult.champion.latency_ms || 0} ms</p>
                                </div>
                            </div>
                        </div>

                        {/* Tournament Leaderboard */}
                        {modelResult.tournament && modelResult.tournament.length > 0 && (
                            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                                <h3 className="font-bold text-slate-800 text-sm uppercase tracking-wider flex items-center gap-2">
                                    <Award size={16} className="text-indigo-600" />
                                    AutoML Tournament Leaderboard
                                </h3>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-xs text-left">
                                        <thead className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                                            <tr>
                                                <th className="px-3 py-2.5">Rank & Model</th>
                                                <th className="px-3 py-2.5">{modelResult.champion.metric_name || 'Score'}</th>
                                                <th className="px-3 py-2.5">Secondary Metric</th>
                                                <th className="px-3 py-2.5">Train Latency</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-100">
                                            {modelResult.tournament.map((m, idx) => (
                                                <tr key={idx} className={m.is_champion ? "bg-emerald-50/70 font-semibold text-emerald-950" : ""}>
                                                    <td className="px-3 py-2.5 flex items-center gap-1.5">
                                                        <span>#{idx + 1}</span>
                                                        <span>{m.model_name}</span>
                                                        {m.is_champion && <span className="bg-emerald-200 text-emerald-800 text-[9px] px-1.5 py-0.2 rounded font-bold">CHAMPION</span>}
                                                    </td>
                                                    <td className="px-3 py-2.5 font-bold text-emerald-600">{m.primary_metric}</td>
                                                    <td className="px-3 py-2.5 text-slate-600">{m.secondary_metric || '-'}</td>
                                                    <td className="px-3 py-2.5 font-mono text-slate-500">{m.latency_ms}ms</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="space-y-6">
                        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                            <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                                <CheckCircle className="text-emerald-500" />
                                Model Performance
                            </h2>
                            {results.download_available && (
                                <button
                                    onClick={handleDownloadModel}
                                    className="mb-4 px-4 py-2 bg-slate-800 text-white rounded-lg flex items-center gap-2 hover:bg-slate-700 text-sm"
                                >
                                    <Download size={16} /> Download .pkl
                                </button>
                            )}
                            <div className="grid grid-cols-2 gap-4">
                                {Object.entries(results.metrics).map(([key, val]) => (
                                    <div key={key} className="bg-slate-50 p-4 rounded-lg">
                                        <p className="text-xs text-slate-500 uppercase font-semibold">{key}</p>
                                        <p className="text-2xl font-bold text-slate-800">{val.toFixed(4)}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* AutoML Comparison Section */}
                        {results.comparison && (
                            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                <h3 className="font-bold text-slate-800 mb-4">Model Comparison</h3>
                                <div className="overflow-x-auto mb-6">
                                    <table className="w-full text-sm text-left">
                                        <thead className="bg-slate-50 text-slate-500 font-semibold">
                                            <tr>
                                                <th className="px-4 py-2">Algorithm</th>
                                                {Object.keys(results.comparison[0].metrics).map(m => <th key={m} className="px-4 py-2 uppercase">{m}</th>)}
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-100">
                                            {results.comparison.map((res, idx) => (
                                                <tr key={idx} className={res.algorithm === results.best_algorithm ? "bg-green-50" : ""}>
                                                    <td className="px-4 py-2 font-medium">
                                                        {res.algorithm === results.best_algorithm && "🏆 "}
                                                        {res.algorithm}
                                                    </td>
                                                    {Object.values(res.metrics).map((val, i) => (
                                                        <td key={i} className="px-4 py-2">{val.toFixed(4)}</td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                                        <BarChart
                                            data={results.comparison.map(c => ({
                                                name: c.algorithm,
                                                score: config.problem_type === 'regression' ? c.metrics.r2 : c.metrics.accuracy
                                            }))}
                                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip />
                                            <Bar dataKey="score" fill="#6366f1" name={config.problem_type === 'regression' ? "R2 Score" : "Accuracy"} radius={[4, 4, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* Classification Report */}
                        {results.classification_report && config.problem_type === 'classification' && (
                            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                <h3 className="font-bold text-slate-800 mb-4">Classification Report</h3>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                        <thead className="bg-slate-50 text-slate-600 font-semibold">
                                            <tr>
                                                <th className="px-4 py-3 text-left">Class</th>
                                                <th className="px-4 py-3 text-center">Precision</th>
                                                <th className="px-4 py-3 text-center">Recall</th>
                                                <th className="px-4 py-3 text-center">F1-Score</th>
                                                <th className="px-4 py-3 text-center">Support</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-100">
                                            {Object.entries(results.classification_report)
                                                .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                                                .map(([className, metrics]) => (
                                                    <tr key={className} className="hover:bg-slate-50">
                                                        <td className="px-4 py-3 font-medium text-slate-800">Class {className}</td>
                                                        <td className="px-4 py-3 text-center">{metrics.precision?.toFixed(3) || 'N/A'}</td>
                                                        <td className="px-4 py-3 text-center">{metrics.recall?.toFixed(3) || 'N/A'}</td>
                                                        <td className="px-4 py-3 text-center">{metrics['f1-score']?.toFixed(3) || 'N/A'}</td>
                                                        <td className="px-4 py-3 text-center">{metrics.support || 'N/A'}</td>
                                                    </tr>
                                                ))}
                                            {results.classification_report['weighted avg'] && (
                                                <>
                                                    <tr className="bg-blue-50 font-semibold">
                                                        <td className="px-4 py-3 text-slate-800">Weighted Average</td>
                                                        <td className="px-4 py-3 text-center">{results.classification_report['weighted avg'].precision?.toFixed(3)}</td>
                                                        <td className="px-4 py-3 text-center">{results.classification_report['weighted avg'].recall?.toFixed(3)}</td>
                                                        <td className="px-4 py-3 text-center">{results.classification_report['weighted avg']['f1-score']?.toFixed(3)}</td>
                                                        <td className="px-4 py-3 text-center">{results.classification_report['weighted avg'].support}</td>
                                                    </tr>
                                                </>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                                <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                                    <p className="text-xs text-blue-800">
                                        <strong>Precision:</strong> Of all predicted positives, how many were correct? |
                                        <strong> Recall:</strong> Of all actual positives, how many were found? |
                                        <strong> F1-Score:</strong> Harmonic mean of precision and recall
                                    </p>
                                </div>
                            </div>
                        )}


                        {results.feature_importance && Object.keys(results.feature_importance).length > 0 && (
                            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                <h3 className="font-bold text-slate-800 mb-4">Feature Importance</h3>
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                                        <BarChart
                                            data={Object.entries(results.feature_importance).map(([k, v]) => ({ name: k, value: v }))}
                                            layout="vertical"
                                            margin={{ left: 20 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                            <XAxis type="number" />
                                            <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
                                            <Tooltip />
                                            <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                            <h3 className="font-bold text-slate-800 mb-4">Predictions Preview</h3>
                            <div className="flex gap-4 text-sm">
                                <div className="flex-1">
                                    <p className="font-semibold mb-2">Actual</p>
                                    <div className="bg-slate-50 p-2 rounded font-mono text-slate-600">
                                        {results.actual_preview.join(', ')}
                                    </div>
                                </div>
                                <div className="flex-1">
                                    <p className="font-semibold mb-2 text-primary">Predicted</p>
                                    <div className="bg-blue-50 p-2 rounded font-mono text-blue-600">
                                        {results.predictions_preview.map(n => typeof n === 'number' ? n.toFixed(2) : n).join(', ')}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
