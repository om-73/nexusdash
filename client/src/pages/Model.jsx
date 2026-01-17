import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import { trainModel, configureAI, downloadModel } from '../services/api';
import { Play, CheckCircle, BarChart2, Sparkles, Loader, Download } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Model() {
    const { dataSummary } = useData();
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
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || "Training failed");
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
                {!results ? (
                    <div className="h-full flex items-center justify-center border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/50 p-12 text-center text-slate-400">
                        <div>
                            <BarChart2 size={48} className="mx-auto mb-4 opacity-50" />
                            <p>Configure and train a model to see results here.</p>
                        </div>
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
                                    <ResponsiveContainer width="100%" height="100%">
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

                        {results.feature_importance && Object.keys(results.feature_importance).length > 0 && (
                            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                <h3 className="font-bold text-slate-800 mb-4">Feature Importance</h3>
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%">
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
