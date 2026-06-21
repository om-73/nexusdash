import React, { useState, useEffect } from 'react';
import { useData } from '../context/DataContext';
import {
    getEDA, cleanData, getDataQuality, addFeature, trainModel,
    getQuantile
} from '../services/api';
import {
    CheckCircle, AlertCircle, ArrowRight, ArrowLeft, Play,
    Database, Activity, Filter, Layers, Calculator, BarChart2
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell
} from 'recharts';

const STEPS = [
    { title: "Data Info", icon: Database },
    { title: "Describe", icon: Activity },
    { title: "Check Nulls", icon: AlertCircle },
    { title: "Check Duplicates", icon: Layers },
    { title: "Fill Nulls", icon: Filter },
    { title: "Recheck", icon: CheckCircle },
    { title: "Redescribe", icon: Activity },
    { title: "Create Target", icon: Calculator },
    { title: "Find Threshold", icon: BarChart2 },
    { title: "Add Feature", icon: Calculator },
    { title: "Model Metrics", icon: Play }
];

export default function Workflow() {
    const { dataSummary, setDataSummary, setDataPreview } = useData();
    const [currentStep, setCurrentStep] = useState(0);
    const [loading, setLoading] = useState(false);

    // Step Data State
    const [edaData, setEdaData] = useState(null);
    const [qualityData, setQualityData] = useState(null);
    const [targetFormula, setTargetFormula] = useState({ name: 'Target', expression: '' });
    const [featureFormula, setFeatureFormula] = useState({ name: 'NewFeature', expression: '' });
    const [modelConfig, setModelConfig] = useState({
        target: '',
        algo: 'rf', // Default Random Forest
        type: 'classification'
    });
    const [modelResults, setModelResults] = useState(null);

    // Fetch Helper
    const loadStepData = async (step) => {
        setLoading(true);
        try {
            if (step === 1 || step === 6) { // Describe / Redescribe
                const res = await getEDA();
                setEdaData(res);
            }
            if (step === 2 || step === 3 || step === 5) { // Nulls / Duplicates / Recheck
                const res = await getDataQuality();
                setQualityData(res);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (dataSummary) loadStepData(currentStep);
    }, [currentStep, dataSummary]);

    const handleNext = () => {
        if (currentStep < STEPS.length - 1) setCurrentStep(c => c + 1);
    };

    const handleBack = () => {
        if (currentStep > 0) setCurrentStep(c => c - 1);
    };

    const handleClean = async (op, params) => {
        setLoading(true);
        try {
            const res = await cleanData({ operation: op, ...params });
            setDataSummary(res);
            setDataPreview(res.preview);
            await loadStepData(currentStep); // Refresh current view
        } catch (e) {
            alert("Cleaning failed: " + e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleAddFeature = async (name, expr) => {
        if (!name || !expr) return;
        setLoading(true);
        try {
            const res = await addFeature(name, expr);
            setDataSummary(res);
            setDataPreview(res.preview);
            alert(`Feature ${name} created!`);
        } catch (e) {
            alert("Feature creation failed: " + e.response?.data?.detail);
        } finally {
            setLoading(false);
        }
    };

    const handleTrain = async () => {
        setLoading(true);
        try {
            const res = await trainModel({
                problem_type: modelConfig.type,
                target_column: modelConfig.target,
                feature_columns: [], // Auto-select all others
                algorithms: [modelConfig.algo]
            });
            setModelResults(res);
        } catch (e) {
            alert("Training failed: " + e.response?.data?.detail);
        } finally {
            setLoading(false);
        }
    };

    if (!dataSummary) return <div className="p-8 text-center">Please load a dataset first.</div>;

    const renderContent = () => {
        switch (currentStep) {
            case 0: // Data Info
                return (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold">Dataset Information</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-50 p-4 rounded">
                                <span className="block text-sm text-slate-500">Rows</span>
                                <span className="text-2xl font-bold">{dataSummary.shape[0]}</span>
                            </div>
                            <div className="bg-slate-50 p-4 rounded">
                                <span className="block text-sm text-slate-500">Columns</span>
                                <span className="text-2xl font-bold">{dataSummary.shape[1]}</span>
                            </div>
                        </div>
                        <div className="bg-white border rounded p-4">
                            <h4 className="font-semibold mb-2">Column Types</h4>
                            <div className="grid grid-cols-3 gap-2 text-sm">
                                {Object.entries(dataSummary.dtypes).map(([col, type]) => (
                                    <div key={col} className="p-2 border rounded">
                                        <span className="font-medium">{col}: </span>
                                        <span className="text-slate-500">{type}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                );
            case 1: // Describe
            case 6: { // Redescribe
                const stats = edaData?.summary_stats || {};
                const numericCols = Object.keys(stats).filter(c => stats[c].mean !== undefined);
                return (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold">{currentStep === 6 ? "Re-Descriptive" : "Descriptive"} Statistics</h3>
                        {loading ? <div className="text-center">Calculating Statistics...</div> : (
                            edaData?.description ? (
                                <div className="overflow-x-auto border rounded-xl">
                                    <table className="w-full text-sm">
                                        <thead className="bg-slate-50">
                                            <tr>
                                                <th className="p-2 text-left">Stat</th>
                                                {Object.keys(edaData.description).map(col => (
                                                    <th key={col} className="p-2">{col}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {['count', 'mean', 'std', 'min', '50%', 'max'].map(stat => (
                                                <tr key={stat} className="border-t">
                                                    <td className="p-2 font-medium bg-slate-50">{stat}</td>
                                                    {Object.values(edaData.description).map((stats, i) => (
                                                        <td key={i} className="p-2">
                                                            {typeof stats[stat] === 'number' ? stats[stat].toFixed(2) : '-'}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            ) : "No numeric data found."
                        )}
                    </div>
                );
            }
            case 2: // Check Nulls
            case 5: { // Recheck
                const missing = qualityData?.metrics?.missing_cells_pct || 0;
                const missingCounts = dataSummary.missing_values || {};
                return (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold">{currentStep === 5 ? "Re-Check" : "Check"} Null Values</h3>
                        <div className="flex items-center gap-4 bg-orange-50 p-6 rounded-xl border border-orange-100">
                            <AlertCircle size={32} className="text-orange-500" />
                            <div>
                                <p className="text-2xl font-bold text-orange-800">{missing}%</p>
                                <p className="text-orange-600">Total Missing Cells</p>
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            {Object.entries(missingCounts).map(([col, count]) => count > 0 && (
                                <div key={col} className="p-3 border border-red-200 bg-red-50 rounded flex justify-between">
                                    <span>{col}</span>
                                    <span className="font-bold text-red-600">{count} missing</span>
                                </div>
                            ))}
                            {Object.values(missingCounts).every(c => c === 0) && <p className="text-green-600 font-medium">No missing values found.</p>}
                        </div>
                    </div>
                );
            }
            case 3: // Check Duplicates
                return (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold">Check Duplicates</h3>
                        <div className="bg-blue-50 p-6 rounded-xl border border-blue-100 text-center">
                            <Layers size={48} className="mx-auto text-blue-500 mb-2" />
                            <p className="text-3xl font-bold text-blue-800">{qualityData?.metrics?.duplicate_rows_pct || 0}%</p>
                            <p className="text-blue-600">Duplicate Rows</p>
                            {qualityData?.metrics?.duplicate_rows_pct > 0 && (
                                <button
                                    onClick={() => handleClean('drop_duplicates', {})}
                                    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                                >
                                    Remove Duplicates
                                </button>
                            )}
                        </div>
                    </div>
                );
            case 4: // Fill Nulls
                return (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold">Fill Null Values</h3>
                        <div className="p-6 border rounded-xl bg-slate-50 space-y-4">
                            <p className="text-slate-600">Choose a strategy to fill missing values across the dataset.</p>
                            <div className="flex gap-4">
                                <button
                                    onClick={() => handleClean('fillna', { value: 'Unknown' })}
                                    className="px-6 py-3 bg-white border border-slate-300 rounded-lg hover:border-primary text-slate-700"
                                >
                                    Fill with "Unknown"
                                </button>
                                <button
                                    onClick={() => handleClean('fillna', { strategy: 'mean' })}
                                    className="px-6 py-3 bg-white border border-slate-300 rounded-lg hover:border-primary text-slate-700"
                                >
                                    Fill Numeric with Mean
                                </button>
                            </div>
                        </div>
                    </div>
                );
            case 7: // Create Target
            case 10: { // Add Feature
                const isTarget = currentStep === 7;
                const state = isTarget ? targetFormula : featureFormula;
                const setState = isTarget ? setTargetFormula : setFeatureFormula;

                return (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold">{isTarget ? "Create Target Variable" : "Add New Feature"}</h3>
                        <p className="text-sm text-slate-500">
                            Enter a name and Python-like expression. (e.g., <code>Sales {'>'} 500</code>)
                        </p>

                        {/* Quantile Helper for Target */}
                        {isTarget && (
                            <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                                <h4 className="font-semibold text-sm mb-2 text-blue-800">💡 Quantile Helper</h4>
                                <p className="text-xs text-blue-600 mb-3">Quickly create a binary target based on the 95th percentile (top 5%).</p>
                                <select
                                    className="w-full p-2 border rounded text-sm mb-2"
                                    onChange={async (e) => {
                                        if (!e.target.value) return;
                                        try {
                                            const res = await getQuantile(e.target.value, 0.95);
                                            setState({
                                                name: 'ghost_order',
                                                expression: `${e.target.value} >= ${res.value.toFixed(2)}`
                                            });
                                        } catch (err) {
                                            alert('Failed to calculate quantile');
                                        }
                                    }}
                                >
                                    <option value="">Select a column for 95th percentile...</option>
                                    {dataSummary.columns.filter(c => dataSummary.dtypes[c] !== 'object').map(c => (
                                        <option key={c} value={c}>{c}</option>
                                    ))}
                                </select>
                            </div>
                        )}

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
                            <div className="space-y-5 bg-white p-6 border border-slate-200 rounded-xl shadow-sm">
                                <div>
                                    <label className="block text-sm font-semibold mb-2 text-slate-700">Feature Name</label>
                                    <input
                                        type="text"
                                        value={state.name}
                                        onChange={e => setState({ ...state, name: e.target.value })}
                                        className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition-shadow"
                                        placeholder="e.g., ghost_order"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold mb-2 text-slate-700">Python Expression</label>
                                    <input
                                        type="text"
                                        value={state.expression}
                                        onChange={e => setState({ ...state, expression: e.target.value })}
                                        placeholder="e.g., Distance_km > 10"
                                        className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition-shadow"
                                    />
                                </div>
                                <button
                                    onClick={() => handleAddFeature(state.name, state.expression)}
                                    className="w-full py-3 font-bold bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:shadow-lg transition transform hover:-translate-y-0.5"
                                >
                                    {isTarget ? "Create Target" : "Create Feature"}
                                </button>
                            </div>

                            {/* Dynamic Helper Info */}
                            <div className="bg-slate-50 border border-slate-200 p-6 rounded-xl text-sm space-y-6 shadow-sm flex flex-col h-full">
                                <div>
                                    <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2"><Database size={16}/> Available Columns</h4>
                                    <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto p-2 bg-white border border-slate-100 rounded-lg">
                                        {dataSummary.columns.map(c => (
                                            <span key={c} className="bg-slate-100 border border-slate-200 px-2.5 py-1 rounded-md text-xs font-semibold text-slate-700">{c}</span>
                                        ))}
                                    </div>
                                </div>
                                <div className="border-t border-slate-200 pt-5 flex-1">
                                    <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2"><Calculator size={16}/> Example Formulas</h4>
                                    <ul className="space-y-3 text-slate-600">
                                        <li className="flex items-start gap-2">
                                            <span className="mt-1 text-blue-500"><CheckCircle size={14} /></span>
                                            <span><code className="bg-white px-2 py-1 rounded border text-indigo-600 font-mono text-xs shadow-sm">Distance_km {'>'} 10</code><br/><span className="text-xs text-slate-500">Creates a binary 1/0 flag</span></span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <span className="mt-1 text-blue-500"><CheckCircle size={14} /></span>
                                            <span><code className="bg-white px-2 py-1 rounded border text-indigo-600 font-mono text-xs shadow-sm">Time.isin(["Night"])</code><br/><span className="text-xs text-slate-500">Checks against categories</span></span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <span className="mt-1 text-blue-500"><CheckCircle size={14} /></span>
                                            <span><code className="bg-white px-2 py-1 rounded border text-indigo-600 font-mono text-xs shadow-sm">(Age {'>'} 18) & (Income {'>'} 50k)</code><br/><span className="text-xs text-slate-500">Combined logic</span></span>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            }
            case 8: // Threshold / Visualization
                return (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold">Find Threshold (Distribution)</h3>
                        <p className="text-slate-500">Analyze the distribution of your target or key features.</p>
                        {edaData?.distributions ? (
                            <div className="grid grid-cols-2 gap-4">
                                {Object.entries(edaData.distributions).slice(0, 4).map(([col, data]) => (
                                    <div key={col} className="h-48 border p-2 rounded">
                                        <p className="text-center font-medium mb-2">{col}</p>
                                        <ResponsiveContainer width="100%" height="90%" minWidth={0}>
                                            <BarChart data={data}>
                                                <XAxis dataKey="range" hide />
                                                <Tooltip />
                                                <Bar dataKey="count" fill="#8884d8" />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                ))}
                            </div>
                        ) : <button onClick={() => loadStepData(1)} className="text-primary underline">Load Distributions</button>}
                    </div>
                );
            case 11: // Model
                return (
                    <div className="space-y-6">
                        <h3 className="text-xl font-bold">Model Metrics</h3>

                        <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-xl">
                            <div>
                                <label className="block text-sm font-medium mb-1">Target</label>
                                <select
                                    value={modelConfig.target}
                                    onChange={e => setModelConfig({ ...modelConfig, target: e.target.value })}
                                    className="w-full p-2 border rounded"
                                >
                                    <option value="">Select Target...</option>
                                    {dataSummary.columns.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1">Algorithm</label>
                                <select
                                    value={modelConfig.algo}
                                    onChange={e => setModelConfig({ ...modelConfig, algo: e.target.value })}
                                    className="w-full p-2 border rounded"
                                >
                                    <option value="logistic">Logistic Regression</option>
                                    <option value="rf">Random Forest</option>
                                    <option value="svm">SVM</option>
                                    <option value="dt">Decision Tree</option>
                                </select>
                            </div>
                        </div>

                        <button
                            onClick={handleTrain}
                            disabled={!modelConfig.target || loading}
                            className="px-6 py-3 bg-green-600 text-white rounded-xl font-bold hover:bg-green-700 w-full"
                        >
                            {loading ? "Training..." : "Train & Get Metrics"}
                        </button>

                        {modelResults && (
                            <div className="space-y-6 animate-fade-in">
                                <div className="bg-white p-6 border rounded-xl shadow-sm">
                                    <h4 className="font-bold text-lg mb-4">Classification Report</h4>

                                    {/* Summary Cards */}
                                    <div className="grid grid-cols-4 gap-4 mb-6">
                                        <div className="p-3 bg-blue-50 rounded text-center">
                                            <div className="text-xs text-blue-500 uppercase">Accuracy</div>
                                            <div className="text-xl font-bold">{modelResults.metrics.accuracy.toFixed(2)}</div>
                                        </div>
                                        <div className="p-3 bg-purple-50 rounded text-center">
                                            <div className="text-xs text-purple-500 uppercase">Precision (W)</div>
                                            <div className="text-xl font-bold">{modelResults.metrics.precision_weighted.toFixed(2)}</div>
                                        </div>
                                        <div className="p-3 bg-pink-50 rounded text-center">
                                            <div className="text-xs text-pink-500 uppercase">Recall (W)</div>
                                            <div className="text-xl font-bold">{modelResults.metrics.recall_weighted.toFixed(2)}</div>
                                        </div>
                                        <div className="p-3 bg-green-50 rounded text-center">
                                            <div className="text-xs text-green-500 uppercase">F1-Score (W)</div>
                                            <div className="text-xl font-bold">{modelResults.metrics.f1_weighted.toFixed(2)}</div>
                                        </div>
                                    </div>

                                    {/* Detailed Report Table */}
                                    {modelResults.classification_report && (
                                        <div className="overflow-x-auto">
                                            <table className="w-full text-sm text-right">
                                                <thead className="bg-slate-100 text-slate-600">
                                                    <tr>
                                                        <th className="p-2 text-left">Class</th>
                                                        <th className="p-2">Precision</th>
                                                        <th className="p-2">Recall</th>
                                                        <th className="p-2">F1-Score</th>
                                                        <th className="p-2">Support</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y">
                                                    {Object.entries(modelResults.classification_report).map(([label, metrics]) => {
                                                        if (typeof metrics !== 'object') return null; // skip 'accuracy' key if present as scalar
                                                        return (
                                                            <tr key={label}>
                                                                <td className="p-2 text-left font-medium">{label}</td>
                                                                <td className="p-2">{metrics.precision.toFixed(2)}</td>
                                                                <td className="p-2">{metrics.recall.toFixed(2)}</td>
                                                                <td className="p-2">{metrics['f1-score'].toFixed(2)}</td>
                                                                <td className="p-2">{metrics.support}</td>
                                                            </tr>
                                                        );
                                                    })}
                                                </tbody>
                                            </table>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="flex h-[calc(100vh-64px)] overflow-hidden">
            {/* Steps Sidebar */}
            <div className="w-64 bg-slate-50 border-r border-slate-200 overflow-y-auto p-4 space-y-1">
                {STEPS.map((step, idx) => {
                    const Icon = step.icon;
                    return (
                        <button
                            key={idx}
                            onClick={() => setCurrentStep(idx)}
                            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors
                                ${currentStep === idx
                                    ? 'bg-primary text-white shadow'
                                    : 'text-slate-600 hover:bg-white'
                                }
                            `}
                        >
                            <div className={`p-1 rounded ${currentStep === idx ? 'bg-white/20' : 'bg-slate-200 text-slate-500'}`}>
                                <Icon size={14} />
                            </div>
                            <span className="flex-1 text-left">{idx + 1}. {step.title}</span>
                            {currentStep > idx && <CheckCircle size={14} className="text-green-500" />}
                        </button>
                    )
                })}
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col relative bg-white">
                <div className="flex-1 overflow-y-auto p-6 md:p-10 pb-24">
                    {renderContent()}
                </div>

                {/* Bottom Navigation */}
                <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-4 px-6 md:px-10 flex justify-between items-center z-10 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
                    <button
                        onClick={handleBack}
                        disabled={currentStep === 0}
                        className="flex items-center gap-2 px-6 py-2.5 rounded-xl border border-slate-200 text-slate-600 font-medium hover:bg-slate-50 transition-colors disabled:opacity-50"
                    >
                        <ArrowLeft size={18} /> Back
                    </button>
                    <button
                        onClick={handleNext}
                        disabled={currentStep === STEPS.length - 1}
                        className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-primary text-white font-medium hover:bg-blue-600 transition-colors shadow-sm disabled:opacity-50"
                    >
                        Next <ArrowRight size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
}
