import React, { useState, useEffect } from 'react';
import { useData } from '../context/DataContext';
import { getModelMetadata, predictModel, downloadModel } from '../services/api';
import { AlertCircle, CheckCircle, Info, ArrowRight, Zap, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';

const ModelPrediction = () => {
    const { predictionResult, modelResult } = useData();
    const [metadata, setMetadata] = useState(null);
    const [inputs, setInputs] = useState({});
    const [prediction, setPrediction] = useState(null);
    const [activeTab, setActiveTab] = useState('single');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [predicting, setPredicting] = useState(false);

    useEffect(() => {
        loadMetadata();
    }, []);

    const loadMetadata = async () => {
        try {
            setLoading(true);
            const data = await getModelMetadata();
            setMetadata(data);

            // Initialize inputs - prefill with synced sample inputs from agent run if available
            const initialInputs = {};
            const featureList = data?.features || data?.feature_columns || [];
            featureList.forEach(feature => {
                if (predictionResult?.sample_inputs && predictionResult.sample_inputs[feature] !== undefined) {
                    initialInputs[feature] = predictionResult.sample_inputs[feature];
                } else {
                    initialInputs[feature] = '';
                }
            });
            setInputs(initialInputs);

            if (predictionResult?.prediction !== null && predictionResult?.prediction !== undefined) {
                setPrediction(predictionResult.prediction);
            }
            setError(null);
        } catch (err) {
            console.error(err);
            setError("Failed to load model metadata. Ensure a model is trained via AutoML Studio or the Autonomous Agent.");
        } finally {
            setLoading(false);
        }
    };

    const handleInputChange = (feature, value) => {
        setInputs(prev => ({
            ...prev,
            [feature]: value
        }));
    };

    const handlePredict = async (e) => {
        e.preventDefault();
        setPredicting(true);
        setPrediction(null);
        setError(null);

        try {
            const formattedInputs = {};
            for (const [key, value] of Object.entries(inputs)) {
                const type = metadata?.feature_types?.[key] || '';
                const isTextType = type === 'text' || type === 'object' || type === 'string' || type === 'category';
                const isTextHeuristic = key.toLowerCase().includes('caption') || key.toLowerCase().includes('text') || key.toLowerCase().includes('description') || key.toLowerCase().includes('message');

                if (isTextType || isTextHeuristic) {
                    formattedInputs[key] = value;
                } else {
                    // Try number
                    const num = Number(value);
                    formattedInputs[key] = isNaN(num) ? value : num;
                }
            }

            const result = await predictModel(formattedInputs);
            setPrediction(result.prediction);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.error || err.response?.data?.detail || err.message || "Prediction failed");
        } finally {
            setPredicting(false);
        }
    };

    const handleDownload = async () => {
        try {
            const blob = await downloadModel();
            const url = window.URL.createObjectURL(new Blob([blob]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'model.pkl');
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
        } catch (err) {
            console.error(err);
            setError("Failed to download model");
        }
    };

    if (loading) return <div className="p-8 text-center">Loading Model Metadata...</div>;

    // If no model, show error nicely
    if (error && !metadata) return (
        <div className="p-8 text-center text-red-500">
            <h2 className="text-xl font-bold mb-4">Model Error</h2>
            <p>{error}</p>
        </div>
    );

    return (
        <div className="container mx-auto p-4 max-w-4xl">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                    Model Prediction
                </h1>
                <button
                    onClick={handleDownload}
                    className="bg-gray-800 text-white px-4 py-2 rounded hover:bg-gray-700 transition"
                >
                    Download Model
                </button>
            </div>

            {/* Synced Model Status Banner */}
            {predictionResult && (
                <div className="mb-6 bg-gradient-to-r from-cyan-50 via-sky-50 to-indigo-50 border border-cyan-200/80 rounded-2xl p-4 shadow-sm flex items-center justify-between gap-3 flex-wrap">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-lg bg-cyan-600 text-white flex items-center justify-center shadow-xs">
                            <Zap size={16} />
                        </div>
                        <div>
                            <h3 className="text-xs font-bold text-cyan-950 flex items-center gap-2">
                                Champion Model Synced for Live Inference: <span className="text-indigo-600 font-mono">{predictionResult.model_name || metadata?.model_name || 'Champion'}</span>
                            </h3>
                            <p className="text-[11px] text-cyan-700">
                                Target: <code className="font-bold">{predictionResult.target_column || metadata?.target_column}</code> • Inputs pre-populated with active model features & sample values.
                            </p>
                        </div>
                    </div>
                    {predictionResult.timestamp && (
                        <span className="text-[10px] font-mono text-cyan-700 bg-white/90 px-2 py-0.5 rounded border border-cyan-200">
                            Synced {new Date(predictionResult.timestamp).toLocaleTimeString()}
                        </span>
                    )}
                </div>
            )}

            <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-100">
                {/* Model Status Banner */}
                <div className="mb-6">
                    {metadata ? (
                        <div className="flex items-center gap-3 border p-4 rounded-xl mb-4 bg-emerald-50 border-emerald-200 text-emerald-800">
                            <CheckCircle size={20} className="text-emerald-600 flex-shrink-0" />
                            <div>
                                <p className="font-semibold text-sm">Model Loaded & Ready for Inference</p>
                                <p className="text-xs text-emerald-700">
                                    Algorithm: <strong>{metadata.type || metadata.model_type || metadata.model_name || "Trained Model"}</strong> • Target: <strong>{metadata.target || metadata.target_column || "Target"}</strong> ({metadata.is_classifier || metadata.problem_type === 'classification' ? "Classification" : "Regression"}) • Input Dimensions: <strong>{(metadata.features || metadata.feature_columns || []).length} features</strong>
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="flex items-center gap-3 border p-4 rounded-xl mb-4 bg-amber-50 border-amber-200 text-amber-800">
                            <AlertCircle size={20} className="text-amber-600 flex-shrink-0" />
                            <div>
                                <p className="font-semibold text-sm">No Trained Model Found</p>
                                <p className="text-xs text-amber-700">Please train a model in Model Studio or launch the Autonomous Agent first.</p>
                            </div>
                        </div>
                    )}
                </div>

                <div className="mb-6 flex flex-wrap gap-2">
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-800">
                        {metadata?.type || metadata?.model_type || metadata?.model_name || "Model"}
                    </span>
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-purple-100 text-purple-800">
                        {metadata?.is_classifier || metadata?.problem_type === 'classification' ? "Classification" : "Regression"}
                    </span>
                    {(metadata?.target || metadata?.target_column) && (
                        <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
                            Target: {metadata.target || metadata.target_column}
                        </span>
                    )}
                </div>

                {/* Tabs for Single vs Batch Prediction */}
                <div className="flex border-b border-gray-200 mb-6">
                    <button
                        className={`py-2 px-4 text-sm font-medium ${activeTab === 'single' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                        onClick={() => setActiveTab('single')}
                    >
                        Single Prediction
                    </button>
                    <button
                        className={`py-2 px-4 text-sm font-medium ${activeTab === 'batch' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                        onClick={() => setActiveTab('batch')}
                    >
                        Batch Prediction
                    </button>
                </div>

                <div className="lg:grid lg:grid-cols-10 lg:gap-8"> {/* Main layout grid */}
                    {/* Left Side - Input Form */}
                    <div className="lg:col-span-6">
                        {/* Single Predict Form */}
                        {activeTab === 'single' && (
                            <form onSubmit={handlePredict} className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {(metadata?.features || metadata?.feature_columns || Object.keys(inputs)).map(feature => {
                                    const type = metadata?.feature_types?.[feature] || '';
                                    const isTextType = type === 'text' || type === 'object' || type === 'string' || type === 'category';
                                    const isTextHeuristic = /(name|text|desc|msg|message|caption|title|label|category|type|status|group|id)/i.test(feature);
                                    const isText = isTextType || isTextHeuristic;

                                    const isTextArea = isText && (/(text|desc|caption|message|summary)/i.test(feature));

                                    return (
                                        <div key={feature} className="flex flex-col">
                                            <label className="text-gray-700 font-medium mb-2 capitalize">
                                                {feature.replace(/_/g, ' ')}
                                            </label>
                                            {isTextArea ? (
                                                <textarea
                                                    className="border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                                    value={inputs[feature]}
                                                    onChange={(e) => handleInputChange(feature, e.target.value)}
                                                    placeholder={`Enter ${feature}...`}
                                                    rows="3"
                                                    required
                                                />
                                            ) : (
                                                <input
                                                    type="text"
                                                    className="border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                                    value={inputs[feature]}
                                                    onChange={(e) => handleInputChange(feature, e.target.value)}
                                                    placeholder={`Enter ${feature}...`}
                                                    required
                                                />
                                            )}
                                        </div>
                                    );
                                })}

                                <div className="md:col-span-2 mt-4">
                                    <button
                                        type="submit"
                                        disabled={predicting}
                                        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 rounded-lg hover:shadow-lg transition transform hover:-translate-y-0.5 disabled:opacity-50"
                                    >
                                        {predicting ? "Running Prediction..." : "Predict Result"}
                                    </button>
                                </div>
                            </form>
                        )}

                        {/* Batch Predict Layout (Simplified for MVP) */}
                        {activeTab === 'batch' && (
                            <div className="bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm text-center">
                                <div className="max-w-md mx-auto py-12">
                                    <div className="w-16 h-16 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                                        <ArrowRight size={24} />
                                    </div>
                                    <h3 className="text-lg font-bold text-slate-800 mb-2">Batch Predictions Pipeline</h3>
                                    <p className="text-slate-500 mb-6">Connect to an external database or upload a new CSV file to run predictions on thousands of rows instantly.</p>
                                    <button className="px-6 py-2.5 bg-slate-800 text-white rounded-xl font-medium shadow-sm hover:bg-slate-700 w-full md:w-auto">
                                        Setup Built-In Database Connection
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right Side - Prediction Results & Info */}
                    <div className="lg:col-span-4 space-y-6 lg:space-y-8 mt-8 lg:mt-0"> {/* Added mt-8 for spacing on smaller screens */}
                        {error && (
                            <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-lg">
                                <p className="font-bold">Error</p>
                                <p>{error}</p>
                            </div>
                        )}

                        {prediction !== null && prediction !== undefined && (
                            <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-2xl border border-green-200 shadow-sm animate-fade-in relative overflow-hidden">
                                <div className="absolute -top-4 -right-4 text-green-200 opacity-50">
                                    <CheckCircle size={120} />
                                </div>
                                <h3 className="font-bold text-green-900 mb-2 relative z-10 flex items-center gap-2">
                                    <CheckCircle size={20} className="text-green-600" /> Prediction Result
                                </h3>
                                <div className="mt-4 relative z-10">
                                    <p className="text-sm text-green-700 uppercase tracking-wider font-semibold mb-1">
                                        Predicted Result for Target <span className="text-green-900 border-b-2 border-green-400 pb-0.5 ml-1">{metadata?.target || metadata?.target_column || "Target"}</span>
                                    </p>
                                    <p className="text-4xl font-extrabold text-green-700 mt-3">
                                        {typeof prediction === 'number' && !Number.isInteger(prediction)
                                            ? prediction.toFixed(2)
                                            : String(prediction)}
                                    </p>
                                </div>
                                <div className="mt-6 pt-4 border-t border-green-200/50 relative z-10">
                                    <p className="text-xs text-green-700/70">Based on Champion Model: {metadata?.type || metadata?.model_type || metadata?.model_name} ({metadata?.is_classifier || metadata?.problem_type === 'classification' ? "Classification" : "Regression"})</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div >
        </div >
    );
};

export default ModelPrediction;
