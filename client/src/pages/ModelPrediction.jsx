import React, { useState, useEffect } from 'react';
import { getModelMetadata, predictModel, downloadModel } from '../services/api';
import { AlertCircle, CheckCircle, Info, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const ModelPrediction = () => {
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

            // Initialize inputs
            const initialInputs = {};
            data?.features?.forEach(feature => {
                initialInputs[feature] = '';
            });
            setInputs(initialInputs);
            setError(null);
        } catch (err) {
            console.error(err);
            setError("Failed to load model metadata. Ensure a model is trained.");
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

            <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-100">
                {/* Dataset Status Banner */}
                {/* Dataset Status Banner */}
                <div className="mb-6">
                    {metadata.dataset_loaded ? (
                        metadata.matched_features && metadata.matched_features.length > 0 ? (
                            <div className={`flex items-center gap-3 border p-4 rounded-lg mb-4 ${metadata.matched_features.length === (metadata?.features?.length || 0) ? 'bg-green-50 border-green-200 text-green-800' : 'bg-blue-50 border-blue-200 text-blue-800'}`}>
                                {metadata.matched_features.length === (metadata?.features?.length || 0) ? <CheckCircle size={20} /> : <Info size={20} />}
                                <div>
                                    <p className="font-semibold">Dataset Loaded</p>
                                    <p className="text-sm">
                                        Matched {metadata.matched_features.length} / {metadata?.features?.length || 0} features.
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <div className="flex items-center gap-3 border p-4 rounded-lg mb-4 bg-red-50 border-red-200 text-red-800">
                                <AlertCircle size={20} />
                                <div>
                                    <p className="font-semibold">No Dataset Loaded</p>
                                    <p className="text-sm">Please train a model first to load a dataset.</p>
                                </div>
                            </div>
                        )
                    ) : (
                        <div className="flex items-center gap-3 border p-4 rounded-lg mb-4 bg-red-50 border-red-200 text-red-800">
                            <AlertCircle size={20} />
                            <div>
                                <p className="font-semibold">No Dataset Loaded</p>
                                <p className="text-sm">Please train a model first to load a dataset.</p>
                            </div>
                        </div>
                    )}
                </div>

                <div className="mb-6">
                    <span className="inline-block px-3 py-1 rounded-full text-sm font-semibold bg-blue-100 text-blue-800 mr-2">
                        {metadata.type}
                    </span>
                    <span className="inline-block px-3 py-1 rounded-full text-sm font-semibold bg-purple-100 text-purple-800">
                        {metadata.is_classifier ? "Classification" : "Regression"}
                    </span>
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
                                {metadata?.features?.map(feature => {
                                    const type = metadata.feature_types?.[feature] || '';
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

                        {prediction && (
                            <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-2xl border border-green-200 shadow-sm animate-fade-in relative overflow-hidden">
                                <div className="absolute -top-4 -right-4 text-green-200 opacity-50">
                                    <CheckCircle size={120} />
                                </div>
                                <h3 className="font-bold text-green-900 mb-2 relative z-10 flex items-center gap-2">
                                    <CheckCircle size={20} className="text-green-600" /> Prediction Result
                                </h3>
                                <div className="mt-4 relative z-10">
                                    <p className="text-sm text-green-700 uppercase tracking-wider font-semibold mb-1">
                                        Predicted Result for Target <span className="text-green-900 border-b-2 border-green-400 pb-0.5 ml-1">{metadata?.target_column || "Value"}</span>
                                    </p>
                                    <p className="text-4xl font-extrabold text-green-700 mt-3">
                                        {typeof prediction === 'number' && !Number.isInteger(prediction)
                                            ? prediction.toFixed(2)
                                            : prediction}
                                    </p>
                                </div>
                                <div className="mt-6 pt-4 border-t border-green-200/50 relative z-10">
                                    <p className="text-xs text-green-700/70">Based on Model: {metadata?.type} ({metadata?.is_classifier ? "Classification" : "Regression"})</p>
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
