import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import { api } from '../services/api';
import { Plus, Trash2, Play, Info, Zap, TrendingUp, Filter, GitBranch, Wand2, Network, Sparkles, Database, Calculator, BarChart2 } from 'lucide-react';

export default function FeatureEngineering() {
    const { dataSummary, dataPreview, refreshData } = useData();
    const [features, setFeatures] = useState([]);
    const [activeTab, setActiveTab] = useState('transform');
    const [currentFeature, setCurrentFeature] = useState({
        type: 'threshold',
        name: '',
        config: {}
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    if (!dataSummary) {
        return <div className="p-8 text-center text-slate-500">Please load data first.</div>;
    }

    const { columns, dtypes } = dataSummary;
    const numericColumns = Object.keys(dtypes).filter(col =>
        dtypes[col].includes('int') || dtypes[col].includes('float')
    );
    const categoricalColumns = Object.keys(dtypes).filter(col =>
        dtypes[col].includes('object') || dtypes[col].includes('category')
    );

    const featureTypes = [
        { value: 'threshold', label: 'Threshold Label', icon: Filter, desc: 'Create binary feature based on threshold' },
        { value: 'quantile', label: 'Quantile Label', icon: TrendingUp, desc: 'Create label from percentile threshold' },
        { value: 'conditional', label: 'Conditional Feature', icon: GitBranch, desc: 'Create feature with AND/OR logic' },
        { value: 'binning', label: 'Binning', icon: Zap, desc: 'Group numeric values into categories' },
        { value: 'interaction', label: 'Interaction', icon: Plus, desc: 'Multiply or combine two features' }
    ];

    const addFeature = () => {
        if (!currentFeature.name) {
            setError('Please provide a feature name');
            return;
        }

        setFeatures([...features, { ...currentFeature, id: Date.now() }]);
        setCurrentFeature({
            type: 'threshold',
            name: '',
            config: {}
        });
        setError(null);
    };

    const removeFeature = (id) => {
        setFeatures(features.filter(f => f.id !== id));
    };

    const applyFeatures = async () => {
        if (features.length === 0) {
            setError('Please add at least one feature');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await api.post('/data/feature/engineer', { features });

            await refreshData();
            setFeatures([]);
            alert('Features created successfully!');
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Feature engineering failed');
        } finally {
            setLoading(false);
        }
    };

    const renderFeatureConfig = () => {
        switch (currentFeature.type) {
            case 'threshold':
                return (
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Source Column</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.column || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, column: e.target.value }
                                })}
                            >
                                <option value="">Select column...</option>
                                {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Operator</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.operator || '>'}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, operator: e.target.value }
                                })}
                            >
                                <option value=">">Greater than (&gt;)</option>
                                <option value=">=">Greater than or equal (&gt;=)</option>
                                <option value="<">Less than (&lt;)</option>
                                <option value="<=">Less than or equal (&lt;=)</option>
                                <option value="==">Equal to (==)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Threshold Value</label>
                            <input
                                type="number"
                                step="any"
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.threshold || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, threshold: parseFloat(e.target.value) }
                                })}
                                placeholder="e.g., 10"
                            />
                        </div>
                    </div>
                );

            case 'quantile':
                return (
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Source Column</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.column || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, column: e.target.value }
                                })}
                            >
                                <option value="">Select column...</option>
                                {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Quantile (0-1)</label>
                            <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="1"
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.quantile || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, quantile: parseFloat(e.target.value) }
                                })}
                                placeholder="e.g., 0.95 for 95th percentile"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Operator</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.operator || '>='}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, operator: e.target.value }
                                })}
                            >
                                <option value=">=">Greater than or equal (&gt;=)</option>
                                <option value=">">Greater than (&gt;)</option>
                                <option value="<=">Less than or equal (&lt;=)</option>
                                <option value="<">Less than (&lt;)</option>
                            </select>
                        </div>
                    </div>
                );

            case 'conditional':
                return (
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Logic Type</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.logic || 'AND'}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, logic: e.target.value }
                                })}
                            >
                                <option value="AND">AND (all conditions must be true)</option>
                                <option value="OR">OR (any condition must be true)</option>
                            </select>
                        </div>
                        <div className="bg-slate-50 p-3 rounded-lg space-y-2">
                            <p className="text-xs font-semibold text-slate-600">Condition 1</p>
                            <div className="grid grid-cols-3 gap-2">
                                <select
                                    className="p-2 border border-slate-300 rounded-lg text-sm"
                                    value={currentFeature.config.column1 || ''}
                                    onChange={(e) => setCurrentFeature({
                                        ...currentFeature,
                                        config: { ...currentFeature.config, column1: e.target.value }
                                    })}
                                >
                                    <option value="">Column...</option>
                                    {columns.map(col => <option key={col} value={col}>{col}</option>)}
                                </select>
                                <select
                                    className="p-2 border border-slate-300 rounded-lg text-sm"
                                    value={currentFeature.config.operator1 || '=='}
                                    onChange={(e) => setCurrentFeature({
                                        ...currentFeature,
                                        config: { ...currentFeature.config, operator1: e.target.value }
                                    })}
                                >
                                    <option value="==">=</option>
                                    <option value=">">&gt;</option>
                                    <option value=">=">&gt;=</option>
                                    <option value="<">&lt;</option>
                                    <option value="<=">&lt;=</option>
                                    <option value="!=">!=</option>
                                </select>
                                <input
                                    type="text"
                                    className="p-2 border border-slate-300 rounded-lg text-sm"
                                    value={currentFeature.config.value1 || ''}
                                    onChange={(e) => setCurrentFeature({
                                        ...currentFeature,
                                        config: { ...currentFeature.config, value1: e.target.value }
                                    })}
                                    placeholder="Value"
                                />
                            </div>
                        </div>
                        <div className="bg-slate-50 p-3 rounded-lg space-y-2">
                            <p className="text-xs font-semibold text-slate-600">Condition 2</p>
                            <div className="grid grid-cols-3 gap-2">
                                <select
                                    className="p-2 border border-slate-300 rounded-lg text-sm"
                                    value={currentFeature.config.column2 || ''}
                                    onChange={(e) => setCurrentFeature({
                                        ...currentFeature,
                                        config: { ...currentFeature.config, column2: e.target.value }
                                    })}
                                >
                                    <option value="">Column...</option>
                                    {columns.map(col => <option key={col} value={col}>{col}</option>)}
                                </select>
                                <select
                                    className="p-2 border border-slate-300 rounded-lg text-sm"
                                    value={currentFeature.config.operator2 || '=='}
                                    onChange={(e) => setCurrentFeature({
                                        ...currentFeature,
                                        config: { ...currentFeature.config, operator2: e.target.value }
                                    })}
                                >
                                    <option value="==">=</option>
                                    <option value=">">&gt;</option>
                                    <option value=">=">&gt;=</option>
                                    <option value="<">&lt;</option>
                                    <option value="<=">&lt;=</option>
                                    <option value="!=">!=</option>
                                </select>
                                <input
                                    type="text"
                                    className="p-2 border border-slate-300 rounded-lg text-sm"
                                    value={currentFeature.config.value2 || ''}
                                    onChange={(e) => setCurrentFeature({
                                        ...currentFeature,
                                        config: { ...currentFeature.config, value2: e.target.value }
                                    })}
                                    placeholder="Value"
                                />
                            </div>
                        </div>
                    </div>
                );

            case 'binning':
                return (
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Source Column</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.column || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, column: e.target.value }
                                })}
                            >
                                <option value="">Select column...</option>
                                {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Number of Bins</label>
                            <input
                                type="number"
                                min="2"
                                max="20"
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.bins || 5}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, bins: parseInt(e.target.value) }
                                })}
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Labels (comma-separated, optional)</label>
                            <input
                                type="text"
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.labels || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, labels: e.target.value }
                                })}
                                placeholder="e.g., Low, Medium, High"
                            />
                        </div>
                    </div>
                );

            case 'interaction':
                return (
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">First Column</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.column1 || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, column1: e.target.value }
                                })}
                            >
                                <option value="">Select column...</option>
                                {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Operation</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.operation || 'multiply'}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, operation: e.target.value }
                                })}
                            >
                                <option value="multiply">Multiply (*)</option>
                                <option value="add">Add (+)</option>
                                <option value="subtract">Subtract (-)</option>
                                <option value="divide">Divide (/)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Second Column</label>
                            <select
                                className="w-full p-2 border border-slate-300 rounded-lg"
                                value={currentFeature.config.column2 || ''}
                                onChange={(e) => setCurrentFeature({
                                    ...currentFeature,
                                    config: { ...currentFeature.config, column2: e.target.value }
                                })}
                            >
                                <option value="">Select column...</option>
                                {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
                            </select>
                        </div>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="p-4 md:p-8 max-w-7xl mx-auto space-y-6 md:space-y-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-200 pb-6">
                <div>
                    <h1 className="text-2xl md:text-3xl font-extrabold text-slate-800 tracking-tight flex items-center gap-3">
                        Feature Engineering
                    </h1>
                    <p className="text-slate-500 mt-1">Transform and create useful features for better models</p>
                </div>
                <div className="flex flex-wrap gap-2 bg-slate-100 p-1.5 rounded-xl border border-slate-200">
                    <button
                        onClick={() => setActiveTab('transform')}
                        className={`px-3 md:px-5 py-2 md:py-2.5 rounded-lg font-medium text-sm transition-all duration-200 flex items-center gap-2 ${activeTab === 'transform' ? 'bg-white text-primary shadow-sm ring-1 ring-slate-200/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'}`}
                    >
                        <Wand2 size={18} /> <span className="hidden sm:inline">Transform</span>
                    </button>
                    <button
                        onClick={() => setActiveTab('encode')}
                        className={`px-3 md:px-5 py-2 md:py-2.5 rounded-lg font-medium text-sm transition-all duration-200 flex items-center gap-2 ${activeTab === 'encode' ? 'bg-white text-primary shadow-sm ring-1 ring-slate-200/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'}`}
                    >
                        <Network size={18} /> <span className="hidden sm:inline">Encoding</span>
                    </button>
                    <button
                        onClick={() => setActiveTab('select')}
                        className={`px-3 md:px-5 py-2 md:py-2.5 rounded-lg font-medium text-sm transition-all duration-200 flex items-center gap-2 ${activeTab === 'select' ? 'bg-white text-primary shadow-sm ring-1 ring-slate-200/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'}`}
                    >
                        <Calculator size={18} /> <span className="hidden sm:inline">Selection</span>
                    </button>
                    <button
                        onClick={() => setActiveTab('pca')}
                        className={`px-3 md:px-5 py-2 md:py-2.5 rounded-lg font-medium text-sm transition-all duration-200 flex items-center gap-2 ${activeTab === 'pca' ? 'bg-white text-primary shadow-sm ring-1 ring-slate-200/50' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'}`}
                    >
                        <BarChart2 size={18} /> <span className="hidden sm:inline">PCA</span>
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
                {/* Main Content Area */}
                <div className="lg:col-span-8 space-y-6 lg:space-y-8">
                    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                        <h2 className="text-lg font-bold text-slate-800 mb-4">
                            {activeTab === 'transform' && 'Create New Feature'}
                            {activeTab === 'encode' && 'Encode Categorical Features'}
                            {activeTab === 'select' && 'Feature Selection'}
                            {activeTab === 'pca' && 'Principal Component Analysis'}
                        </h2>

                        <div className="space-y-4">
                            {activeTab === 'transform' && (
                                <>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-2">Feature Type</label>
                                        <div className="grid grid-cols-1 gap-2">
                                            {featureTypes.map(ft => {
                                                const Icon = ft.icon;
                                                return (
                                                    <button
                                                        key={ft.value}
                                                        onClick={() => setCurrentFeature({
                                                            ...currentFeature,
                                                            type: ft.value,
                                                            config: {}
                                                        })}
                                                        className={`p-3 rounded-lg border-2 text-left transition-all ${currentFeature.type === ft.value
                                                            ? 'border-primary bg-blue-50'
                                                            : 'border-slate-200 hover:border-slate-300'
                                                            }`}
                                                    >
                                                        <div className="flex items-start gap-3">
                                                            <Icon size={20} className={currentFeature.type === ft.value ? 'text-primary' : 'text-slate-400'} />
                                                            <div className="flex-1">
                                                                <p className="font-semibold text-sm">{ft.label}</p>
                                                                <p className="text-xs text-slate-500 mt-1">{ft.desc}</p>
                                                            </div>
                                                        </div>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Feature Name</label>
                                        <input
                                            type="text"
                                            className="w-full p-2 border border-slate-300 rounded-lg"
                                            value={currentFeature.name}
                                            onChange={(e) => setCurrentFeature({ ...currentFeature, name: e.target.value })}
                                            placeholder="e.g., is_high_value"
                                        />
                                    </div>

                                    {renderFeatureConfig()}
                                </>
                            )}

                            {activeTab === 'encode' && (
                                <>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Select Categorical Column</label>
                                        <select
                                            className="w-full p-2 border border-slate-300 rounded-lg"
                                            value={currentFeature.config.column || ''}
                                            onChange={(e) => setCurrentFeature({
                                                ...currentFeature,
                                                type: 'encode',
                                                name: `enc_${e.target.value}`,
                                                config: { ...currentFeature.config, column: e.target.value, strategy: currentFeature.config.strategy || 'onehot' }
                                            })}
                                        >
                                            <option value="">Select column...</option>
                                            {categoricalColumns.map(col => <option key={col} value={col}>{col}</option>)}
                                        </select>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Encoding Algorithm</label>
                                        <select
                                            className="w-full p-2 border border-slate-300 rounded-lg"
                                            value={currentFeature.config.strategy || 'onehot'}
                                            onChange={(e) => setCurrentFeature({
                                                ...currentFeature,
                                                config: { ...currentFeature.config, strategy: e.target.value }
                                            })}
                                        >
                                            <option value="onehot">One-Hot Encoding (Best for nominal, low cardinality)</option>
                                            <option value="label">Label Encoding (Best for ordinal or target variables)</option>
                                        </select>
                                    </div>
                                </>
                            )}

                            {activeTab === 'select' && (
                                <>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Selection Strategy</label>
                                        <select
                                            className="w-full p-2 border border-slate-300 rounded-lg"
                                            value={currentFeature.config.strategy || 'variance'}
                                            onChange={(e) => setCurrentFeature({
                                                ...currentFeature,
                                                type: 'select',
                                                name: `select_${e.target.value}`,
                                                config: { ...currentFeature.config, strategy: e.target.value, threshold: 0.8 }
                                            })}
                                        >
                                            <option value="variance">Low Variance Filter (Drop constants)</option>
                                            <option value="correlation">High Correlation Filter (Drop redundant pairs)</option>
                                        </select>
                                    </div>

                                    {(currentFeature.config.strategy === 'correlation') && (
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-1">Pearson Threshold (0-1)</label>
                                            <input
                                                type="number" step="0.05" min="0" max="1"
                                                className="w-full p-2 border border-slate-300 rounded-lg"
                                                value={currentFeature.config.threshold || 0.8}
                                                onChange={(e) => setCurrentFeature({
                                                    ...currentFeature,
                                                    config: { ...currentFeature.config, threshold: parseFloat(e.target.value) }
                                                })}
                                                placeholder="e.g., 0.85"
                                            />
                                            <p className="text-xs text-slate-500 mt-1">Drops one column from any pair with correlation &gt; threshold.</p>
                                        </div>
                                    )}
                                </>
                            )}

                            {activeTab === 'pca' && (
                                <>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Reduction Strategy</label>
                                        <select
                                            className="w-full p-2 border border-slate-300 rounded-lg"
                                            value={currentFeature.config.strategy || 'components'}
                                            onChange={(e) => setCurrentFeature({
                                                ...currentFeature,
                                                type: 'pca',
                                                name: `pca_transformed`,
                                                config: { ...currentFeature.config, strategy: e.target.value, n_components: 2 }
                                            })}
                                        >
                                            <option value="components">Fixed Number of Components (N)</option>
                                            <option value="variance">Target Explained Variance (%)</option>
                                        </select>
                                    </div>

                                    {currentFeature.config.strategy !== 'variance' ? (
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-1">N Components</label>
                                            <input
                                                type="number" step="1" min="1" max={Math.max(1, numericColumns.length)}
                                                className="w-full p-2 border border-slate-300 rounded-lg"
                                                value={currentFeature.config.n_components || 2}
                                                onChange={(e) => setCurrentFeature({
                                                    ...currentFeature,
                                                    config: { ...currentFeature.config, n_components: parseInt(e.target.value) }
                                                })}
                                            />
                                        </div>
                                    ) : (
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-1">Explained Variance Ratio (0-1)</label>
                                            <input
                                                type="number" step="0.05" min="0.1" max="0.99"
                                                className="w-full p-2 border border-slate-300 rounded-lg"
                                                value={currentFeature.config.n_components || 0.95}
                                                onChange={(e) => setCurrentFeature({
                                                    ...currentFeature,
                                                    config: { ...currentFeature.config, n_components: parseFloat(e.target.value) }
                                                })}
                                            />
                                        </div>
                                    )}
                                </>
                            )}
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-2">Feature Type</label>
                                <div className="grid grid-cols-1 gap-2">
                                    {featureTypes.map(ft => {
                                        const Icon = ft.icon;
                                        return (
                                            <button
                                                key={ft.value}
                                                onClick={() => setCurrentFeature({
                                                    ...currentFeature,
                                                    type: ft.value,
                                                    config: {}
                                                })}
                                                className={`p-3 rounded-lg border-2 text-left transition-all ${currentFeature.type === ft.value
                                                    ? 'border-primary bg-blue-50'
                                                    : 'border-slate-200 hover:border-slate-300'
                                                    }`}
                                            >
                                                <div className="flex items-start gap-3">
                                                    <Icon size={20} className={currentFeature.type === ft.value ? 'text-primary' : 'text-slate-400'} />
                                                    <div className="flex-1">
                                                        <p className="font-semibold text-sm">{ft.label}</p>
                                                        <p className="text-xs text-slate-500 mt-1">{ft.desc}</p>
                                                    </div>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Feature Name</label>
                                <input
                                    type="text"
                                    className="w-full p-2 border border-slate-300 rounded-lg"
                                    value={currentFeature.name}
                                    onChange={(e) => setCurrentFeature({ ...currentFeature, name: e.target.value })}
                                    placeholder="e.g., is_high_value"
                                />
                            </div>

                            {renderFeatureConfig()}

                            <button
                                onClick={addFeature}
                                className="w-full py-2 bg-primary text-white rounded-lg font-medium hover:bg-blue-600 flex items-center justify-center gap-2"
                            >
                                <Plus size={18} />
                                Add to Queue
                            </button>
                        </div>
                    </div>

                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <div className="flex gap-2">
                            <Info size={18} className="text-blue-600 flex-shrink-0 mt-0.5" />
                            <div className="text-sm text-blue-800">
                                <p className="font-semibold mb-1">Tips:</p>
                                <ul className="list-disc list-inside space-y-1 text-xs">
                                    <li>Create threshold features for binary classification</li>
                                    <li>Use quantiles to handle outliers (e.g., 95th percentile)</li>
                                    <li>Combine conditions with AND/OR logic for complex rules</li>
                                    <li>Interaction features can capture relationships between variables</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Secondary Panel - Preview or Suggestions */}
                <div className="lg:col-span-4 space-y-6 lg:space-y-8">
                    {/* Activity Feed / Suggestions */}
                    <div className="bg-gradient-to-br from-indigo-50 to-blue-50 p-6 rounded-2xl border border-indigo-100 shadow-sm relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-4 opacity-10">
                            <Sparkles size={64} className="text-indigo-600" />
                        </div>
                        <h3 className="font-bold text-indigo-900 mb-4 flex items-center gap-2">
                            <Sparkles size={18} className="text-indigo-600" /> AI Suggestions
                        </h3>
                        <div className="space-y-3 relative z-10">
                            <div className="bg-white/80 backdrop-blur-sm p-3 rounded-lg border border-white/50 shadow-sm text-sm text-slate-700">
                                <strong>Log Transform</strong> suggested for skewed numerical columns like <span className="font-mono text-xs bg-slate-100 px-1 rounded">Amount</span>.
                            </div>
                            <div className="bg-white/80 backdrop-blur-sm p-3 rounded-lg border border-white/50 shadow-sm text-sm text-slate-700">
                                <strong>One-Hot Encoding</strong> is recommended for categorical variables with &lt; 10 unique values.
                            </div>
                        </div>
                    </div>

                    {/* Dataset Info Summary */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                        <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2 text-sm uppercase tracking-wider">
                            Dataset Status
                        </h3>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
                                <span className="text-slate-500 text-sm">Total Features</span>
                                <span className="font-semibold text-slate-800">{columns.length}</span>
                            </div>
                            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
                                <span className="text-slate-500 text-sm">Numerical Features</span>
                                <span className="font-medium text-slate-700">{numericColumns.length}</span>
                            </div>
                            <div className="flex justify-between items-center pb-3 border-b border-slate-100">
                                <span className="text-slate-500 text-sm">Categorical Features</span>
                                <span className="font-medium text-slate-700">{categoricalColumns.length}</span>
                            </div>
                            <div className="pt-2">
                                <button className="w-full py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2">
                                    <Database size={16} /> View Dataset Snapshot
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Feature Queue */}
                    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                        <h2 className="text-lg font-bold text-slate-800 mb-4">Feature Queue ({features.length})</h2>

                        {features.length === 0 ? (
                            <div className="text-center py-12 text-slate-400">
                                <Zap size={48} className="mx-auto mb-3 opacity-50" />
                                <p>No features queued yet</p>
                                <p className="text-sm mt-1">Add features to get started</p>
                            </div>
                        ) : (
                            <div className="space-y-3 max-h-96 overflow-y-auto">
                                {features.map((feature, idx) => (
                                    <div key={feature.id} className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                                        <div className="flex items-start justify-between mb-2">
                                            <div className="flex-1">
                                                <p className="font-semibold text-slate-800">{feature.name}</p>
                                                <p className="text-xs text-slate-500 mt-1">
                                                    {featureTypes.find(ft => ft.value === feature.type)?.label}
                                                </p>
                                            </div>
                                            <button
                                                onClick={() => removeFeature(feature.id)}
                                                className="text-red-500 hover:text-red-700 p-1"
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                        <div className="text-xs text-slate-600 bg-white p-2 rounded border border-slate-200 font-mono">
                                            {JSON.stringify(feature.config, null, 2)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {features.length > 0 && (
                            <button
                                onClick={applyFeatures}
                                disabled={loading}
                                className={`w-full mt-4 py-3 rounded-xl font-bold flex items-center justify-center gap-2 text-white ${loading ? 'bg-slate-400' : 'bg-emerald-600 hover:bg-emerald-700 shadow-lg'
                                    }`}
                            >
                                <Play size={20} />
                                {loading ? 'Applying...' : `Apply ${features.length} Feature${features.length > 1 ? 's' : ''}`}
                            </button>
                        )}

                        {error && (
                            <div className="mt-4 p-3 bg-red-50 text-red-600 rounded-lg text-sm">
                                {error}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Dataset Preview Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mt-8">
                <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/50 flex justify-between items-center">
                    <div>
                        <h3 className="font-bold text-slate-800">Dataset Preview (First 5 Rows)</h3>
                        <p className="text-xs text-slate-500 mt-0.5">Showing columns and sample rows of the active dataset</p>
                    </div>
                    <span className="text-xs font-semibold bg-slate-200 text-slate-700 px-3 py-1 rounded-full">
                        {dataSummary.shape[0].toLocaleString()} rows x {dataSummary.shape[1]} columns
                    </span>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left text-slate-600">
                        <thead className="text-xs text-slate-500 uppercase bg-slate-50 border-b border-slate-200">
                            <tr>
                                {columns.map(col => (
                                    <th key={col} className="px-4 md:px-6 py-3 font-semibold whitespace-nowrap">{col}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {dataPreview?.slice(0, 5).map((row, i) => (
                                <tr key={i} className="bg-white border-b border-slate-100 hover:bg-slate-50 transition-colors">
                                    {columns.map(col => (
                                        <td key={`${i}-${col}`} className="px-4 md:px-6 py-4 whitespace-nowrap">
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
    );
}
