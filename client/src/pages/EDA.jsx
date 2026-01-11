import React, { useEffect, useState } from 'react';
import { useData } from '../context/DataContext';
import { getEDA, getEDASummary, analyzeDrivers } from '../services/api';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ScatterChart, Scatter, ZAxis, Cell
} from 'recharts';
import { Loader, Target } from 'lucide-react';

export default function EDA() {
    const { dataSummary } = useData();
    const [edaData, setEdaData] = useState(null);
    const [edaSummaryDetails, setEdaSummaryDetails] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [scatterX, setScatterX] = useState('');
    const [scatterY, setScatterY] = useState('');

    // Drivers Analysis State
    const [driversData, setDriversData] = useState(null);
    const [loadingDrivers, setLoadingDrivers] = useState(false);
    const [selectedTarget, setSelectedTarget] = useState('');

    useEffect(() => {
        if (dataSummary) {
            fetchEDA();
        }
    }, [dataSummary]);

    const fetchEDA = async () => {
        setLoading(true);
        try {
            const res = await getEDA();
            setEdaData(res);

            const summaryRes = await getEDASummary();
            setEdaSummaryDetails(summaryRes.insights);
            // Default scatter selection
            if (res.scatter_data && res.scatter_data.length > 0) {
                const keys = Object.keys(res.scatter_data[0]);
                if (keys.length >= 2) {
                    setScatterX(keys[0]);
                    setScatterY(keys[1]);
                }
            }
        } catch (err) {
            console.error(err);
            setError('Failed to fetch EDA data');
        } finally {
            setLoading(false);
        }
    };

    const handleAnalyzeDrivers = async () => {
        if (!selectedTarget) return;
        setLoadingDrivers(true);
        try {
            const res = await analyzeDrivers(selectedTarget);
            setDriversData(res);
        } catch (err) {
            console.error(err);
            setError('Failed to analyze drivers');
        } finally {
            setLoadingDrivers(false);
        }
    };

    if (!dataSummary) return <div className="p-8 text-center text-slate-500">Please load data first.</div>;
    if (loading) return <div className="p-8 flex flex-col items-center justify-center min-h-[50vh]"><Loader className="animate-spin text-primary mb-4" size={40} /><p className="text-slate-500">Generating insights...</p></div>;
    if (error) return <div className="p-8 text-red-500">{error}</div>;
    if (!edaData) return null;

    const { distributions, categorical_counts, description, correlation, scatter_data, box_plot_data } = edaData;

    // Prepare Heatmap Data (Basic Implementation)
    // Recharts doesn't have a native Heatmap, so we use ScatterChart with custom cells or just a grid.
    // For simplicity MVP, we will render a color-coded table for Correlation.

    return (
        <div className="p-8 max-w-7xl mx-auto space-y-12">
            <div>
                <h1 className="text-3xl font-bold text-slate-800 mb-2">Exploratory Analysis</h1>
                <p className="text-slate-500">Understanding your data distribution and statistics.</p>
            </div>

            {/* AI Insights Section */}
            {edaSummaryDetails && edaSummaryDetails.length > 0 && (
                <section className="bg-indigo-50 border border-indigo-100 rounded-xl p-6">
                    <h2 className="flex items-center text-lg font-semibold text-indigo-700 mb-4">
                        <span className="mr-2">✨</span> AI Insights
                    </h2>
                    <ul className="space-y-2">
                        {edaSummaryDetails.map((insight, idx) => (
                            <li key={idx} className="flex items-start text-indigo-900 border-l-2 border-indigo-300 pl-4 py-1 bg-white/50 rounded-r-md">
                                {insight}
                            </li>
                        ))}
                    </ul>
                </section>
            )}

            {/* Key Drivers Analysis */}
            <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                            <Target className="text-blue-600" /> Key Drivers Analysis
                        </h2>
                        <p className="text-sm text-slate-500">Discover which factors influence a specific target column.</p>
                    </div>
                </div>

                <div className="flex gap-4 items-end mb-6">
                    <div className="flex-1 max-w-xs">
                        <label className="text-sm font-medium text-slate-700 block mb-1">Target Column (e.g. Outcome)</label>
                        <select
                            value={selectedTarget}
                            onChange={e => setSelectedTarget(e.target.value)}
                            className="w-full p-2 border border-slate-300 rounded-lg"
                        >
                            <option value="">Select Target...</option>
                            {Object.keys(distributions).map(col => <option key={col} value={col}>{col}</option>)}
                            {/* Also include categorical columns if likely targets */}
                            {Object.keys(categorical_counts).map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                    </div>
                    <button
                        onClick={handleAnalyzeDrivers}
                        disabled={loadingDrivers || !selectedTarget}
                        className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
                    >
                        {loadingDrivers && <Loader className="animate-spin" size={16} />}
                        Analyze Drivers
                    </button>
                </div>

                {driversData && (
                    <div className="space-y-6 animate-in fade-in duration-500">
                        <div className="p-4 bg-blue-50 text-blue-800 rounded-lg text-sm border border-blue-100">
                            <strong>Analysis Type:</strong> {driversData.problem_type} Model used to predict <strong>{driversData.target}</strong>.
                        </div>
                        <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={driversData.drivers} layout="vertical" margin={{ left: 100 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                    <XAxis type="number" domain={[0, 1]} />
                                    <YAxis dataKey="feature" type="category" width={100} tick={{ fontSize: 12 }} />
                                    <Tooltip formatter={(value) => (value * 100).toFixed(1) + '%'} />
                                    <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} name="Importance Score" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </section>

            {/* 1. Correlation Matrix (Heatmap Table style) */}
            {correlation && correlation.length > 0 && (
                <section>
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-2">Correlation Matrix</h2>
                    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm overflow-x-auto">
                        <CorrelationHeatmap data={correlation} />
                    </div>
                </section>
            )}

            {/* 2. Scatter Plot */}
            {scatter_data && scatter_data.length > 0 && (
                <section>
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-2">Scatter Plot Analysis</h2>
                    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                        <div className="flex gap-4 mb-4">
                            <div>
                                <label className="text-sm font-medium text-slate-700 block mb-1">X Axis</label>
                                <select value={scatterX} onChange={e => setScatterX(e.target.value)} className="p-2 border rounded">
                                    {Object.keys(scatter_data[0]).map(k => <option key={k} value={k}>{k}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="text-sm font-medium text-slate-700 block mb-1">Y Axis</label>
                                <select value={scatterY} onChange={e => setScatterY(e.target.value)} className="p-2 border rounded">
                                    {Object.keys(scatter_data[0]).map(k => <option key={k} value={k}>{k}</option>)}
                                </select>
                            </div>
                        </div>

                        <div className="h-96">
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid />
                                    <XAxis type="number" dataKey={scatterX} name={scatterX} label={{ value: scatterX, position: 'insideBottomRight', offset: -10 }} />
                                    <YAxis type="number" dataKey={scatterY} name={scatterY} label={{ value: scatterY, angle: -90, position: 'insideLeft' }} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                    <Scatter name="Data" data={scatter_data} fill="#8884d8" />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </section>
            )}

            {/* 3. Numerical Distributions */}
            {Object.keys(distributions).length > 0 && (
                <section>
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-2">Numeric Distributions</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {Object.entries(distributions).map(([col, data]) => (
                            <div key={col} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                <h3 className="font-medium text-slate-700 mb-4">{col}</h3>
                                <div className="h-48">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={data}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                            <XAxis dataKey="range" tick={{ fontSize: 0 }} />
                                            <YAxis />
                                            <Tooltip />
                                            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            {/* 4. Categorical Distributions */}
            {Object.keys(categorical_counts).length > 0 && (
                <section>
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-2">Categorical Distributions</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {Object.entries(categorical_counts).map(([col, data]) => (
                            <div key={col} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                <h3 className="font-medium text-slate-700 mb-4">{col}</h3>
                                <div className="h-48">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={data} layout="vertical">
                                            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                            <XAxis type="number" />
                                            <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 10 }} />
                                            <Tooltip />
                                            <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            {/* 6. Box Plot Analysis (Outliers) */}
            {box_plot_data && Object.keys(box_plot_data).length > 0 && (
                <section>
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-2">Box Plot Analysis (Outliers)</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {Object.entries(box_plot_data).map(([col, stats]) => (
                            <div key={col} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                <h3 className="font-medium text-slate-700 mb-4">{col}</h3>
                                <div className="h-48 relative border-b border-l border-slate-200 m-4">
                                    {/* Simplified Custom Box Plot using CSS/HTML because Recharts is tricky for BoxPlots */}
                                    <SimpleBoxPlot stats={stats} />
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            {/* 5. Descriptive Statistics */}
            <section>
                <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-2">Descriptive Statistics</h2>
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs text-slate-500 uppercase bg-slate-50 border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-3 font-semibold">Stat</th>
                                {Object.keys(description).map(col => (
                                    <th key={col} className="px-6 py-3 font-semibold whitespace-nowrap">{col}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map(stat => (
                                <tr key={stat} className="hover:bg-slate-50/50">
                                    <td className="px-6 py-3 font-medium text-slate-700 bg-slate-50/30">{stat}</td>
                                    {Object.keys(description).map(col => (
                                        <td key={`${col}-${stat}`} className="px-6 py-3 text-slate-600">
                                            {description[col][stat] !== undefined && description[col][stat] !== null
                                                ? (typeof description[col][stat] === 'number' ? description[col][stat].toFixed(2) : description[col][stat])
                                                : '-'}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
}

// Simple Heatmap Table Component
const CorrelationHeatmap = ({ data }) => {
    // Unique variables
    const vars = [...new Set(data.map(d => d.x))];

    // Create matrix
    const matrix = {};
    vars.forEach(v => matrix[v] = {});
    data.forEach(d => matrix[d.x][d.y] = d.value);

    // Color scale helper
    const getColor = (val) => {
        val = Math.max(-1, Math.min(1, val));
        // Simple red-white-blue scale
        if (val > 0) return `rgba(59, 130, 246, ${val})`; // Blue
        if (val < 0) return `rgba(239, 68, 68, ${Math.abs(val)})`; // Red
        return 'white';
    };

    return (
        <table className="w-full border-collapse">
            <thead>
                <tr>
                    <th className="p-2 border border-slate-100"></th>
                    {vars.map(v => <th key={v} className="p-2 text-xs text-slate-500 font-medium rotate-0 border border-slate-100">{v}</th>)}
                </tr>
            </thead>
            <tbody>
                {vars.map(row => (
                    <tr key={row}>
                        <th className="p-2 text-xs text-slate-500 font-medium text-left border border-slate-100">{row}</th>
                        {vars.map(col => {
                            const val = matrix[row][col];
                            return (
                                <td key={col} className="p-2 text-center text-xs border border-slate-100" style={{ backgroundColor: getColor(val) }}>
                                    {val.toFixed(2)}
                                </td>
                            );
                        })}
                    </tr>
                ))}
            </tbody>
        </table>
    );
};

const SimpleBoxPlot = ({ stats }) => {
    const { min, q1, median, q3, max } = stats;
    const range = max - min;
    const getPos = (val) => ((val - min) / range) * 100;

    return (
        <div className="w-full h-full relative">
            {/* Whiskers Line */}
            <div className="absolute top-1/2 left-0 h-0.5 bg-slate-300 w-full transform -translate-y-1/2"></div>

            {/* Box */}
            <div
                className="absolute top-1/2 h-12 bg-blue-100 border border-blue-500 transform -translate-y-1/2"
                style={{ left: `${getPos(q1)}%`, width: `${getPos(q3) - getPos(q1)}%` }}
            ></div>

            {/* Median Line */}
            <div
                className="absolute top-1/2 h-12 w-1 bg-blue-700 transform -translate-y-1/2"
                style={{ left: `${getPos(median)}%` }}
            ></div>

            {/* Min/Max Ticks */}
            <div className="absolute top-1/2 h-6 w-0.5 bg-slate-500 transform -translate-y-1/2 left-0"></div>
            <div className="absolute top-1/2 h-6 w-0.5 bg-slate-500 transform -translate-y-1/2 right-0"></div>

            {/* Labels */}
            <div className="absolute -bottom-6 w-full flex justify-between text-xs text-slate-500">
                <span>{min.toFixed(1)}</span>
                <span>{max.toFixed(1)}</span>
            </div>
        </div>
    );
};
