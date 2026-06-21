import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import { Link } from 'react-router-dom';
import { Database, TrendingUp, AlertOctagon, Layers, Search, Sparkles, Upload, ArrowRight, Table2, AlertTriangle, Activity, FileText, CheckCircle } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, CartesianGrid, XAxis, YAxis } from 'recharts';
import { queryMachineLearning, getKPIs } from '../services/api';
import SmartCharts from '../components/SmartCharts';
import ReportGenerator from '../components/ReportGenerator';
import KPIGrid from '../components/KPIGrid';

export default function Dashboard() {
    const { dataSummary, dataPreview } = useData();
    // Load saved KPIs from local storage or rely on effect to fetch defaults
    // Load saved KPIs from local storage
    const [kpis, setKpis] = useState(() => {
        try {
            const saved = localStorage.getItem('dashboard_kpis');
            if (saved) {
                const parsed = JSON.parse(saved);
                // Support both legacy array and new object format
                return Array.isArray(parsed) ? parsed : (parsed.kpis || []);
            }
            return [];
        } catch (e) {
            return [];
        }
    });

    // 1. Validate KPIs against current dataset signature
    React.useEffect(() => {
        if (dataSummary) {
            const saved = localStorage.getItem('dashboard_kpis');
            if (saved) {
                try {
                    const parsed = JSON.parse(saved);
                    const currentSig = dataSummary.columns.join(',');
                    const savedSig = Array.isArray(parsed) ? 'legacy' : parsed.signature;

                    // If we have a stored signature and it doesn't match, or if it's legacy and columns don't match strict logic
                    if (savedSig !== currentSig) {
                        // Mismatch detected - clear KPIs to trigger auto-fetch
                        setKpis([]);
                    }
                } catch (e) {
                    console.error("Error validating KPIs:", e);
                }
            }
        }
    }, [dataSummary]); // Only run when dataset signature changes

    // 2. Fetch initial KPIs if none exist (fresh load or reset)
    React.useEffect(() => {
        if (dataSummary && kpis.length === 0) {
            getKPIs().then(setKpis).catch(console.error);
        }
    }, [dataSummary, kpis.length]);

    // 3. Persist KPIs with dataset signature
    React.useEffect(() => {
        if (kpis.length > 0 && dataSummary) {
            const signature = dataSummary.columns.join(',');
            localStorage.setItem('dashboard_kpis', JSON.stringify({ kpis, signature }));
        }
    }, [kpis, dataSummary]);

    // Function to reset KPIs to auto-detected defaults
    const resetKpis = () => {
        localStorage.removeItem('dashboard_kpis');
        setKpis([]); // This will trigger the effect above to re-fetch
    };

    // NL Query State
    const [query, setQuery] = useState('');
    const [queryResult, setQueryResult] = useState(null);
    const [queryLoading, setQueryLoading] = useState(false);
    const [queryError, setQueryError] = useState(null);

    const handleQuery = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;
        setQueryLoading(true);
        setQueryError(null);
        setQueryResult(null);
        try {
            const res = await queryMachineLearning(query);
            setQueryResult(res);
        } catch (err) {
            console.error(err);
            setQueryError('Failed to process query. Please try asking differently.');
        } finally {
            setQueryLoading(false);
        }
    };

    if (!dataSummary) {
        return (
            <div className="p-6 md:p-12 text-center text-slate-500 min-h-[50vh] flex flex-col items-center justify-center">
                <Database size={48} className="mb-4 text-slate-300" />
                <h2 className="text-xl font-medium mb-2 text-slate-700">No Data Available</h2>
                <p className="mb-6">Please load a dataset to view the dashboard.</p>
                <Link to="/" className="px-6 py-2 bg-primary text-white rounded-lg hover:bg-blue-600 font-medium">
                    Go to Data Source
                </Link>
            </div>
        );
    }

    const { shape, missing_values, columns } = dataSummary;
    const totalCells = shape[0] * shape[1];
    const totalMissing = Object.values(missing_values).reduce((a, b) => a + b, 0);
    const missingPercentage = ((totalMissing / totalCells) * 100).toFixed(1);

    // Data for charts
    const dataTypeCounts = Object.entries(dataSummary.dtypes).reduce((acc, [col, type]) => {
        const existing = acc.find(item => item.name === type);
        if (existing) {
            existing.value += 1;
        } else {
            acc.push({ name: type, value: 1 });
        }
        return acc;
    }, []);

    const missingDataChart = Object.entries(dataSummary.missing_values)
        .filter(([, count]) => count > 0)
        .map(([col, count]) => ({ name: col, missing: count }));

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

    return (
        <div className="p-4 md:p-8 max-w-7xl mx-auto space-y-6 md:space-y-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-200 pb-6">
                <div>
                    <h1 className="text-2xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600">
                        Dataset Overview
                    </h1>
                    <p className="text-slate-500 mt-1">High-level summary of your current data</p>
                </div>
                <div className="flex flex-wrap gap-3">
                    <Link to="/" className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 text-slate-600 font-medium transition-colors text-sm md:text-base">
                        <Upload size={18} /> Change Dataset
                    </Link>
                    <Link to="/clean" className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-blue-600 shadow-sm shadow-blue-200 font-medium transition-colors text-sm md:text-base">
                        Clean Data <ArrowRight size={18} />
                    </Link>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
                <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Database size={64} className="text-blue-500" />
                    </div>
                    <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Total Rows</p>
                    <p className="text-4xl font-black text-slate-800">{dataSummary.shape[0].toLocaleString()}</p>
                </div>

                <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Table2 size={64} className="text-emerald-500" />
                    </div>
                    <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Total Columns</p>
                    <p className="text-4xl font-black text-slate-800">{dataSummary.shape[1]}</p>
                </div>

                <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <AlertTriangle size={64} className="text-amber-500" />
                    </div>
                    <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Missing Values</p>
                    <p className="text-4xl font-black text-amber-600">{Object.values(dataSummary.missing_values).reduce((a, b) => a + b, 0).toLocaleString()}</p>
                </div>

                <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Activity size={64} className="text-purple-500" />
                    </div>
                    <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Memory Usage</p>
                    <p className="text-4xl font-black text-slate-800">{dataSummary.memory_usage}</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
                {/* Data Types Summary */}
                <div className="bg-white p-4 md:p-6 rounded-2xl border border-slate-200 shadow-sm">
                    <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                        <FileText className="text-blue-500" size={20} /> Column Types
                    </h2>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                            <PieChart>
                                <Pie
                                    data={dataTypeCounts}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {dataTypeCounts.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                />
                                <Legend verticalAlign="bottom" height={36} iconType="circle" />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Missing Values Chart */}
                <div className="bg-white p-4 md:p-6 rounded-2xl border border-slate-200 shadow-sm">
                    <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                        <AlertTriangle className="text-amber-500" size={20} /> Missing Values by Column
                    </h2>
                    {missingDataChart.length > 0 ? (
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                                <BarChart data={missingDataChart} margin={{ top: 5, right: 20, bottom: 25, left: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                    <XAxis
                                        dataKey="name"
                                        tick={{ fill: '#64748b', fontSize: 12 }}
                                        tickLine={false}
                                        axisLine={false}
                                    />
                                    <YAxis
                                        tick={{ fill: '#64748b', fontSize: 12 }}
                                        tickLine={false}
                                        axisLine={false}
                                    />
                                    <Tooltip
                                        cursor={{ fill: '#f8fafc' }}
                                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                    />
                                    <Bar dataKey="missing" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div className="h-64 flex flex-col items-center justify-center text-emerald-500 bg-emerald-50 rounded-xl border border-emerald-100">
                            <CheckCircle size={48} className="mb-2" />
                            <p className="font-medium">No missing values found!</p>
                        </div>
                    )}
                </div>
            </div>

            {/* KPI Section */}
            <KPIGrid kpis={kpis} setKpis={setKpis} columns={columns} dtypes={dataSummary.dtypes} onReset={resetKpis} />

            {/* NL Query Section */}
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-8 mb-8 text-white shadow-lg">
                <h2 className="text-2xl font-bold mb-2 flex items-center">
                    <Sparkles className="mr-2 text-yellow-300" />
                    Ask your data
                </h2>
                <p className="text-indigo-100 mb-6">Use natural language to filter and explore your dataset.</p>

                <form onSubmit={handleQuery} className="relative max-w-2xl">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="e.g. 'Show rows where age > 30' or 'Find customers from New York'"
                        className="w-full px-6 py-4 rounded-full text-slate-800 focus:outline-none focus:ring-4 focus:ring-indigo-300 shadow-xl"
                    />
                    <button
                        type="submit"
                        disabled={queryLoading}
                        className="absolute right-2 top-2 bg-indigo-600 hover:bg-indigo-700 text-white p-2 rounded-full transition-colors disabled:opacity-50"
                    >
                        <Search size={24} />
                    </button>
                </form>

                {/* Query Results */}
                {queryError && <p className="mt-4 text-red-200 bg-red-900/20 p-2 rounded inline-block">{queryError}</p>}

                {queryResult && (
                    <div className="mt-6 bg-white/10 rounded-xl p-4 backdrop-blur-sm border border-white/20">
                        <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold">Query Results ({queryResult.shape[0]} rows found)</h3>
                            <button onClick={() => setQueryResult(null)} className="text-xs text-indigo-200 hover:text-white">Clear</button>
                        </div>
                        <div className="overflow-x-auto max-h-64 rounded-lg border border-white/10">
                            <table className="w-full text-left text-sm text-indigo-50">
                                <thead className="bg-black/20 text-xs uppercase">
                                    <tr>
                                        {queryResult.columns.map(col => <th key={col} className="px-4 py-2">{col}</th>)}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/10">
                                    {queryResult.preview.map((row, i) => (
                                        <tr key={i} className="hover:bg-white/5">
                                            {queryResult.columns.map(col => <td key={col} className="px-4 py-2 whitespace-nowrap">{row[col]}</td>)}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>

            {/* Smart Charts Section */}
            <div className="mb-8">
                <SmartCharts />
            </div>

            {/* Data Preview Modal / Table Preview */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4 md:p-6 border-b border-slate-200 bg-slate-50">
                    <h2 className="text-lg font-bold text-slate-800">Data Preview (First 5 Rows)</h2>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left text-slate-600">
                        <thead className="text-xs text-slate-500 uppercase bg-slate-50 border-b border-slate-200">
                            <tr>
                                {dataSummary.columns.map(col => (
                                    <th key={col} className="px-4 md:px-6 py-3 font-semibold whitespace-nowrap">{col}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {dataPreview?.slice(0, 5).map((row, i) => (
                                <tr key={i} className="bg-white border-b border-slate-100 hover:bg-slate-50 transition-colors">
                                    {dataSummary.columns.map(col => (
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
