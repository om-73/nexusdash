import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import { Link } from 'react-router-dom';
import { Database, TrendingUp, AlertOctagon, Layers, Search, Sparkles } from 'lucide-react';
import { queryMachineLearning, getKPIs } from '../services/api';
import SmartCharts from '../components/SmartCharts';
import ReportGenerator from '../components/ReportGenerator';
import KPIGrid from '../components/KPIGrid';

const StatCard = ({ icon: Icon, label, value, color }) => (
    <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm flex items-center gap-4">
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center bg-${color}-50 text-${color}-500`}>
            <Icon size={24} />
        </div>
        <div>
            <p className="text-slate-500 text-sm font-medium">{label}</p>
            <h3 className="text-2xl font-bold text-slate-800">{value}</h3>
        </div>
    </div>
);

export default function Dashboard() {
    const { dataSummary, dataPreview } = useData();
    const [kpis, setKpis] = useState([]);

    React.useEffect(() => {
        if (dataSummary) {
            getKPIs().then(setKpis).catch(console.error);
        }
    }, [dataSummary]);

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
            <div className="p-12 flex flex-col items-center justify-center min-h-[60vh] text-center">
                <div className="w-24 h-24 bg-slate-100 rounded-full flex items-center justify-center mb-6 text-slate-400">
                    <Database size={48} />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mb-2">No Data Loaded</h2>
                <p className="text-slate-500 max-w-md mb-8">
                    Upload a dataset to start visualizing and analyzing your data.
                </p>
                <Link
                    to="/load"
                    className="px-6 py-3 bg-primary text-white rounded-xl font-medium shadow-lg shadow-primary/25 hover:shadow-xl hover:bg-blue-600 transition-all"
                >
                    Connect Data Source
                </Link>
            </div>
        );
    }

    const { shape, missing_values, columns } = dataSummary;
    const totalCells = shape[0] * shape[1];
    const totalMissing = Object.values(missing_values).reduce((a, b) => a + b, 0);
    const missingPercentage = ((totalMissing / totalCells) * 100).toFixed(1);

    return (
        <div className="p-8" id="dashboard-container">
            <div className="flex items-center justify-between mb-8" data-html2canvas-ignore>
                <div>
                    <h1 className="text-2xl font-bold text-slate-800">Dataset Overview</h1>
                    <p className="text-slate-500">Analysis of recent upload</p>
                </div>
                <div className="flex gap-2">
                    <ReportGenerator targetId="dashboard-container" />
                    <Link to="/load" className="px-4 py-2 text-primary bg-blue-50 hover:bg-blue-100 rounded-lg font-medium transition-colors">
                        Change Dataset
                    </Link>
                    <Link to="/dashboard/custom" className="px-4 py-2 text-indigo-600 bg-indigo-50 hover:bg-indigo-100 rounded-lg font-medium transition-colors">
                        Custom Dashboard
                    </Link>
                    <Link to="/clean" className="px-4 py-2 bg-primary text-white rounded-lg font-medium shadow-md shadow-primary/20 transition-colors">
                        Clean Data
                    </Link>
                </div>
            </div>

            {/* KPI Section */}
            <KPIGrid kpis={kpis} />

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

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard icon={Layers} label="Total Rows" value={shape[0].toLocaleString()} color="blue" />
                <StatCard icon={Database} label="Columns" value={shape[1]} color="indigo" />
                <StatCard icon={AlertOctagon} label="Missing Values" value={totalMissing} color="amber" />
                <StatCard icon={TrendingUp} label="Data Quality" value={`${100 - missingPercentage}%`} color="emerald" />
            </div>

            {/* Data Preview Table */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50">
                    <h3 className="font-bold text-slate-800">Data Preview (First 50 rows)</h3>
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
                            {dataPreview.map((row, i) => (
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
        </div >
    );
}
