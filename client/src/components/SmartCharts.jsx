import React, { useEffect, useState } from 'react';
import { getChartRecommendations } from '../services/api';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ScatterChart, Scatter, Cell
} from 'recharts';
import { Loader, BarChart2, ScatterChart as ScatterIcon, TrendingUp } from 'lucide-react';
import { useData } from '../context/DataContext';

export default function SmartCharts() {
    const { dataSummary } = useData();
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (dataSummary) {
            fetchRecommendations();
        }
    }, [dataSummary]);

    const fetchRecommendations = async () => {
        setLoading(true);
        try {
            const res = await getChartRecommendations();
            if (res.recommendations) {
                setRecommendations(res.recommendations);
            }
        } catch (err) {
            console.error(err);
            // Don't show error to user, just hide section if fails or empty
            setError('Failed to load recommendations');
        } finally {
            setLoading(false);
        }
    };

    if (!dataSummary || loading || recommendations.length === 0) return null;

    return (
        <section className="space-y-6">
            <h2 className="text-xl font-bold text-slate-800 flex items-center">
                <TrendingUp className="mr-2 text-indigo-600" size={24} />
                Smart Recommendations
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {recommendations.map((rec) => (
                    <div key={rec.id} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                        <div className="mb-4">
                            <h3 className="font-semibold text-slate-800">{rec.title}</h3>
                            <p className="text-sm text-slate-500">{rec.description}</p>
                        </div>

                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                {renderChart(rec)}
                            </ResponsiveContainer>
                        </div>
                    </div>
                ))}
            </div>
        </section>
    );
}

const renderChart = (rec) => {
    switch (rec.chartType) {
        case 'bar':
            return (
                <BarChart data={rec.data}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey={rec.x} tick={{ fontSize: 10 }} interval={0} />
                    <YAxis />
                    <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    <Bar dataKey={rec.y} fill="#6366f1" radius={[4, 4, 0, 0]} animationDuration={1000} />
                </BarChart>
            );
        case 'scatter':
            return (
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid />
                    <XAxis type="number" dataKey={rec.x} name={rec.x} tick={{ fontSize: 10 }} />
                    <YAxis type="number" dataKey={rec.y} name={rec.y} tick={{ fontSize: 10 }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    <Scatter name={rec.title} data={rec.data} fill="#8b5cf6" />
                </ScatterChart>
            );
        default:
            return <div className="flex items-center justify-center h-full text-slate-400">Unsupported Chart Type</div>;
    }
};
