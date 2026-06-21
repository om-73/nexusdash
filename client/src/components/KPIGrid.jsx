import React, { useState } from 'react';
import { TrendingUp, DollarSign, Activity, Hash, Tag, Edit2, Plus, Trash2, X, Check, Save } from 'lucide-react';
import { calculateKPI } from '../services/api';

const KPICard = ({ kpi, onDelete, isEditing }) => {
    let Icon = Activity;
    let color = 'blue';

    if (kpi.format === 'currency') {
        Icon = DollarSign;
        color = 'emerald';
    } else if (kpi.format === 'percent') {
        Icon = TrendingUp;
        color = 'purple';
    } else if (kpi.format === 'text') {
        Icon = Tag;
        color = 'amber';
    } else {
        Icon = Hash;
        color = 'indigo'; // Default number
    }

    const formatValue = (val, format) => {
        if (val === null || val === undefined) return '-';
        if (format === 'currency') {
            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);
        }
        if (format === 'percent') {
            return `${val.toFixed(1)}%`;
        }
        if (typeof val === 'number') {
            return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
        }
        return val;
    };

    return (
        <div className="relative bg-white p-6 rounded-xl border border-slate-100 shadow-sm flex items-center gap-4 hover:shadow-md transition-shadow group">
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center bg-${color}-50 text-${color}-500`}>
                <Icon size={24} />
            </div>
            <div>
                <p className="text-slate-500 text-sm font-medium">{kpi.label}</p>
                <h3 className="text-2xl font-bold text-slate-800">{formatValue(kpi.value, kpi.format)}</h3>
            </div>

            {isEditing && (
                <button
                    onClick={() => onDelete(kpi)}
                    className="absolute top-2 right-2 p-1.5 bg-red-50 text-red-500 rounded-full hover:bg-red-100 opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Remove KPI"
                >
                    <Trash2 size={16} />
                </button>
            )}
        </div>
    );
};

import AddKPIModal from './AddKPIModal';

export default function KPIGrid({ kpis = [], setKpis, columns = [], dtypes = {}, onReset }) {
    const [isEditing, setIsEditing] = useState(false);
    const [showAddModal, setShowAddModal] = useState(false);

    if (!kpis && !isEditing) return null;

    const handleDelete = (kpiToDelete) => {
        if (setKpis) {
            // Filter based on ID if available, otherwise strict equality
            setKpis(prev => prev.filter(k => k.id ? k.id !== kpiToDelete.id : k !== kpiToDelete));
        }
    };

    const handleAdd = (newKpi) => {
        if (setKpis) {
            setKpis(prev => [...prev, newKpi]);
        }
    };

    return (
        <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-slate-700">Key Performance Indicators</h2>
                {setKpis && (
                    <div className="flex gap-2">
                        {isEditing && onReset && (
                            <button
                                onClick={() => {
                                    if (window.confirm('Reset to default KPIs?')) onReset();
                                }}
                                className="px-3 py-1.5 text-sm font-medium text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                            >
                                Reset Defaults
                            </button>
                        )}
                        <button
                            onClick={() => setIsEditing(!isEditing)}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${isEditing ? 'bg-indigo-100 text-indigo-700' : 'text-slate-500 hover:bg-slate-100'}`}
                        >
                            {isEditing ? <><Check size={16} /> Done</> : <><Edit2 size={16} /> Customize</>}
                        </button>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {kpis.map((kpi, idx) => (
                    <KPICard
                        key={kpi.id || idx}
                        kpi={kpi}
                        isEditing={isEditing}
                        onDelete={handleDelete}
                    />
                ))}

                {isEditing && (
                    <button
                        onClick={() => setShowAddModal(true)}
                        className="border-2 border-dashed border-slate-200 rounded-xl p-6 flex flex-col items-center justify-center gap-2 text-slate-400 hover:border-indigo-300 hover:text-indigo-500 hover:bg-indigo-50 transition-all min-h-[120px]"
                    >
                        <Plus size={32} />
                        <span className="font-medium">Add Metric</span>
                    </button>
                )}
            </div>

            <AddKPIModal
                isOpen={showAddModal}
                onClose={() => setShowAddModal(false)}
                columns={columns}
                dtypes={dtypes}
                onAdd={handleAdd}
            />
        </div>
    );
}
