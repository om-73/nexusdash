import React from 'react';
import { TrendingUp, DollarSign, Activity, Hash, Tag } from 'lucide-react';

const KPICard = ({ kpi }) => {
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
        <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm flex items-center gap-4 hover:shadow-md transition-shadow">
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center bg-${color}-50 text-${color}-500`}>
                <Icon size={24} />
            </div>
            <div>
                <p className="text-slate-500 text-sm font-medium">{kpi.label}</p>
                <h3 className="text-2xl font-bold text-slate-800">{formatValue(kpi.value, kpi.format)}</h3>
            </div>
        </div>
    );
};

export default function KPIGrid({ kpis = [] }) {
    if (!kpis || kpis.length === 0) return null;

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {kpis.map((kpi, idx) => (
                <KPICard key={kpi.id || idx} kpi={kpi} />
            ))}
        </div>
    );
}
