import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { calculateKPI } from '../services/api';

const AddKPIModal = ({ isOpen, onClose, columns, dtypes, onAdd }) => {
    const [column, setColumn] = useState(columns[0] || '');
    const [operation, setOperation] = useState('sum');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Reset default operation when column changes based on type
    useEffect(() => {
        if (!column || !dtypes) return;
        const type = String(dtypes[column] || '');
        const isNumeric = type.includes('int') || type.includes('float');

        if (!isNumeric && (operation === 'sum' || operation === 'mean')) {
            setOperation('count');
        } else if (isNumeric && operation === 'count') {
            setOperation('sum');
        }
    }, [column, dtypes, operation]);

    if (!isOpen) return null;

    const handleAdd = async () => {
        setLoading(true);
        setError(null);
        try {
            const newKpi = await calculateKPI(column, operation);
            onAdd(newKpi);
            onClose();
        } catch (err) {
            console.error('KPI Calculation Error:', err);
            const detail = err.response?.data?.detail || err.response?.data?.error || err.message;
            setError(`Calculation Error: ${detail}`);
        } finally {
            setLoading(false);
        }
    };

    const currentType = dtypes ? String(dtypes[column] || '') : '';
    const isNumeric = currentType.includes('int') || currentType.includes('float');

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-sm">
            <div className="bg-white rounded-xl shadow-2xl p-6 w-full max-w-md">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-bold text-slate-800">Add New KPI</h3>
                    <button onClick={onClose}><X size={20} className="text-slate-400" /></button>
                </div>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Column</label>
                        <select
                            value={column}
                            onChange={e => setColumn(e.target.value)}
                            className="w-full p-2 border border-slate-300 rounded-lg"
                        >
                            {columns.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                        <p className="text-xs text-slate-400 mt-1">Type: {currentType}</p>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Operation</label>
                        <select
                            value={operation}
                            onChange={e => setOperation(e.target.value)}
                            className="w-full p-2 border border-slate-300 rounded-lg"
                        >
                            {/* Always allow Count/Unique */}
                            <option value="count">Count (Rows)</option>
                            <option value="unique">Unique Count</option>

                            {/* Numeric only (or coerced) */}
                            {isNumeric && (
                                <>
                                    <option value="sum">Sum</option>
                                    <option value="mean">Average</option>
                                    <option value="max">Max</option>
                                    <option value="min">Min</option>
                                </>
                            )}
                            {!isNumeric && (
                                <>
                                    <option value="sum">Sum (Try Convert)</option>
                                    <option value="mean">Average (Try Convert)</option>
                                    <option value="max">Max</option>
                                    <option value="min">Min</option>
                                </>
                            )}
                        </select>
                    </div>

                    {error && <div className="p-3 bg-red-50 text-red-600 text-sm rounded-lg border border-red-100">{error}</div>}

                    <button
                        onClick={handleAdd}
                        disabled={loading}
                        className="w-full py-2 bg-primary text-white rounded-lg font-medium hover:bg-blue-600 disabled:opacity-50"
                    >
                        {loading ? 'Calculating...' : 'Add KPI'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default AddKPIModal;
