```
import React, { useState, useEffect, useRef } from 'react';
import { useData } from '../context/DataContext';
import { saveDashboard, listDashboards, getDashboard } from '../services/api';
import { Plus, Save, Layout, Trash2, Grid } from 'lucide-react';
import ReportGenerator from '../components/ReportGenerator';
import {
    BarChart, Bar, LineChart, Line, PieChart as RePie, Pie, Cell, AreaChart, Area,
    ScatterChart, Scatter, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    RadialBarChart, RadialBar, Treemap,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

import { Responsive } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

// --- Custom Width Provider HOC ---
// Replacing broken WidthProvider from react-grid-layout ESM build
const withWidth = (Component) => {
    return (props) => {
        const [width, setWidth] = useState(1200);
        const ref = useRef(null);

        useEffect(() => {
            if (!ref.current) return;

            const measure = () => {
                if (ref.current) {
                    setWidth(ref.current.offsetWidth);
                }
            };

            // Initial measure
            measure();

            // Web API for resize detection
            const resizeObserver = new ResizeObserver(() => measure());
            resizeObserver.observe(ref.current);

            // Fallback window resize listener
            window.addEventListener('resize', measure);

            return () => {
                resizeObserver.disconnect();
                window.removeEventListener('resize', measure);
            };
        }, []);

        return (
            <div ref={ref} className={props.className} style={{ width: '100%', minHeight: '100px' }}>
                <Component {...props} width={width} />
            </div>
        );
    };
};

const ResponsiveGridLayout = withWidth(Responsive);

const COLORS = [
    '#6366f1', // Indigo
    '#ec4899', // Pink
    '#10b981', // Emerald
    '#f59e0b', // Amber
    '#8b5cf6', // Violet
    '#06b6d4', // Cyan
    '#f43f5e', // Rose
    '#84cc16'  // Lime
];

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("Dashboard Error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="p-8 text-red-600 bg-red-50 rounded-xl m-8 border border-red-200">
                    <h2 className="text-xl font-bold mb-2">Something went wrong.</h2>
                    <p className="mb-4 text-sm text-red-800">Please try refreshing the page or checking your widget configuration.</p>
                    <pre className="p-4 bg-white rounded border border-red-100 text-xs font-mono overflow-auto max-h-64 shadow-inner">
                        {this.state.error?.toString()}
                        {this.state.error?.stack}
                    </pre>
                </div>
            );
        }
        return this.props.children;
    }
}

function DashboardContent() {
    const { dataSummary, dataPreview } = useData();
    const [dashboards, setDashboards] = useState([]);
    const [currentLayout, setCurrentLayout] = useState([]);
    const [dashboardName, setDashboardName] = useState('My Dashboard');
    const [showAddModal, setShowAddModal] = useState(false);
    const [mounted, setMounted] = useState(false);

    // Widget Config State
    const [newWidget, setNewWidget] = useState({
        type: 'bar',
        title: 'New Chart',
        xAxis: '',
        yAxis: '',
        aggregation: 'count',
        stacked: false,
        horizontal: false
    });

    useEffect(() => {
        setMounted(true);
        loadDashboards();
    }, []);

    const loadDashboards = async () => {
        try {
            const list = await listDashboards();
            setDashboards(list);
        } catch (err) {
            console.error("Failed to load dashboards", err);
        }
    };

    const handleSave = async () => {
        if (!currentLayout.length) return;
        try {
            await saveDashboard(dashboardName, currentLayout);
            alert("Dashboard saved!");
            loadDashboards();
        } catch (err) {
            console.error(err);
            const msg = err.response?.data?.error || err.response?.data?.detail || err.message;
            alert(`Failed to save: ${ msg } `);
        }
    };

    const handleLoad = async (id) => {
        try {
            const dash = await getDashboard(id);
            if (dash) {
                setDashboardName(dash.name);
                setCurrentLayout(dash.layout);
            }
        } catch (err) {
            console.error(err);
        }
    };

    const addWidget = () => {
        const newItem = {
            ...newWidget,
            id: Date.now(),
            x: (currentLayout.length * 2) % 4,
            y: Infinity, // puts it at the bottom
            w: 2,
            h: 3
        };
        setCurrentLayout([...currentLayout, newItem]);
        setShowAddModal(false);
    };

    const removeWidget = (id) => {
        setCurrentLayout(currentLayout.filter(w => w.id !== id));
    };

    // Helper to process data for charts
    const processData = (config) => {
        if (!dataPreview) return [];

        // 1. SCATTER PLOT (Raw X/Y values)
        if (config.type === 'scatter') {
            return dataPreview
                .map(row => ({
                    x: Number(row[config.xAxis] || 0),
                    y: Number(row[config.yAxis] || 0),
                    z: Number(row[config.zAxis] || 1) // Bubble size
                }))
                .filter(d => !isNaN(d.x) && !isNaN(d.y));
        }

        // 3. AGGREGATION (Bar, Line, Area, Pie, Radar, Treemap)
        const counts = {};

        dataPreview.forEach(row => {
            const key = String(row[config.xAxis] || 'Unknown');
            if (!counts[key]) counts[key] = { name: key, value: 0 };

            if (config.aggregation === 'count') {
                counts[key].value += 1;
            } else if (config.aggregation === 'sum') {
                counts[key].value += Number(row[config.yAxis] || 0);
            } else if (config.aggregation === 'avg') {
                // simplified avg
                counts[key].value += Number(row[config.yAxis] || 0);
            }
        });

        const result = Object.values(counts);

        // Sorting usually helps visual hierarchy
        if (config.type !== 'line') {
            result.sort((a, b) => b.value - a.value);
        }

        return result.slice(0, 50); // increased limit
    };

    if (!dataSummary) return <div className="p-8">Please load a dataset first.</div>;
    if (!mounted) return <div className="p-8">Loading dashboard...</div>;

    const columns = dataSummary.columns;

    return (
        <div id="dashboard-content" className="p-8 max-w-7xl mx-auto space-y-8">
            <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-slate-200">
                <div className="flex items-center gap-4">
                    <Layout className="text-indigo-600" />
                    <input
                        value={dashboardName}
                        onChange={(e) => setDashboardName(e.target.value)}
                        className="text-xl font-bold text-slate-800 border-none focus:ring-0"
                    />
                </div>
                <div className="flex gap-2">
                    <select
                        onChange={(e) => handleLoad(e.target.value)}
                        className="px-4 py-2 border rounded-lg text-sm"
                    >
                        <option value="">Load Dashboard...</option>
                        {dashboards.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                    </select>
                    <button onClick={() => setShowAddModal(true)} className="flex items-center gap-2 px-4 py-2 bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100 transition-colors">
                        <Plus size={18} /> Add Widget
                    </button>
                    <ReportGenerator targetId="dashboard-content" fileName={`${ dashboardName }.pdf`} />
                    <button onClick={handleSave} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 shadow-md transition-all">
                        <Save size={18} /> Save
                    </button>
                </div>
            </div>

            {/* Draggable Grid Layout */}
            <ResponsiveGridLayout
                className="layout"
                layouts={{ lg: currentLayout }}
                breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                cols={{ lg: 4, md: 4, sm: 2, xs: 1, xxs: 1 }}
                rowHeight={100}
                onLayoutChange={(layout) => {
                    const updated = currentLayout.map(w => {
                        const match = layout.find(l => String(l.i) === String(w.id));
                        return match ? { ...w, ...match } : w;
                    });
                    if (JSON.stringify(updated) !== JSON.stringify(currentLayout)) {
                        setCurrentLayout(updated);
                    }
                }}
            >
                {currentLayout.map((widget) => {
                    const gridItemProps = {
                        key: String(widget.id),
                        "data-grid": {
                            x: widget.x || 0,
                            y: widget.y || 0,
                            w: widget.w || 2,
                            h: widget.h || 3,
                            i: String(widget.id)
                        }
                    };

                    return (
                        <div key={widget.id} {...gridItemProps} className="bg-white p-2 rounded-xl shadow-sm border border-slate-100 flex flex-col group hover:shadow-md transition-shadow overflow-hidden">
                            <div className="flex justify-between items-start mb-1 px-1 cursor-grab active:cursor-grabbing handle">
                                <h3 className="font-semibold text-slate-700 text-sm truncate pr-2">{widget.title}</h3>
                                <button
                                    onMouseDown={(e) => e.stopPropagation()}
                                    onClick={() => removeWidget(widget.id)}
                                    className="text-slate-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                            <div className="flex-1 min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    {widget.type === 'bar' && (
                                        <BarChart data={processData(widget)} layout={widget.horizontal ? 'vertical' : 'horizontal'} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                            <XAxis
                                                type={widget.horizontal ? 'number' : 'category'}
                                                dataKey={widget.horizontal ? undefined : 'name'}
                                                tick={{ fontSize: 10, angle: -45, textAnchor: 'end' }}
                                                height={50}
                                                interval={0}
                                            />
                                            <YAxis
                                                type={widget.horizontal ? 'category' : 'number'}
                                                dataKey={widget.horizontal ? 'name' : undefined}
                                                width={60}
                                                tick={{ fontSize: 10 }}
                                            />
                                            <Tooltip />
                                            {widget.stacked ? (
                                                <Bar dataKey="value" stackId="a" fill="#6366f1" radius={[0, 4, 4, 0]} />
                                            ) : (
                                                <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} />
                                            )}
                                            <Legend wrapperStyle={{ fontSize: '10px' }} />
                                        </BarChart>
                                    )}
                                    {widget.type === 'line' && (
                                        <LineChart data={processData(widget)} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis
                                                dataKey="name"
                                                tick={{ fontSize: 10, angle: -45, textAnchor: 'end' }}
                                                height={50}
                                            />
                                            <YAxis tick={{ fontSize: 10 }} width={40} />
                                            <Tooltip />
                                            <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                                            <Legend wrapperStyle={{ fontSize: '10px' }} />
                                        </LineChart>
                                    )}
                                    {widget.type === 'area' && (
                                        <AreaChart data={processData(widget)} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis
                                                dataKey="name"
                                                tick={{ fontSize: 10, angle: -45, textAnchor: 'end' }}
                                                height={50}
                                            />
                                            <YAxis tick={{ fontSize: 10 }} width={40} />
                                            <Tooltip />
                                            <Area type="monotone" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                                            <Legend wrapperStyle={{ fontSize: '10px' }} />
                                        </AreaChart>
                                    )}
                                    {widget.type === 'scatter' && (
                                        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                                            <CartesianGrid />
                                            <XAxis type="number" dataKey="x" name={widget.xAxis} tick={{ fontSize: 10 }} />
                                            <YAxis type="number" dataKey="y" name={widget.yAxis} tick={{ fontSize: 10 }} />
                                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                            <Scatter name={widget.title} data={processData(widget)} fill="#ec4899" />
                                        </ScatterChart>
                                    )}
                                    {widget.type === 'radar' && (
                                        <RadarChart outerRadius="65%" data={processData(widget)}>
                                            <PolarGrid />
                                            <PolarAngleAxis dataKey="name" tick={{ fontSize: 10 }} />
                                            <PolarRadiusAxis angle={30} domain={[0, 'auto']} tick={{ fontSize: 10 }} />
                                            <Radar name={widget.title} dataKey="value" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.6} />
                                            <Legend wrapperStyle={{ fontSize: '10px' }} />
                                        </RadarChart>
                                    )}
                                    {widget.type === 'pie' && (
                                        <RePie>
                                            <Pie
                                                data={processData(widget)}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius="40%"
                                                outerRadius="70%"
                                                paddingAngle={5}
                                                dataKey="value"
                                                label={({ percent }) => `${ (percent * 100).toFixed(0) }% `}
                                                labelLine={false}
                                            >
                                                {processData(widget).map((entry, index) => (
                                                    <Cell key={`cell - ${ index } `} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                            <Tooltip />
                                            <Legend verticalAlign="bottom" height={20} iconSize={10} wrapperStyle={{ fontSize: '10px' }} />
                                        </RePie>
                                    )}
                                    {widget.type === 'treemap' && (
                                        <Treemap
                                            data={processData(widget)}
                                            dataKey="value"
                                            aspectRatio={4 / 3}
                                            stroke="#fff"
                                            fill="#06b6d4"
                                        >
                                            <Tooltip />
                                        </Treemap>
                                    )}
                                    {widget.type === 'radial' && (
                                        <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="70%" barSize={20} data={processData(widget)}>
                                            <RadialBar
                                                minAngle={15}
                                                label={{ position: 'insideStart', fill: '#fff' }}
                                                background
                                                clockWise
                                                dataKey="value"
                                            />
                                            <Legend iconSize={10} layout="vertical" verticalAlign="middle" wrapperStyle={{ fontSize: '10px', right: 0 }} />
                                            <Tooltip />
                                        </RadialBarChart>
                                    )}
                                </ResponsiveContainer>
                            </div>
                        </div>
                    );
                })}
            </ResponsiveGridLayout>

            {currentLayout.length === 0 && (
                <div className="flex flex-col items-center justify-center p-12 border-2 border-dashed border-slate-200 rounded-xl text-slate-400">
                    <Grid size={48} className="mb-4 text-slate-300" />
                    <p>No widgets yet. Click "Add Widget" to build your dashboard.</p>
                </div>
            )}

            {/* Add Widget Modal */}
            {showAddModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm">
                    <div className="bg-white rounded-xl p-8 max-w-md w-full shadow-2xl transform scale-100 transition-all">
                        <h2 className="text-xl font-bold mb-6 text-slate-800">Configs Widget</h2>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Title</label>
                                <input
                                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none"
                                    value={newWidget.title}
                                    onChange={e => setNewWidget({ ...newWidget, title: e.target.value })}
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">Chart Type</label>
                                    <select
                                        className="w-full p-2 border rounded-lg"
                                        value={newWidget.type}
                                        onChange={e => setNewWidget({ ...newWidget, type: e.target.value })}
                                    >
                                        <option value="bar">Bar Chart</option>
                                        <option value="line">Line Chart</option>
                                        <option value="area">Area Chart</option>
                                        <option value="pie">Pie Chart</option>
                                        <option value="scatter">Scatter Plot</option>
                                        <option value="radar">Radar Chart</option>
                                        <option value="radial">Radial Bar</option>
                                        <option value="treemap">Treemap</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">Aggregation</label>
                                    <select
                                        className="w-full p-2 border rounded-lg"
                                        value={newWidget.aggregation}
                                        onChange={e => setNewWidget({ ...newWidget, aggregation: e.target.value })}
                                        disabled={newWidget.type === 'scatter'}
                                    >
                                        <option value="count">Count Rows</option>
                                        <option value="sum">Sum Values</option>
                                        <option value="avg">Average</option>
                                    </select>
                                </div>
                            </div>

                            {/* Visual Options */}
                            <div className="flex gap-4">
                                <label className="flex items-center gap-2 text-sm text-slate-600">
                                    <input
                                        type="checkbox"
                                        checked={newWidget.stacked}
                                        onChange={e => setNewWidget({ ...newWidget, stacked: e.target.checked })}
                                        className="rounded text-indigo-600 focus:ring-indigo-500"
                                    />
                                    Stacked
                                </label>
                                <label className="flex items-center gap-2 text-sm text-slate-600">
                                    <input
                                        type="checkbox"
                                        checked={newWidget.horizontal}
                                        onChange={e => setNewWidget({ ...newWidget, horizontal: e.target.checked })}
                                        className="rounded text-indigo-600 focus:ring-indigo-500"
                                        disabled={newWidget.type !== 'bar'}
                                    />
                                    Horizontal
                                </label>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">X-Axis ({newWidget.type === 'scatter' ? 'Numeric' : 'Category'})</label>
                                <select
                                    className="w-full p-2 border rounded-lg"
                                    value={newWidget.xAxis}
                                    onChange={e => setNewWidget({ ...newWidget, xAxis: e.target.value })}
                                >
                                    <option value="">Select Column...</option>
                                    {columns.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </div>

                            {newWidget.aggregation === 'sum' && (
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">Y-Axis (Value)</label>
                                    <select
                                        className="w-full p-2 border rounded-lg"
                                        value={newWidget.yAxis}
                                        onChange={e => setNewWidget({ ...newWidget, yAxis: e.target.value })}
                                    >
                                        <option value="">Select Column...</option>
                                        {columns.map(c => <option key={c} value={c}>{c}</option>)}
                                    </select>
                                </div>
                            )}
                        </div>

                        <div className="flex justify-end gap-3 mt-8">
                            <button onClick={() => setShowAddModal(false)} className="px-4 py-2 text-slate-600 hover:bg-slate-50 rounded-lg">Cancel</button>
                            <button
                                onClick={addWidget}
                                disabled={!newWidget.xAxis || (newWidget.aggregation === 'sum' && !newWidget.yAxis)}
                                className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 shadow-md"
                            >
                                Add Widget
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default function CustomDashboard() {
    return (
        <ErrorBoundary>
            <DashboardContent />
        </ErrorBoundary>
    );
}
