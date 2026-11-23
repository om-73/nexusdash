import React, { useState, useEffect, useRef } from 'react';
import { useData } from '../context/DataContext';
import { useToast } from '../context/ToastContext';
import { saveDashboard, listDashboards, getDashboard, calculateKPI } from '../services/api';
import {
    Plus, Save, Layout, Trash2, Grid, TrendingUp, DollarSign, Hash,
    Users, Activity, ShoppingBag, CreditCard, Percent, Sparkles,
    Database, RefreshCw, RotateCcw, AlertTriangle, CheckCircle, BarChart2, Layers
} from 'lucide-react';
// ... (keep existing imports)

// KPI Card Component
const KPIWidget = ({ widget }) => {
    const [value, setValue] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const iconMap = {
        'trend': TrendingUp,
        'dollar': DollarSign,
        'hash': Hash,
        'users': Users,
        'activity': Activity,
        'shop': ShoppingBag,
        'card': CreditCard,
        'percent': Percent,
        'sparkles': Sparkles,
        'database': Database
    };

    const Icon = iconMap[widget.icon] || Hash;
    const colorMap = {
        'indigo': 'border-indigo-500 text-indigo-600 bg-indigo-50',
        'emerald': 'border-emerald-500 text-emerald-600 bg-emerald-50',
        'rose': 'border-rose-500 text-rose-600 bg-rose-50',
        'amber': 'border-amber-500 text-amber-600 bg-amber-50',
        'cyan': 'border-cyan-500 text-cyan-600 bg-cyan-50'
    };
    const theme = colorMap[widget.color] || colorMap['indigo'];
    const [borderColor, textColor, bgColor] = theme ? theme.split(' ') : ['border-indigo-500', 'text-indigo-600', 'bg-indigo-50'];

    useEffect(() => {
        if (widget.customValue !== undefined && widget.customValue !== null) {
            setValue(widget.customValue);
            return;
        }

        const skipKPI = ['model', 'score', 'importance', 'target'].includes(widget.aggregation);
        if (skipKPI) {
            setValue(widget.xAxis || '-');
            return;
        }

        const fetchKPI = async () => {
            if (!widget.xAxis && !widget.column) return; // Support generic column field
            const col = widget.xAxis || widget.column; // Reuse xAxis as column selector or add new field

            setLoading(true);
            try {
                const res = await calculateKPI(col, widget.aggregation || 'count');
                setValue(res.value);
            } catch (err) {
                console.error("KPI Error", err);
                setError("Error");
            } finally {
                setLoading(false);
            }
        };
        fetchKPI();
    }, [widget]);

    const formatValue = (val) => {
        if (val === null || val === undefined) return '-';
        if (typeof val === 'number') {
            return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
        }
        return val;
    };

    return (
        <div className={`relative flex flex-col justify-between h-full p-6 bg-white overflow-hidden border-l-4 ${borderColor}`}>
            {/* Background Icon Watermark */}
            <div className={`absolute -right-4 -bottom-4 opacity-10 ${textColor}`}>
                <Icon size={100} strokeWidth={1.5} />
            </div>

            <div className="flex justify-between items-start z-10">
                <div>
                    <h4 className="text-slate-500 text-sm font-semibold uppercase tracking-wider">{widget.title}</h4>
                    <p className="text-xs text-slate-400 mt-1 font-medium">
                        {widget.subtitle || (widget.customValue ? (widget.xAxis || '') : `${widget.aggregation} of ${widget.xAxis}`)}
                    </p>
                </div>
                <div className={`p-2 rounded-lg ${bgColor} ${textColor}`}>
                    <Icon size={24} />
                </div>
            </div>

            <div className="mt-4 z-10">
                {loading ? (
                    <div className="animate-pulse h-12 w-32 bg-slate-100 rounded"></div>
                ) : error ? (
                    <span className="text-red-500 text-sm font-medium">Unable to load</span>
                ) : (
                    <span className={`text-4xl md:text-5xl font-bold tracking-tight ${textColor}`} title={String(value)}>
                        {formatValue(value)}
                    </span>
                )}
            </div>
        </div>
    );
};

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

const generateSchemaAdaptedWidgets = (summary, preview, sourceTitle = 'Connected Data') => {
    if (!summary || !summary.columns || summary.columns.length === 0) {
        return { widgets: [], layouts: { lg: [], md: [], sm: [] } };
    }

    const cols = summary.columns;
    const dtypes = summary.dtypes || {};

    // Categorize columns
    const numericCols = cols.filter(c => {
        const t = String(dtypes[c] || '').toLowerCase();
        return t.includes('int') || t.includes('float') || t.includes('number');
    });
    const categoricalCols = cols.filter(c => !numericCols.includes(c));

    const totalRows = summary.shape ? summary.shape[0] : (preview ? preview.length : 0);
    const ts = Date.now();

    const newWidgets = [];

    // KPI 1: Total Records
    newWidgets.push({
        id: `kpi_records_${ts}`,
        type: 'kpi_card',
        title: 'Total Records',
        xAxis: cols[0],
        aggregation: 'count',
        customValue: Number(totalRows).toLocaleString(),
        icon: 'hash',
        color: 'indigo'
    });

    // KPI 2: Primary Metric
    if (numericCols.length > 0) {
        newWidgets.push({
            id: `kpi_num1_${ts}`,
            type: 'kpi_card',
            title: `Average ${numericCols[0]}`,
            xAxis: numericCols[0],
            aggregation: 'avg',
            icon: 'trend',
            color: 'emerald'
        });
    } else {
        newWidgets.push({
            id: `kpi_cols_${ts}`,
            type: 'kpi_card',
            title: 'Schema Attributes',
            xAxis: cols[0],
            aggregation: 'count',
            customValue: `${cols.length} Columns`,
            icon: 'database',
            color: 'emerald'
        });
    }

    // KPI 3: Secondary Metric
    if (numericCols.length > 1) {
        newWidgets.push({
            id: `kpi_num2_${ts}`,
            type: 'kpi_card',
            title: `Average ${numericCols[1]}`,
            xAxis: numericCols[1],
            aggregation: 'avg',
            icon: 'activity',
            color: 'amber'
        });
    } else if (categoricalCols.length > 0) {
        newWidgets.push({
            id: `kpi_cat1_${ts}`,
            type: 'kpi_card',
            title: `Dimension: ${categoricalCols[0]}`,
            xAxis: categoricalCols[0],
            aggregation: 'count',
            icon: 'users',
            color: 'amber'
        });
    } else {
        newWidgets.push({
            id: `kpi_shape_${ts}`,
            type: 'kpi_card',
            title: 'Data Integrity',
            xAxis: cols[0],
            aggregation: 'score',
            customValue: '100% Live',
            icon: 'sparkles',
            color: 'amber'
        });
    }

    // KPI 4: Segment or Category
    if (categoricalCols.length > 1) {
        newWidgets.push({
            id: `kpi_cat2_${ts}`,
            type: 'kpi_card',
            title: `Segment: ${categoricalCols[1]}`,
            xAxis: categoricalCols[1],
            aggregation: 'count',
            icon: 'users',
            color: 'cyan'
        });
    } else {
        newWidgets.push({
            id: `kpi_health_${ts}`,
            type: 'kpi_card',
            title: 'Connection Status',
            xAxis: cols[0],
            aggregation: 'score',
            customValue: 'Synchronized',
            icon: 'percent',
            color: 'rose'
        });
    }

    // Chart 1: Bar Chart
    const barX = categoricalCols.length > 0 ? categoricalCols[0] : cols[0];
    const barY = numericCols.length > 0 ? numericCols[0] : cols[0];
    newWidgets.push({
        id: `chart_bar_${ts}`,
        type: 'bar',
        title: `${barX} Breakdown`,
        xAxis: barX,
        yAxis: barY,
        aggregation: numericCols.length > 0 ? 'avg' : 'count',
        stacked: false,
        horizontal: false,
        color: 'indigo'
    });

    // Chart 2: Area / Trend Chart
    const areaX = categoricalCols.length > 1 ? categoricalCols[1] : barX;
    const areaY = numericCols.length > 0 ? numericCols[0] : cols[0];
    newWidgets.push({
        id: `chart_area_${ts}`,
        type: 'area',
        title: `${areaY} Distribution by ${areaX}`,
        xAxis: areaX,
        yAxis: areaY,
        aggregation: 'avg',
        color: 'emerald'
    });

    // Chart 3: Pie / Donut
    const pieCol = categoricalCols.length > 0 ? categoricalCols[0] : cols[0];
    newWidgets.push({
        id: `chart_pie_${ts}`,
        type: 'pie',
        title: `${pieCol} Composition`,
        xAxis: pieCol,
        yAxis: pieCol,
        aggregation: 'count',
        color: 'rose'
    });

    // Chart 4: Scatter or Line
    if (numericCols.length >= 2) {
        newWidgets.push({
            id: `chart_scatter_${ts}`,
            type: 'scatter',
            title: `${numericCols[0]} vs ${numericCols[1]} Signal Correlation`,
            xAxis: numericCols[0],
            yAxis: numericCols[1],
            aggregation: 'count',
            color: 'cyan'
        });
    } else {
        newWidgets.push({
            id: `chart_line_${ts}`,
            type: 'line',
            title: `${barY} Trajectory across ${barX}`,
            xAxis: barX,
            yAxis: barY,
            aggregation: numericCols.length > 0 ? 'avg' : 'count',
            color: 'cyan'
        });
    }

    const lg = [
        { i: newWidgets[0].id, x: 0, y: 0, w: 3, h: 2 },
        { i: newWidgets[1].id, x: 3, y: 0, w: 3, h: 2 },
        { i: newWidgets[2].id, x: 6, y: 0, w: 3, h: 2 },
        { i: newWidgets[3].id, x: 9, y: 0, w: 3, h: 2 },
        { i: newWidgets[4].id, x: 0, y: 2, w: 6, h: 4 },
        { i: newWidgets[5].id, x: 6, y: 2, w: 6, h: 4 },
        { i: newWidgets[6].id, x: 0, y: 6, w: 6, h: 4 },
        { i: newWidgets[7].id, x: 6, y: 6, w: 6, h: 4 }
    ];

    const md = [
        { i: newWidgets[0].id, x: 0, y: 0, w: 3, h: 2 },
        { i: newWidgets[1].id, x: 3, y: 0, w: 3, h: 2 },
        { i: newWidgets[2].id, x: 0, y: 2, w: 3, h: 2 },
        { i: newWidgets[3].id, x: 3, y: 2, w: 3, h: 2 },
        { i: newWidgets[4].id, x: 0, y: 4, w: 6, h: 4 },
        { i: newWidgets[5].id, x: 0, y: 8, w: 6, h: 4 },
        { i: newWidgets[6].id, x: 0, y: 12, w: 6, h: 4 },
        { i: newWidgets[7].id, x: 0, y: 16, w: 6, h: 4 }
    ];

    const sm = newWidgets.map((w, idx) => ({
        i: w.id,
        x: 0,
        y: idx * 3,
        w: 1,
        h: w.type === 'kpi_card' ? 2 : 3
    }));

    return {
        widgets: newWidgets,
        layouts: { lg, md, sm }
    };
};

function DashboardContent() {
    const {
        dataSummary,
        dataPreview,
        setDataPreview,
        agentDashboard,
        activeDatabaseConfig,
        refreshFromDatabase,
        uploadedDatasets,
        activeDatasetId
    } = useData();

    const { addToast } = useToast();
    const [dashboards, setDashboards] = useState([]);

    // Hydrate dataPreview if missing but dataSummary exists
    useEffect(() => {
        if (dataSummary && (!dataPreview || dataPreview.length === 0)) {
            import('../services/api').then(({ getState }) => {
                getState().then(res => {
                    if (res && res.preview && res.preview.length > 0 && setDataPreview) {
                        setDataPreview(res.preview);
                    }
                }).catch(() => {});
            });
        }
    }, [dataSummary, dataPreview, setDataPreview]);
    const [widgets, setWidgets] = useState([]);
    const [layouts, setLayouts] = useState({ lg: [], md: [], sm: [] });
    const [dashboardName, setDashboardName] = useState('My Dashboard');
    const [showAddModal, setShowAddModal] = useState(false);
    const [mounted, setMounted] = useState(false);
    const [dbRefreshing, setDbRefreshing] = useState(false);
    const initializedRef = useRef(false);

    // Widget Config State
    const [newWidget, setNewWidget] = useState({
        type: 'bar',
        title: 'New Chart',
        xAxis: '',
        yAxis: '',
        aggregation: 'count',
        stacked: false,
        horizontal: false,
        icon: 'hash',
        color: 'indigo'
    });

    useEffect(() => {
        setMounted(true);
        loadDashboards();
    }, []);

    // Auto-initialize dashboard layout based on AI Agent synthesis or connected database
    useEffect(() => {
        if (!mounted) return;

        if (agentDashboard && agentDashboard.widgets && agentDashboard.widgets.length > 0 && !initializedRef.current) {
            setWidgets(agentDashboard.widgets);
            setLayouts(agentDashboard.layouts || { lg: [], md: [], sm: [] });
            setDashboardName(agentDashboard.name || 'AI Synthesized Dashboard');
            initializedRef.current = true;
            return;
        }

        if (activeDatabaseConfig && dataSummary && dataSummary.columns && !initializedRef.current) {
            const adapted = generateSchemaAdaptedWidgets(dataSummary, dataPreview, activeDatabaseConfig.db_type);
            setWidgets(adapted.widgets);
            setLayouts(adapted.layouts);
            setDashboardName(`Live DB: ${activeDatabaseConfig.db_type?.toUpperCase()} Dashboard`);
            initializedRef.current = true;
            return;
        }

        if (dataSummary && dataSummary.columns && dataSummary.columns.length > 0 && !initializedRef.current && widgets.length === 0) {
            const adapted = generateSchemaAdaptedWidgets(dataSummary, dataPreview, 'Default');
            setWidgets(adapted.widgets);
            setLayouts(adapted.layouts);
            setDashboardName('Auto-Generated Overview');
            initializedRef.current = true;
        }
    }, [mounted, agentDashboard, activeDatabaseConfig, dataSummary]);

    // Check if current widgets have columns that are missing from dataSummary
    const hasSchemaMismatch = React.useMemo(() => {
        if (!dataSummary?.columns || widgets.length === 0) return false;
        const availableCols = new Set(dataSummary.columns);
        const invalidWidgets = widgets.filter(w => {
            if (w.type === 'kpi_card' && w.customValue !== undefined) return false;
            if (w.xAxis && !availableCols.has(w.xAxis)) return true;
            if (w.yAxis && !availableCols.has(w.yAxis)) return true;
            return false;
        });
        return invalidWidgets.length > 0;
    }, [dataSummary, widgets]);

    const handleAutoAdaptToDatabase = () => {
        if (!dataSummary) return;
        const title = activeDatabaseConfig ? `Live DB (${activeDatabaseConfig.db_type?.toUpperCase()})` : 'Adapted';
        const adapted = generateSchemaAdaptedWidgets(dataSummary, dataPreview, title);
        setWidgets(adapted.widgets);
        setLayouts(adapted.layouts);
        setDashboardName(`${title} Custom Dashboard`);
        addToast?.('Dashboard auto-adapted to current schema successfully!', 'success');
    };

    const handleRefreshDb = async () => {
        if (!activeDatabaseConfig) return;
        setDbRefreshing(true);
        try {
            await refreshFromDatabase();
            addToast?.('Live database records refreshed successfully!', 'success');
        } catch (err) {
            console.error("Live DB refresh failed:", err);
            addToast?.('Database refresh failed: ' + (err.message || 'Unknown error'), 'error');
        } finally {
            setDbRefreshing(false);
        }
    };

    async function loadDashboards() {
        try {
            const list = await listDashboards();
            setDashboards(list);
        } catch (err) {
            console.error("Failed to load dashboards", err);
        }
    }

    const handleSave = async () => {
        if (!widgets.length) return;
        try {
            const payload = { widgets, layouts };
            await saveDashboard(dashboardName, payload);
            addToast?.('Dashboard saved successfully!', 'success');
            loadDashboards();
        } catch (err) {
            console.error(err);
            const msg = err.response?.data?.error || err.response?.data?.detail || err.message;
            addToast?.('Failed to save: ' + msg, 'error');
        }
    };

    const handleLoad = async (id) => {
        try {
            const dash = await getDashboard(id);
            if (dash) {
                setDashboardName(dash.name);

                if (Array.isArray(dash.layout)) {
                    const legacyMixin = dash.layout;
                    setWidgets(legacyMixin);
                    setLayouts({ lg: legacyMixin, md: legacyMixin, sm: legacyMixin });
                } else if (dash.layout && dash.layout.widgets) {
                    setWidgets(dash.layout.widgets);
                    setLayouts(dash.layout.layouts || { lg: [], md: [], sm: [] });
                }
            }
        } catch (err) {
            console.error(err);
        }
    };

    const addWidget = () => {
        const id = String(Date.now());
        const newItemConfig = { ...newWidget, id };

        setWidgets([...widgets, newItemConfig]);

        const newLayoutItem = {
            i: id,
            x: 0,
            y: Infinity,
            w: 2,
            h: 3
        };

        setLayouts(prev => ({
            ...prev,
            lg: [...(prev.lg || []), newLayoutItem],
            md: [...(prev.md || []), newLayoutItem],
            sm: [...(prev.sm || []), { ...newLayoutItem, w: 1 }]
        }));

        setShowAddModal(false);
    };

    const removeWidget = (id) => {
        setWidgets(widgets.filter(w => String(w.id) !== String(id)));
        setLayouts(prev => {
            const next = { ...prev };
            Object.keys(next).forEach(bp => {
                next[bp] = (next[bp] || []).filter(l => l.i !== String(id));
            });
            return next;
        });
    };

    // Helper to process data for charts
    const processData = (config) => {
        if (!dataPreview || !Array.isArray(dataPreview) || dataPreview.length === 0) return [];

        const xCol = config.xAxis || (dataSummary?.columns ? dataSummary.columns[0] : null);
        const yCol = config.yAxis || (dataSummary?.columns && dataSummary.columns.length > 1 ? dataSummary.columns[1] : xCol);
        if (!xCol) return [];

        if (config.type === 'scatter') {
            const isXNumeric = dataPreview.some(r => typeof r[xCol] === 'number' || (!isNaN(parseFloat(r[xCol])) && isFinite(r[xCol])));
            const isYNumeric = yCol && dataPreview.some(r => typeof r[yCol] === 'number' || (!isNaN(parseFloat(r[yCol])) && isFinite(r[yCol])));

            if (isXNumeric && isYNumeric) {
                const points = dataPreview
                    .map((row, idx) => {
                        const x = parseFloat(row[xCol]);
                        const y = parseFloat(row[yCol]);
                        return {
                            x: isNaN(x) ? 0 : x,
                            y: isNaN(y) ? 0 : y,
                            name: String(row[xCol] || `Item ${idx + 1}`)
                        };
                    })
                    .filter(d => !isNaN(d.x) && !isNaN(d.y));
                if (points.length > 0) return points.slice(0, 50);
            }
            // Fallback if categorical: process as aggregated category breakdown
        }

        const counts = {};

        dataPreview.forEach(row => {
            const rawVal = row[xCol];
            const key = rawVal !== undefined && rawVal !== null && String(rawVal).trim() !== '' 
                ? String(rawVal) 
                : 'Unknown';

            if (!counts[key]) counts[key] = { name: key, value: 0, count: 0, sum: 0 };

            const rawY = yCol ? row[yCol] : null;
            const parsedY = rawY !== null && rawY !== undefined ? parseFloat(rawY) : NaN;
            const validY = isNaN(parsedY) ? 0 : parsedY;

            if (config.aggregation === 'count' || !yCol) {
                counts[key].value += 1;
            } else if (config.aggregation === 'sum') {
                counts[key].value += validY;
            } else if (config.aggregation === 'avg' || config.aggregation === 'mean') {
                counts[key].sum += validY;
                counts[key].count += 1;
                counts[key].value = Number((counts[key].sum / Math.max(1, counts[key].count)).toFixed(2));
            } else {
                counts[key].value += 1;
            }
        });

        const result = Object.values(counts);

        if (config.type !== 'line') {
            result.sort((a, b) => b.value - a.value);
        }

        return result.slice(0, 30);
    };

    const renderWidgetChart = (widget) => {
        const data = processData(widget);
        if (!data || data.length === 0) {
            return (
                <div className="h-full flex flex-col items-center justify-center text-slate-400 text-xs p-4">
                    <BarChart2 className="mb-2 text-slate-300" size={26} />
                    <span className="font-semibold text-slate-600">No chart records for {widget.xAxis || 'selection'}</span>
                    <span className="text-[11px] text-slate-400 mt-0.5">Records will appear as data streams</span>
                </div>
            );
        }

        switch (widget.type) {
            case 'bar':
                return (
                    <BarChart data={data} layout={widget.horizontal ? 'vertical' : 'horizontal'} margin={{ top: 10, right: 20, left: 10, bottom: 35 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis
                            type={widget.horizontal ? 'number' : 'category'}
                            dataKey={widget.horizontal ? undefined : 'name'}
                            tick={{ fontSize: 10, angle: -30, textAnchor: 'end' }}
                            height={45}
                            interval="preserveStartEnd"
                        />
                        <YAxis
                            type={widget.horizontal ? 'category' : 'number'}
                            dataKey={widget.horizontal ? 'name' : undefined}
                            width={55}
                            tick={{ fontSize: 10 }}
                        />
                        <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                        <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                );

            case 'line':
                return (
                    <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 35 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                            dataKey="name"
                            tick={{ fontSize: 10, angle: -30, textAnchor: 'end' }}
                            height={45}
                            interval="preserveStartEnd"
                        />
                        <YAxis tick={{ fontSize: 10 }} width={45} />
                        <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }} />
                        <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2.5} dot={{ r: 3 }} />
                    </LineChart>
                );

            case 'area':
                return (
                    <AreaChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 35 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                            dataKey="name"
                            tick={{ fontSize: 10, angle: -30, textAnchor: 'end' }}
                            height={45}
                            interval="preserveStartEnd"
                        />
                        <YAxis tick={{ fontSize: 10 }} width={45} />
                        <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }} />
                        <Area type="monotone" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.4} />
                    </AreaChart>
                );

            case 'scatter':
                return (
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                        <CartesianGrid stroke="#f1f5f9" />
                        <XAxis type="number" dataKey="x" name={widget.xAxis} tick={{ fontSize: 10 }} />
                        <YAxis type="number" dataKey="y" name={widget.yAxis} tick={{ fontSize: 10 }} width={45} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name={widget.title} data={data} fill="#ec4899" />
                    </ScatterChart>
                );

            case 'pie':
                return (
                    <RePie margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="50%"
                            innerRadius="35%"
                            outerRadius="65%"
                            paddingAngle={4}
                            dataKey="value"
                            nameKey="name"
                            label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                            labelLine={false}
                        >
                            {data.map((entry, index) => (
                                <Cell key={'cell-' + index} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }} />
                        <Legend verticalAlign="bottom" height={24} iconSize={8} wrapperStyle={{ fontSize: '10px' }} />
                    </RePie>
                );

            case 'radar':
                return (
                    <RadarChart outerRadius="60%" data={data} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                        <PolarGrid stroke="#f1f5f9" />
                        <PolarAngleAxis dataKey="name" tick={{ fontSize: 9 }} />
                        <PolarRadiusAxis angle={30} tick={{ fontSize: 9 }} />
                        <Radar name={widget.title} dataKey="value" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.5} />
                    </RadarChart>
                );

            default:
                return (
                    <BarChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 35 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="name" tick={{ fontSize: 10, angle: -30, textAnchor: 'end' }} height={45} />
                        <YAxis tick={{ fontSize: 10 }} width={45} />
                        <Tooltip />
                        <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                );
        }
    };

    if (!dataSummary) return <div className="p-8">Please load a dataset first.</div>;
    if (!mounted) return <div className="p-8">Loading dashboard...</div>;

    const columns = dataSummary.columns;

    return (
        <div id="dashboard-content" className="p-4 md:p-8 max-w-7xl mx-auto space-y-6 md:space-y-8">
            {/* Top Control Bar */}
            <div className="flex flex-col md:flex-row justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-slate-200 gap-4">
                <div className="flex items-center gap-4 w-full md:w-auto">
                    <Layout className="text-indigo-600" />
                    <input
                        value={dashboardName}
                        onChange={(e) => setDashboardName(e.target.value)}
                        className="text-xl font-bold text-slate-800 border-none focus:ring-0 w-full md:w-auto"
                    />
                </div>
                <div className="flex flex-wrap gap-2 w-full md:w-auto justify-end">
                    <select
                        onChange={(e) => handleLoad(e.target.value)}
                        className="px-4 py-2 border rounded-lg text-sm bg-white"
                    >
                        <option value="">Load Dashboard...</option>
                        {dashboards.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                    </select>
                    <button onClick={() => setShowAddModal(true)} className="flex items-center gap-2 px-4 py-2 bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100 transition-colors text-sm">
                        <Plus size={18} /> Add
                    </button>
                    <ReportGenerator targetId="dashboard-content" fileName={dashboardName + '.pdf'} />
                    <button onClick={handleSave} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 shadow-md transition-all text-sm">
                        <Save size={18} /> Save
                    </button>
                </div>
            </div>

            {/* Banner 1: Autonomous Agent Synthesized Dashboard */}
            {agentDashboard && (
                <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white p-4 rounded-2xl shadow-sm border border-indigo-500/30 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 bg-indigo-500/20 border border-indigo-400/30 rounded-xl text-indigo-400">
                            <Sparkles size={20} className="animate-pulse" />
                        </div>
                        <div>
                            <div className="flex items-center gap-2">
                                <span className="font-bold text-sm tracking-wide text-indigo-200 uppercase">Autonomous Agent Synthesized Dashboard</span>
                                <span className="bg-indigo-500/30 text-indigo-300 text-[10px] px-2 py-0.5 rounded-full border border-indigo-400/20 font-bold">
                                    ACTIVE
                                </span>
                            </div>
                            <p className="text-xs text-slate-300 mt-0.5">
                                Tailored analytics for target: <strong className="text-white">{agentDashboard.target_column}</strong> ({agentDashboard.problem_type}) with 8 synthesized analytical widgets.
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                        <button
                            onClick={() => {
                                setWidgets(agentDashboard.widgets);
                                setLayouts(agentDashboard.layouts || { lg: [], md: [], sm: [] });
                                setDashboardName(agentDashboard.name || 'AI Dashboard');
                                addToast?.('Reset to AI Agent synthesized blueprint', 'info');
                            }}
                            className="px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white rounded-lg text-xs font-semibold border border-white/20 transition flex items-center gap-1.5 cursor-pointer"
                        >
                            <RotateCcw size={13} />
                            Reset to AI Blueprint
                        </button>
                    </div>
                </div>
            )}

            {/* Banner 2: Live Database Connection */}
            {activeDatabaseConfig && (
                <div className="bg-slate-900 text-white p-4 rounded-2xl shadow-sm border border-emerald-500/40 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 bg-emerald-500/20 border border-emerald-400/30 rounded-xl text-emerald-400">
                            <Database size={20} className="animate-pulse" />
                        </div>
                        <div>
                            <div className="flex items-center gap-2">
                                <span className="font-bold text-sm tracking-wide text-emerald-400 uppercase">Live Database Connected</span>
                                <span className="bg-emerald-500/20 text-emerald-300 text-[10px] px-2 py-0.5 rounded-full border border-emerald-500/30 font-mono font-bold">
                                    {activeDatabaseConfig.db_type?.toUpperCase()}
                                </span>
                            </div>
                            <p className="text-xs text-slate-300 mt-0.5">
                                {activeDatabaseConfig.query ? (
                                    <span>Query: <code className="bg-slate-800 px-1.5 py-0.5 rounded text-emerald-300 font-mono text-[11px]">{activeDatabaseConfig.query}</code></span>
                                ) : activeDatabaseConfig.collection ? (
                                    <span>Collection: <code className="bg-slate-800 px-1.5 py-0.5 rounded text-emerald-300 font-mono text-[11px]">{activeDatabaseConfig.collection}</code></span>
                                ) : (
                                    <span>Connected to live database instance. Changes in records reflect automatically upon sync.</span>
                                )}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0 flex-wrap">
                        <button
                            onClick={handleRefreshDb}
                            disabled={dbRefreshing}
                            className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold shadow-sm transition flex items-center gap-1.5 disabled:opacity-50 cursor-pointer"
                        >
                            <RefreshCw size={13} className={dbRefreshing ? 'animate-spin' : ''} />
                            <span>{dbRefreshing ? 'Syncing...' : 'Sync Live DB'}</span>
                        </button>
                        <button
                            onClick={handleAutoAdaptToDatabase}
                            className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-emerald-500/40 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 cursor-pointer"
                        >
                            <Sparkles size={13} className="text-emerald-400" />
                            <span>Auto-Adapt to DB Schema</span>
                        </button>
                    </div>
                </div>
            )}

            {/* Banner 3: Schema Mismatch Alert */}
            {hasSchemaMismatch && (
                <div className="bg-amber-50 border border-amber-200 text-amber-900 p-4 rounded-xl flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <AlertTriangle className="text-amber-500 shrink-0" size={20} />
                        <div className="text-xs">
                            <span className="font-bold">Schema Change Detected:</span> Some widgets refer to columns that don't exist in the current dataset or database table.
                        </div>
                    </div>
                    <button
                        onClick={handleAutoAdaptToDatabase}
                        className="px-3 py-1.5 bg-amber-600 hover:bg-amber-700 text-white rounded-lg text-xs font-bold transition flex items-center gap-1.5 shrink-0 cursor-pointer"
                    >
                        <Sparkles size={13} />
                        Auto-Adapt Dashboard to Schema
                    </button>
                </div>
            )}

            {/* Draggable Grid Layout */}
            <ResponsiveGridLayout
                className="layout"
                layouts={layouts}
                breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                cols={{ lg: 12, md: 6, sm: 2, xs: 1, xxs: 1 }}
                rowHeight={100}
                onLayoutChange={(currentLayout, allLayouts) => {
                    setLayouts(allLayouts);
                }}
                margin={[16, 16]}
                draggableHandle=".draggable-handle"
            >
                {widgets.map((widget) => {
                    return (
                        <div key={String(widget.id)} className="bg-white p-3 rounded-2xl shadow-sm border border-slate-200 flex flex-col group hover:shadow-md transition-all relative h-full w-full overflow-hidden">
                            <div className="draggable-handle flex justify-between items-center mb-2 px-1 cursor-grab active:cursor-grabbing shrink-0">
                                <h3 className="font-semibold text-slate-700 text-xs truncate pr-2 uppercase tracking-wide">{widget.title}</h3>
                                <button
                                    onMouseDown={(e) => e.stopPropagation()}
                                    onClick={() => removeWidget(widget.id)}
                                    className="text-slate-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded cursor-pointer"
                                    title="Remove Widget"
                                >
                                    <Trash2 size={13} />
                                </button>
                            </div>
                            {widget.type === 'kpi_card' ? (
                                <div className="flex-1 w-full h-full overflow-hidden">
                                    <KPIWidget widget={widget} />
                                </div>
                            ) : (
                                <div className="flex-1 w-full h-full min-h-[170px] relative overflow-hidden" style={{ minHeight: 170 }}>
                                    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={170}>
                                        {renderWidgetChart(widget)}
                                    </ResponsiveContainer>
                                </div>
                            )}
                        </div>
                    );
                })}
            </ResponsiveGridLayout>

            {widgets.length === 0 && (
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

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                                        <option value="kpi_card">KPI Score Card</option>
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
                            <div className={`flex flex-wrap gap-4 ${newWidget.type === 'kpi_card' ? 'hidden' : ''}`}>
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

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">
                                        {newWidget.type === 'kpi_card' ? 'Value Column' : `X-Axis (${newWidget.type === 'scatter' ? 'Numeric' : 'Category'})`}
                                    </label>
                                    <select
                                        className="w-full p-2 border rounded-lg"
                                        value={newWidget.xAxis}
                                        onChange={e => setNewWidget({ ...newWidget, xAxis: e.target.value })}
                                    >
                                        <option value="">Select Column...</option>
                                        {columns.map(c => <option key={c} value={c}>{c}</option>)}
                                    </select>
                                </div>

                                {newWidget.aggregation === 'sum' && newWidget.type !== 'kpi_card' && (
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

                            {newWidget.type === 'kpi_card' && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Icon</label>
                                        <select
                                            className="w-full p-2 border rounded-lg"
                                            value={newWidget.icon}
                                            onChange={e => setNewWidget({ ...newWidget, icon: e.target.value })}
                                        >
                                            <option value="hash"># Number</option>
                                            <option value="dollar">$ Money</option>
                                            <option value="users">Users</option>
                                            <option value="trend">Trend</option>
                                            <option value="activity">Activity</option>
                                            <option value="shop">Shopping</option>
                                            <option value="card">Card</option>
                                            <option value="percent">% Percent</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Color Theme</label>
                                        <select
                                            className="w-full p-2 border rounded-lg"
                                            value={newWidget.color}
                                            onChange={e => setNewWidget({ ...newWidget, color: e.target.value })}
                                        >
                                            <option value="indigo">Indigo User</option>
                                            <option value="emerald">Emerald Success</option>
                                            <option value="rose">Rose Danger</option>
                                            <option value="amber">Amber Warning</option>
                                            <option value="cyan">Cyan Info</option>
                                        </select>
                                    </div>
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
