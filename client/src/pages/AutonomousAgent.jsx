import React, { useState, useEffect, useRef } from 'react';
import { useData } from '../context/DataContext';
import { useToast } from '../context/ToastContext';
import {
    runAutonomousAgent,
    loadAgentSample,
    getAgentPlan,
    predictModel,
    downloadModel,
    exportData
} from '../services/api';
import {
    Bot, Play, RotateCcw, Award, CheckCircle2, AlertCircle,
    ArrowRight, Terminal, BarChart3, Database, Sparkles, Cpu,
    Download, ShieldCheck, Zap, RefreshCw, Layers, TrendingUp,
    Check, Copy, ChevronRight, Sliders, ExternalLink
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell
} from 'recharts';

const GOAL_PRESETS = [
    { id: 'automl', label: '🏆 Full Auto-Pilot: Profile, Clean, Engineer & Crown Best Model', desc: 'End-to-end autonomous AutoML optimization' },
    { id: 'quality', label: '🧹 Self-Healing Quality: Anomaly Repair & Deduplication', desc: 'Prioritize data hygiene and imputations' },
    { id: 'classification', label: '🎯 High-Precision Classifier with Target Optimization', desc: 'Train classification models for categorical targets' },
    { id: 'regression', label: '📈 Predictive Value Regression & Driver Discovery', desc: 'Continuous metric modeling with driver attribution' }
];

const SAMPLE_DATASETS = [
    { id: 'churn', name: 'Telco Customer Churn', type: 'Classification', target: 'Churn', badge: 'Popular', icon: '📱' },
    { id: 'housing', name: 'Real Estate House Prices', type: 'Regression', target: 'SalePrice', badge: 'High Variance', icon: '🏡' },
    { id: 'retention', name: 'Employee Retention', type: 'Classification', target: 'Attrition', badge: 'HR Analytics', icon: '👥' }
];

export default function AutonomousAgent() {
    const { dataSummary, setDataSummary, setDataPreview } = useData();
    const { addToast } = useToast();

    // Configuration State
    const [selectedGoal, setSelectedGoal] = useState(GOAL_PRESETS[0].label);
    const [targetColumn, setTargetColumn] = useState('auto');
    const [speedMode, setSpeedMode] = useState('live'); // 'live' or 'turbo'
    const [activeTab, setActiveTab] = useState('tournament'); // 'tournament' | 'diff' | 'drivers' | 'brief' | 'predict'

    // Execution State
    const [agentStatus, setAgentStatus] = useState('idle'); // 'idle' | 'running' | 'completed' | 'error'
    const [currentStageIndex, setCurrentStageIndex] = useState(-1);
    const [runData, setRunData] = useState(null);
    const [visibleLogs, setVisibleLogs] = useState([]);
    const [terminalAutoScroll, setTerminalAutoScroll] = useState(true);
    const [copiedBrief, setCopiedBrief] = useState(false);

    // Live Prediction State
    const [predictInputs, setPredictInputs] = useState({});
    const [predictionResult, setPredictionResult] = useState(null);
    const [isPredicting, setIsPredicting] = useState(false);

    const terminalEndRef = useRef(null);
    const streamingTimerRef = useRef(null);

    // Scroll terminal to bottom
    useEffect(() => {
        if (terminalAutoScroll && terminalEndRef.current) {
            terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [visibleLogs, terminalAutoScroll]);

    // Cleanup timer on unmount
    useEffect(() => {
        return () => {
            if (streamingTimerRef.current) clearInterval(streamingTimerRef.current);
        };
    }, []);

    // Setup initial target column if data exists
    useEffect(() => {
        if (dataSummary && dataSummary.columns && dataSummary.columns.length > 0) {
            if (targetColumn === 'auto' || !dataSummary.columns.includes(targetColumn)) {
                // Find default candidate
                const common = ['Churn', 'target', 'Target', 'SalePrice', 'price', 'Price', 'Attrition', 'left'];
                const found = dataSummary.columns.find(c => common.includes(c));
                setTargetColumn(found || dataSummary.columns[dataSummary.columns.length - 1]);
            }
        }
    }, [dataSummary]);

    // Handle Streaming of Logs & Stages
    const streamExecution = (result) => {
        const allLogs = result.logs || [];
        const allStages = result.stages || [];

        if (speedMode === 'turbo') {
            setVisibleLogs(allLogs);
            setCurrentStageIndex(allStages.length - 1);
            setAgentStatus('completed');
            setRunData(result);
            if (result.sample_inputs) setPredictInputs(result.sample_inputs);
            return;
        }

        // Live Simulated Human-Speed Stream
        let logIndex = 0;
        let stageIdx = 0;
        setVisibleLogs([]);
        setCurrentStageIndex(0);

        const interval = setInterval(() => {
            if (logIndex < allLogs.length) {
                const currentLog = allLogs[logIndex];
                setVisibleLogs(prev => [...prev, currentLog]);

                // Advance stage based on log stage
                if (currentLog.stage && currentLog.stage - 1 > stageIdx) {
                    stageIdx = currentLog.stage - 1;
                    setCurrentStageIndex(stageIdx);
                }
                logIndex++;
            } else {
                clearInterval(interval);
                setCurrentStageIndex(allStages.length - 1);
                setAgentStatus('completed');
                setRunData(result);
                if (result.sample_inputs) setPredictInputs(result.sample_inputs);
                addToast('Autonomous Agent execution completed!', 'success');
            }
        }, 120);

        streamingTimerRef.current = interval;
    };

    // Execute Autonomous Agent
    const handleLaunchAgent = async () => {
        setAgentStatus('running');
        setVisibleLogs([]);
        setCurrentStageIndex(0);
        setRunData(null);
        setPredictionResult(null);

        try {
            const payload = {
                goal: selectedGoal,
                target_column: targetColumn === 'auto' ? undefined : targetColumn,
                problem_type: 'auto',
                feature_engineering: true,
                outlier_handling: true
            };

            const result = await runAutonomousAgent(payload);

            // Update app data context with new dataset
            if (result.new_summary) {
                setDataSummary(result.new_summary);
                if (result.new_summary.preview) {
                    setDataPreview(result.new_summary.preview);
                }
            }

            streamExecution(result);
        } catch (err) {
            console.error('Agent execution error:', err);
            setAgentStatus('error');
            const msg = err.response?.data?.error || err.message || 'Agent failed to complete task';
            addToast(`Agent Error: ${msg}`, 'error');
            setVisibleLogs(prev => [...prev, {
                timestamp: new Date().toLocaleTimeString(),
                stage: 1,
                type: 'warning',
                message: `[CRITICAL FATAL]: ${msg}`
            }]);
        }
    };

    // Load Sample Dataset and Auto-Run
    const handleLoadSample = async (datasetId) => {
        setAgentStatus('running');
        setVisibleLogs([]);
        setCurrentStageIndex(0);
        setRunData(null);
        setPredictionResult(null);

        try {
            const result = await loadAgentSample(datasetId, true, selectedGoal);

            // Update app context
            if (result.new_summary) {
                setDataSummary(result.new_summary);
                if (result.new_summary.preview) {
                    setDataPreview(result.new_summary.preview);
                }
            }
            if (result.target_column) {
                setTargetColumn(result.target_column);
            }

            streamExecution(result);
        } catch (err) {
            console.error('Sample run error:', err);
            setAgentStatus('error');
            addToast(`Failed to load sample: ${err.message}`, 'error');
        }
    };

    // Handle Live Instant Prediction
    const handlePredict = async (e) => {
        if (e) e.preventDefault();
        setIsPredicting(true);
        try {
            const res = await predictModel(predictInputs);
            setPredictionResult(res);
            addToast('Prediction computed successfully!', 'success');
        } catch (err) {
            console.error('Prediction error:', err);
            addToast(`Prediction failed: ${err.response?.data?.error || err.message}`, 'error');
        } finally {
            setIsPredicting(false);
        }
    };

    // Handle Model Download
    const handleDownloadModel = async () => {
        try {
            const blob = await downloadModel();
            const url = window.URL.createObjectURL(new Blob([blob]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'nexus_champion_model.pkl');
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
            addToast('Champion model downloaded (.pkl)', 'success');
        } catch (err) {
            addToast('Download failed: No trained model available.', 'error');
        }
    };

    // Copy Executive Report
    const handleCopyReport = () => {
        if (!runData?.executive_summary) return;
        const text = `# NexusDash Autonomous Agent Report\n\n${runData.executive_summary}\n\n## Recommendations:\n${runData.recommendations.map((r, i) => `${i + 1}. ${r}`).join('\n')}`;
        navigator.clipboard.writeText(text);
        setCopiedBrief(true);
        addToast('Executive brief copied to clipboard!', 'success');
        setTimeout(() => setCopiedBrief(false), 2500);
    };

    // Helper for log badges
    const getLogBadge = (type) => {
        switch (type) {
            case 'thought':
                return <span className="bg-purple-950/80 text-purple-400 border border-purple-800/60 px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-semibold">REASONING</span>;
            case 'tool':
                return <span className="bg-cyan-950/80 text-cyan-400 border border-cyan-800/60 px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-semibold">TOOL CALL</span>;
            case 'metric':
                return <span className="bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-semibold">BENCHMARK</span>;
            case 'success':
                return <span className="bg-green-950/80 text-green-300 border border-green-700/60 px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-semibold">SUCCESS</span>;
            case 'warning':
                return <span className="bg-rose-950/80 text-rose-400 border border-rose-800/60 px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-semibold">ALERT</span>;
            default:
                return <span className="bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded text-[10px] font-mono uppercase">INFO</span>;
        }
    };

    const defaultStages = [
        { id: 1, title: 'Deep Dataset Profiling', desc: 'Scan types, distributions & target definition' },
        { id: 2, title: 'Self-Healing Cleaning', desc: 'Impute missing values, drop duplicates & outliers' },
        { id: 3, title: 'Autonomous Feature Engineering', desc: 'Synthesize interaction terms & log transforms' },
        { id: 4, title: 'AutoML Model Tournament', desc: 'Benchmark algorithms & crown champion' },
        { id: 5, title: 'Feature Attribution & Drivers', desc: 'Explainability & top predictive drivers' },
        { id: 6, title: 'Executive Synthesis', desc: 'Briefing, diff analysis & instant deployment' }
    ];

    const stagesList = runData?.stages || defaultStages;

    return (
        <div className="p-4 md:p-6 lg:p-8 space-y-6 max-w-7xl mx-auto">
            {/* Header & Agent Identity */}
            <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 border border-indigo-900/40 p-6 md:p-8 shadow-2xl">
                <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />
                <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

                <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="space-y-2">
                        <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-500/30 text-white">
                                <Bot size={28} className={agentStatus === 'running' ? 'animate-pulse' : ''} />
                            </div>
                            <div>
                                <div className="flex items-center gap-2">
                                    <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
                                        Nexus Autonomous Agent
                                    </h1>
                                    <span className="bg-gradient-to-r from-indigo-500 to-cyan-400 text-white text-[11px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider shadow-sm">
                                        AutoML Core v2.5
                                    </span>
                                </div>
                                <p className="text-slate-400 text-sm">
                                    1-Click Autonomous Data Ingestion, Self-Healing Cleaning, Multi-Model Arena & Executive Insights
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Agent Status Badge */}
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 bg-slate-800/80 border border-slate-700/80 backdrop-blur-md px-3.5 py-1.5 rounded-full text-xs font-medium">
                            <span className={`w-2.5 h-2.5 rounded-full ${
                                agentStatus === 'running' ? 'bg-amber-400 animate-ping' :
                                agentStatus === 'completed' ? 'bg-emerald-400' :
                                agentStatus === 'error' ? 'bg-rose-400' : 'bg-cyan-400'
                            }`} />
                            <span className="text-slate-200 uppercase tracking-wider font-mono text-[11px]">
                                {agentStatus === 'running' ? 'EXECUTING PIPELINE' :
                                 agentStatus === 'completed' ? 'AGENT RUN COMPLETE' :
                                 agentStatus === 'error' ? 'EXECUTION ERROR' : 'READY TO LAUNCH'}
                            </span>
                        </div>

                        {runData?.total_duration_ms && (
                            <div className="bg-indigo-950/60 border border-indigo-800/50 text-indigo-300 px-3 py-1.5 rounded-full text-xs font-mono">
                                ⚡ {runData.total_duration_ms}ms
                            </div>
                        )}
                    </div>
                </div>

                {/* Control Bar: Goal, Target, Speed & Quick Samples */}
                <div className="mt-8 pt-6 border-t border-slate-800/80 grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
                    {/* Goal Preset */}
                    <div className="md:col-span-5 space-y-1.5">
                        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                            <Sparkles size={14} className="text-cyan-400" />
                            Autonomous Agent Objective
                        </label>
                        <select
                            value={selectedGoal}
                            onChange={(e) => setSelectedGoal(e.target.value)}
                            disabled={agentStatus === 'running'}
                            className="w-full bg-slate-800/90 border border-slate-700 text-white text-sm rounded-xl px-3.5 py-2.5 focus:outline-none focus:border-cyan-400 transition"
                        >
                            {GOAL_PRESETS.map(g => (
                                <option key={g.id} value={g.label}>{g.label}</option>
                            ))}
                        </select>
                    </div>

                    {/* Target Column */}
                    <div className="md:col-span-3 space-y-1.5">
                        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                            <Database size={14} className="text-indigo-400" />
                            Target Column
                        </label>
                        <select
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            disabled={agentStatus === 'running'}
                            className="w-full bg-slate-800/90 border border-slate-700 text-white text-sm rounded-xl px-3.5 py-2.5 focus:outline-none focus:border-indigo-400 transition"
                        >
                            <option value="auto">⚡ Auto-Detect Best Target</option>
                            {dataSummary?.columns?.map(col => (
                                <option key={col} value={col}>{col}</option>
                            ))}
                        </select>
                    </div>

                    {/* Execution Speed */}
                    <div className="md:col-span-2 space-y-1.5">
                        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                            <Zap size={14} className="text-amber-400" />
                            Execution Speed
                        </label>
                        <div className="grid grid-cols-2 gap-1 bg-slate-800/90 p-1 rounded-xl border border-slate-700">
                            <button
                                type="button"
                                onClick={() => setSpeedMode('live')}
                                className={`text-xs font-medium py-1.5 rounded-lg transition ${
                                    speedMode === 'live' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
                                }`}
                            >
                                Live Stream
                            </button>
                            <button
                                type="button"
                                onClick={() => setSpeedMode('turbo')}
                                className={`text-xs font-medium py-1.5 rounded-lg transition ${
                                    speedMode === 'turbo' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
                                }`}
                            >
                                Turbo
                            </button>
                        </div>
                    </div>

                    {/* Launch Action */}
                    <div className="md:col-span-2">
                        <button
                            onClick={handleLaunchAgent}
                            disabled={agentStatus === 'running' || !dataSummary}
                            className={`w-full py-2.5 px-4 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-lg transition-all duration-200 ${
                                agentStatus === 'running'
                                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                                    : !dataSummary
                                    ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                                    : 'bg-gradient-to-r from-indigo-500 via-purple-600 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:-translate-y-0.5'
                            }`}
                        >
                            {agentStatus === 'running' ? (
                                <>
                                    <RefreshCw size={16} className="animate-spin" />
                                    <span>Executing...</span>
                                </>
                            ) : (
                                <>
                                    <Play size={16} fill="currentColor" />
                                    <span>Launch Agent</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Quick 1-Click Sample Datasets Bar */}
                <div className="mt-4 pt-4 border-t border-slate-800/40 flex flex-wrap items-center gap-3">
                    <span className="text-xs text-slate-400 font-medium">Or 1-Click Launch with Sample Data:</span>
                    {SAMPLE_DATASETS.map(sample => (
                        <button
                            key={sample.id}
                            onClick={() => handleLoadSample(sample.id)}
                            disabled={agentStatus === 'running'}
                            className="bg-slate-800/70 hover:bg-slate-700/80 border border-slate-700 text-slate-200 hover:text-white px-3 py-1.5 rounded-lg text-xs flex items-center gap-2 transition group"
                        >
                            <span>{sample.icon}</span>
                            <span className="font-medium">{sample.name}</span>
                            <span className="text-[10px] bg-indigo-950 text-indigo-300 border border-indigo-800/60 px-1.5 py-0.2 rounded font-mono">
                                {sample.type}
                            </span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Main Stage & Studio Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                {/* Column 1: Vertical Stage Progress Flow (4 Cols) */}
                <div className="lg:col-span-4 space-y-4">
                    <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                                <Layers size={16} className="text-primary" />
                                Execution Stages
                            </h2>
                            <span className="text-xs text-slate-500 font-mono">
                                {currentStageIndex >= 0 ? `${Math.min(currentStageIndex + 1, 6)}/6 Stages` : 'Idle'}
                            </span>
                        </div>

                        <div className="space-y-3">
                            {stagesList.map((stage, idx) => {
                                const isCurrent = agentStatus === 'running' && idx === currentStageIndex;
                                const isDone = idx < currentStageIndex || (agentStatus === 'completed');
                                const isPending = idx > currentStageIndex && agentStatus !== 'completed';

                                return (
                                    <div
                                        key={stage.id}
                                        className={`relative p-3.5 rounded-xl border transition-all duration-300 ${
                                            isCurrent
                                                ? 'bg-indigo-50/80 border-indigo-300 shadow-sm ring-1 ring-indigo-200'
                                                : isDone
                                                ? 'bg-slate-50/60 border-slate-200 text-slate-800'
                                                : 'bg-white border-slate-100 text-slate-400'
                                        }`}
                                    >
                                        <div className="flex items-start gap-3">
                                            <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold mt-0.5 shrink-0 ${
                                                isCurrent
                                                    ? 'bg-indigo-600 text-white shadow animate-pulse'
                                                    : isDone
                                                    ? 'bg-emerald-600 text-white'
                                                    : 'bg-slate-200 text-slate-500'
                                            }`}>
                                                {isDone ? <Check size={14} strokeWidth={3} /> : idx + 1}
                                            </div>

                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center justify-between">
                                                    <h3 className={`text-xs font-bold truncate ${
                                                        isCurrent ? 'text-indigo-900' : isDone ? 'text-slate-800' : 'text-slate-400'
                                                    }`}>
                                                        {stage.title}
                                                    </h3>
                                                    {stage.duration_ms && isDone && (
                                                        <span className="text-[10px] font-mono text-slate-400 shrink-0">
                                                            {stage.duration_ms}ms
                                                        </span>
                                                    )}
                                                </div>
                                                <p className="text-[11px] text-slate-500 mt-0.5 line-clamp-2 leading-relaxed">
                                                    {stage.summary || stage.description || stage.desc}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Quick Model Download / Export Card */}
                    {runData?.champion_model && (
                        <div className="bg-gradient-to-br from-indigo-900 to-slate-900 text-white rounded-2xl p-5 shadow-lg border border-indigo-800/40 space-y-3">
                            <div className="flex items-center gap-2">
                                <Award size={20} className="text-amber-400" />
                                <h3 className="font-bold text-sm">Champion Artifact Ready</h3>
                            </div>
                            <p className="text-xs text-indigo-200">
                                {runData.champion_model.name} scored {runData.champion_model.metric_value} on {runData.champion_model.metric_name}.
                            </p>
                            <div className="grid grid-cols-2 gap-2 pt-1">
                                <button
                                    onClick={handleDownloadModel}
                                    className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition"
                                >
                                    <Download size={14} />
                                    <span>Model (.pkl)</span>
                                </button>
                                <button
                                    onClick={handleCopyReport}
                                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition border border-slate-700"
                                >
                                    {copiedBrief ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
                                    <span>{copiedBrief ? 'Copied!' : 'Copy Brief'}</span>
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Column 2: Terminal Output & Results Studio (8 Cols) */}
                <div className="lg:col-span-8 space-y-6">
                    {/* Live Agent Terminal Stream */}
                    <div className="rounded-2xl overflow-hidden border border-slate-800 bg-[#090d16] shadow-xl">
                        {/* Terminal Title Bar */}
                        <div className="bg-[#101726] px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <div className="flex gap-1.5">
                                    <div className="w-3 h-3 rounded-full bg-rose-500/80" />
                                    <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                                    <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                                </div>
                                <span className="ml-2 text-xs font-mono text-slate-400 flex items-center gap-1.5">
                                    <Terminal size={14} className="text-cyan-400" />
                                    autonomous_agent_stream.sh
                                </span>
                            </div>

                            <div className="flex items-center gap-3">
                                <button
                                    onClick={() => setTerminalAutoScroll(!terminalAutoScroll)}
                                    className={`text-[11px] font-mono px-2 py-0.5 rounded transition ${
                                        terminalAutoScroll ? 'bg-cyan-950 text-cyan-400 border border-cyan-800/60' : 'text-slate-500'
                                    }`}
                                >
                                    Auto-Scroll: {terminalAutoScroll ? 'ON' : 'OFF'}
                                </button>
                                <span className="text-[11px] font-mono text-slate-500">
                                    {visibleLogs.length} events
                                </span>
                            </div>
                        </div>

                        {/* Terminal Body */}
                        <div className="p-4 max-h-72 overflow-y-auto font-mono text-xs space-y-2 select-text">
                            {visibleLogs.length === 0 ? (
                                <div className="text-slate-500 italic py-8 text-center">
                                    {agentStatus === 'running'
                                        ? '⚡ Launching autonomous agent reasoning engine...'
                                        : 'Agent idle. Click "Launch Agent" or select a sample dataset above to start real-time execution.'}
                                </div>
                            ) : (
                                visibleLogs.map((log, i) => (
                                    <div key={i} className="flex items-start gap-2.5 leading-relaxed hover:bg-slate-900/60 p-1 rounded transition">
                                        <span className="text-slate-600 text-[10px] shrink-0 font-mono">
                                            [{log.timestamp}]
                                        </span>
                                        <span className="shrink-0">{getLogBadge(log.type)}</span>
                                        <span className={`flex-1 break-words ${
                                            log.type === 'thought' ? 'text-purple-300 font-sans' :
                                            log.type === 'tool' ? 'text-cyan-300 font-mono' :
                                            log.type === 'metric' ? 'text-emerald-300 font-medium' :
                                            log.type === 'success' ? 'text-green-300 font-bold' :
                                            log.type === 'warning' ? 'text-rose-400' : 'text-slate-300'
                                        }`}>
                                            {log.message}
                                        </span>
                                    </div>
                                ))
                            )}
                            <div ref={terminalEndRef} />
                        </div>
                    </div>

                    {/* Results Workspace: Tabs & Visualizations */}
                    {runData && (
                        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
                            {/* Tabs Navigation */}
                            <div className="flex items-center gap-1 border-b border-slate-200 bg-slate-50/70 p-2 overflow-x-auto">
                                <button
                                    onClick={() => setActiveTab('tournament')}
                                    className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition shrink-0 ${
                                        activeTab === 'tournament'
                                            ? 'bg-white text-indigo-600 shadow-sm border border-slate-200'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    <Award size={15} />
                                    <span>AutoML Tournament</span>
                                </button>
                                <button
                                    onClick={() => setActiveTab('diff')}
                                    className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition shrink-0 ${
                                        activeTab === 'diff'
                                            ? 'bg-white text-indigo-600 shadow-sm border border-slate-200'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    <ShieldCheck size={15} />
                                    <span>Data Diff & Hygiene</span>
                                </button>
                                <button
                                    onClick={() => setActiveTab('drivers')}
                                    className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition shrink-0 ${
                                        activeTab === 'drivers'
                                            ? 'bg-white text-indigo-600 shadow-sm border border-slate-200'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    <BarChart3 size={15} />
                                    <span>Feature Drivers</span>
                                </button>
                                <button
                                    onClick={() => setActiveTab('brief')}
                                    className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition shrink-0 ${
                                        activeTab === 'brief'
                                            ? 'bg-white text-indigo-600 shadow-sm border border-slate-200'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    <Bot size={15} />
                                    <span>Executive Brief</span>
                                </button>
                                <button
                                    onClick={() => setActiveTab('predict')}
                                    className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition shrink-0 ${
                                        activeTab === 'predict'
                                            ? 'bg-white text-indigo-600 shadow-sm border border-slate-200'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    <Zap size={15} />
                                    <span>Instant Predictor</span>
                                </button>
                            </div>

                            {/* Tab Content */}
                            <div className="p-6">
                                {/* TAB 1: AutoML Tournament */}
                                {activeTab === 'tournament' && (
                                    <div className="space-y-6">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <h3 className="font-bold text-slate-800 text-base">
                                                    AutoML Tournament Leaderboard
                                                </h3>
                                                <p className="text-xs text-slate-500">
                                                    Ranked by primary metric for {runData.problem_type} on target '{runData.target_column}'
                                                </p>
                                            </div>
                                            <span className="text-xs font-mono bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded-full border border-indigo-200">
                                                80/20 Train-Test Split
                                            </span>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {runData.tournament?.map((model, i) => (
                                                <div
                                                    key={model.model_name}
                                                    className={`p-4 rounded-xl border relative transition-all ${
                                                        model.is_champion
                                                            ? 'bg-gradient-to-br from-amber-50/60 to-white border-amber-300 ring-2 ring-amber-200/60 shadow-md'
                                                            : 'bg-white border-slate-200 hover:border-slate-300 shadow-sm'
                                                    }`}
                                                >
                                                    {model.is_champion && (
                                                        <div className="absolute top-3 right-3 flex items-center gap-1 bg-amber-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider shadow">
                                                            <Award size={12} />
                                                            <span>Champion</span>
                                                        </div>
                                                    )}

                                                    <div className="flex items-center gap-2 mb-2">
                                                        <span className="text-xs font-bold text-slate-400 font-mono">
                                                            #{i + 1}
                                                        </span>
                                                        <h4 className="font-bold text-slate-800 text-sm">
                                                            {model.model_name}
                                                        </h4>
                                                    </div>

                                                    <div className="grid grid-cols-2 gap-3 mt-3 pt-3 border-t border-slate-100">
                                                        <div>
                                                            <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                                                                {model.primary_metric_name}
                                                            </span>
                                                            <div className="text-lg font-extrabold text-slate-900">
                                                                {model.primary_metric}{model.primary_metric_name === 'Accuracy' ? '%' : ''}
                                                            </div>
                                                        </div>
                                                        <div>
                                                            <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                                                                {model.secondary_metric_name}
                                                            </span>
                                                            <div className="text-lg font-extrabold text-slate-900">
                                                                {model.secondary_metric}
                                                            </div>
                                                        </div>
                                                    </div>

                                                    <div className="flex items-center justify-between mt-3 pt-2 text-[11px] text-slate-500 font-mono">
                                                        <span>Train Latency: {model.latency_ms}ms</span>
                                                        {model.precision !== undefined && (
                                                            <span>Precision: {model.precision}</span>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* TAB 2: Data Diff & Hygiene */}
                                {activeTab === 'diff' && runData.data_diff && (
                                    <div className="space-y-6">
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
                                                <span className="text-[11px] font-semibold text-slate-500 uppercase">Quality Score</span>
                                                <div className="flex items-center justify-center gap-2 mt-1">
                                                    <span className="text-slate-400 line-through text-sm">{runData.data_diff.before.quality_score}%</span>
                                                    <ArrowRight size={14} className="text-emerald-500" />
                                                    <span className="text-2xl font-black text-emerald-600">{runData.data_diff.after.quality_score}%</span>
                                                </div>
                                            </div>

                                            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
                                                <span className="text-[11px] font-semibold text-slate-500 uppercase">Missing Values</span>
                                                <div className="flex items-center justify-center gap-2 mt-1">
                                                    <span className="text-slate-400 line-through text-sm">{runData.data_diff.before.missing_cells}</span>
                                                    <ArrowRight size={14} className="text-emerald-500" />
                                                    <span className="text-2xl font-black text-emerald-600">0</span>
                                                </div>
                                            </div>

                                            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
                                                <span className="text-[11px] font-semibold text-slate-500 uppercase">Feature Count</span>
                                                <div className="flex items-center justify-center gap-2 mt-1">
                                                    <span className="text-slate-400 text-sm">{runData.data_diff.before.columns} cols</span>
                                                    <ArrowRight size={14} className="text-indigo-500" />
                                                    <span className="text-2xl font-black text-indigo-600">{runData.data_diff.after.columns} cols</span>
                                                </div>
                                            </div>

                                            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
                                                <span className="text-[11px] font-semibold text-slate-500 uppercase">Duplicate Rows</span>
                                                <div className="flex items-center justify-center gap-2 mt-1">
                                                    <span className="text-slate-400 line-through text-sm">{runData.data_diff.before.duplicate_rows}</span>
                                                    <ArrowRight size={14} className="text-emerald-500" />
                                                    <span className="text-2xl font-black text-emerald-600">0</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-indigo-50/50 border border-indigo-100 rounded-xl p-4 space-y-2">
                                            <h4 className="text-xs font-bold text-indigo-950 uppercase tracking-wider">
                                                Autonomous Remediation Actions
                                            </h4>
                                            <ul className="space-y-1.5 text-xs text-indigo-900">
                                                {runData.data_diff.changes?.map((change, idx) => (
                                                    <li key={idx} className="flex items-center gap-2">
                                                        <CheckCircle2 size={14} className="text-emerald-600 shrink-0" />
                                                        <span>{change}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                )}

                                {/* TAB 3: Feature Drivers */}
                                {activeTab === 'drivers' && (
                                    <div className="space-y-6">
                                        <div>
                                            <h3 className="font-bold text-slate-800 text-base">
                                                Predictive Feature Drivers
                                            </h3>
                                            <p className="text-xs text-slate-500">
                                                Relative feature importance from champion model ({runData.champion_model?.name})
                                            </p>
                                        </div>

                                        <div className="w-full min-w-0" style={{ height: 260, minHeight: 260 }}>
                                            <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={260}>
                                                <BarChart data={runData.feature_importance} layout="vertical" margin={{ left: 40, right: 20 }}>
                                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                                                    <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
                                                    <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} width={120} />
                                                    <Tooltip
                                                        formatter={(val) => [`${val}%`, 'Relative Weight']}
                                                        contentStyle={{ backgroundColor: '#0f172a', color: '#fff', borderRadius: '8px', fontSize: '12px' }}
                                                    />
                                                    <Bar dataKey="percentage" radius={[0, 4, 4, 0]}>
                                                        {runData.feature_importance?.map((entry, index) => (
                                                            <Cell
                                                                key={`cell-${index}`}
                                                                fill={index === 0 ? '#6366f1' : index === 1 ? '#818cf8' : '#a5b4fc'}
                                                            />
                                                        ))}
                                                    </Bar>
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                )}

                                {/* TAB 4: Executive Brief */}
                                {activeTab === 'brief' && (
                                    <div className="space-y-6">
                                        <div className="prose prose-sm max-w-none text-slate-700">
                                            <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 leading-relaxed">
                                                <h4 className="text-sm font-bold text-slate-900 mb-2 flex items-center gap-2">
                                                    <Bot size={16} className="text-indigo-600" />
                                                    Executive Summary
                                                </h4>
                                                <p className="text-xs text-slate-600">{runData.executive_summary}</p>
                                            </div>
                                        </div>

                                        <div className="space-y-3">
                                            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                                                Autonomous Actionable Recommendations
                                            </h4>
                                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                                {runData.recommendations?.map((rec, i) => (
                                                    <div key={i} className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm hover:border-indigo-300 transition">
                                                        <div className="w-6 h-6 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center text-xs font-bold mb-2">
                                                            {i + 1}
                                                        </div>
                                                        <p className="text-xs text-slate-700 leading-relaxed font-medium">
                                                            {rec}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* TAB 5: Instant Live Predictor */}
                                {activeTab === 'predict' && (
                                    <div className="space-y-6">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <h3 className="font-bold text-slate-800 text-base">
                                                    Instant Live Predictor ({runData.champion_model?.name})
                                                </h3>
                                                <p className="text-xs text-slate-500">
                                                    Tweak feature inputs to run real-time inference against the trained champion model
                                                </p>
                                            </div>
                                        </div>

                                        <form onSubmit={handlePredict} className="space-y-4">
                                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 max-h-72 overflow-y-auto p-1">
                                                {Object.keys(predictInputs).map((col) => (
                                                    <div key={col} className="space-y-1">
                                                        <label className="text-[11px] font-semibold text-slate-600 truncate block">
                                                            {col}
                                                        </label>
                                                        <input
                                                            type="text"
                                                            value={predictInputs[col] ?? ''}
                                                            onChange={(e) => setPredictInputs({ ...predictInputs, [col]: e.target.value })}
                                                            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-indigo-500 font-mono"
                                                        />
                                                    </div>
                                                ))}
                                            </div>

                                            <div className="flex items-center justify-between pt-3 border-t border-slate-100">
                                                <button
                                                    type="submit"
                                                    disabled={isPredicting}
                                                    className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs py-2 px-5 rounded-xl shadow transition flex items-center gap-2"
                                                >
                                                    {isPredicting ? <RefreshCw size={14} className="animate-spin" /> : <Zap size={14} />}
                                                    <span>Compute Instant Prediction</span>
                                                </button>

                                                {predictionResult && (
                                                    <div className="flex items-center gap-3 bg-emerald-50 border border-emerald-200 px-4 py-2 rounded-xl">
                                                        <span className="text-xs font-semibold text-emerald-800">
                                                            Predicted {runData.target_column}:
                                                        </span>
                                                        <span className="text-base font-black text-emerald-700 font-mono">
                                                            {predictionResult.prediction}
                                                        </span>
                                                        {predictionResult.probability !== undefined && (
                                                            <span className="text-[11px] text-emerald-600 font-mono">
                                                                (Conf: {Math.round(predictionResult.probability * 100)}%)
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        </form>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
