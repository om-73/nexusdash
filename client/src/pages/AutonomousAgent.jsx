import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
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
    Check, Copy, ChevronRight, Sliders, ExternalLink, FileText,
    Loader2, FileDown, FolderKanban, LayoutDashboard
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
    const { 
        dataSummary, setDataSummary, setDataPreview, 
        syncAgentRunResults, setPredictionResult: setGlobalPredictionResult,
        uploadedDatasets, activeDatasetId, isUnifiedMode,
        useAllFilesTogether, useFileSeparately 
    } = useData();
    const { addToast } = useToast();

    const navigate = useNavigate();
    const location = useLocation();
    const [userQuery, setUserQuery] = useState(() => location.state?.initialQuery || '');
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
    const [isGeneratingReport, setIsGeneratingReport] = useState(false);

    // Live Prediction State
    const [predictInputs, setPredictInputs] = useState({});
    const [predictionResult, setPredictionResult] = useState(null);
    const [isPredicting, setIsPredicting] = useState(false);

    const terminalEndRef = useRef(null);
    const streamingTimerRef = useRef(null);
    const autoRunTriggered = useRef(false);

    // Dynamic suggested queries based on active dataset columns
    const suggestedQueries = React.useMemo(() => {
        if (!dataSummary?.columns || dataSummary.columns.length === 0) return [];
        const cols = dataSummary.columns;
        const targetCol = (targetColumn !== 'auto' && cols.includes(targetColumn))
            ? targetColumn
            : cols[cols.length - 1];
        return [
            `Predict ${targetCol} with champion AutoML model`,
            `Analyze top key drivers & explainability of ${targetCol}`,
            `Clean data anomalies & optimize ${targetCol}`,
            `Run full auto-pilot pipeline on ${targetCol}`
        ];
    }, [dataSummary, targetColumn]);

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
                const common = ['Churn', 'target', 'Target', 'SalePrice', 'price', 'Price', 'Attrition', 'left', 'Weekly_Sales', 'sales'];
                const found = dataSummary.columns.find(c => common.some(cand => cand.toLowerCase() === c.toLowerCase()));
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
            syncAgentRunResults?.(result);
            addToast('Autonomous Agent execution completed! Executive report ready to download.', 'success');
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
                syncAgentRunResults?.(result);
                addToast('Autonomous Agent execution completed! Executive report ready to download.', 'success');
            }
        }, 120);

        streamingTimerRef.current = interval;
    };

    // Execute Autonomous Agent
    const handleLaunchAgent = async (overrideQuery = null) => {
        setAgentStatus('running');
        setVisibleLogs([]);
        setCurrentStageIndex(0);
        setRunData(null);
        setPredictionResult(null);

        const activeQuery = (typeof overrideQuery === 'string' && overrideQuery.trim())
            ? overrideQuery.trim()
            : (userQuery.trim() || selectedGoal);

        try {
            const payload = {
                goal: activeQuery,
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

    // Auto-run if arrived from another page with initialQuery and autoRun
    useEffect(() => {
        if (!autoRunTriggered.current && location.state?.autoRun && location.state?.initialQuery && dataSummary) {
            autoRunTriggered.current = true;
            handleLaunchAgent(location.state.initialQuery);
        }
    }, [location.state, dataSummary]);

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
            setGlobalPredictionResult?.(prev => ({
                ...prev,
                prediction: res.prediction,
                sample_inputs: predictInputs,
                model_name: runData?.champion_model?.name || prev?.model_name,
                target_column: runData?.target_column || prev?.target_column,
                problem_type: runData?.problem_type || prev?.problem_type,
                timestamp: new Date().toISOString()
            }));
            addToast('Prediction computed successfully & synced to Prediction page!', 'success');
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

    // Generate and Download Full Autonomous Agent Executive Report (PDF or Markdown)
    const handleDownloadReport = async (format = 'pdf') => {
        if (!runData) {
            addToast('No agent execution results available to generate report.', 'error');
            return;
        }

        setIsGeneratingReport(true);
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const targetCol = runData.target_column || 'Target';
            const problemType = (runData.problem_type || 'Machine Learning').toUpperCase();
            const champ = runData.champion_model || {};
            const diff = runData.data_diff || {};
            const drivers = runData.feature_importance || [];
            const recs = runData.recommendations || [];
            const queryText = runData.query || userQuery || selectedGoal;
            const fe = runData.feature_engineering || {};
            const feTransformations = fe.transformations || [];
            const feExplanation = fe.explanation || (
                feTransformations.length > 0
                    ? `The Autonomous Feature Engineering Engine synthesized ${fe.total_synthetic || feTransformations.length} new predictive signals across non-linear transforms, temporal seasonality, interaction ratios, and categorical mappings.`
                    : 'The Autonomous Agent analyzed feature distributions, evaluated skewness, and prepared optimal encodings for machine learning modeling.'
            );

            if (format === 'markdown' || format === 'md') {
                // Generate comprehensive Markdown document
                let md = `# NexusDash Autonomous AI Agent - Executive Intelligence Report\n\n`;
                md += `**Execution ID:** \`${runData.execution_id || 'N/A'}\`  \n`;
                md += `**Generated Date:** ${new Date().toLocaleString()}  \n`;
                md += `**Directive / User Query:** "${queryText}"  \n`;
                md += `**Problem Type:** ${problemType}  \n`;
                md += `**Target Column:** \`${targetCol}\`  \n`;
                md += `**Pipeline Latency:** ${runData.total_duration_ms || 0} ms  \n\n`;

                md += `---\n\n## 1. Executive Summary\n\n`;
                md += `${runData.executive_summary || 'Autonomous agent successfully optimized dataset and crowned champion model.'}\n\n`;

                md += `## 2. Champion Model & Tournament Performance\n\n`;
                md += `- **Champion Model:** **${champ.name || 'AutoML Champion'}**\n`;
                md += `- **Primary Metric (${champ.metric_name || 'Score'}):** **${champ.metric_value ?? 'N/A'}**\n`;
                if (champ.secondary_metric_name) {
                    md += `- **Secondary Metric (${champ.secondary_metric_name}):** **${champ.secondary_metric_value ?? 'N/A'}**\n`;
                }
                md += `- **Inference Latency:** ${champ.latency_ms || 0} ms\n\n`;

                if (runData.tournament && runData.tournament.length > 0) {
                    md += `### AutoML Tournament Leaderboard\n\n`;
                    md += `| Rank | Model Algorithm | ${champ.metric_name || 'Primary Metric'} | Secondary Metric | Latency |\n`;
                    md += `|---|---|---|---|---|\n`;
                    runData.tournament.forEach((m, idx) => {
                        md += `| #${idx + 1} | ${m.model_name}${m.is_champion ? ' 🏆 (Champion)' : ''} | ${m.primary_metric} | ${m.secondary_metric || '-'} | ${m.latency_ms}ms |\n`;
                    });
                    md += `\n`;
                }

                md += `## 3. Data Quality & Self-Healing Audit\n\n`;
                if (diff.before && diff.after) {
                    md += `| Audit Dimension | Pre-Run (Raw) | Post-Run (Engineered) | Status |\n`;
                    md += `|---|---|---|---|\n`;
                    md += `| **Overall Quality Score** | ${diff.before.quality_score}% | **${diff.after.quality_score}%** | Improved |\n`;
                    md += `| **Row Count** | ${diff.before.rows?.toLocaleString?.() ?? diff.before.rows} | ${diff.after.rows?.toLocaleString?.() ?? diff.after.rows} | Validated |\n`;
                    md += `| **Feature Columns** | ${diff.before.columns} | ${diff.after.columns} | Engineered |\n`;
                    md += `| **Missing Cell Count** | ${diff.before.missing_cells} | **${diff.after.missing_cells}** | Repaired |\n`;
                    md += `| **Duplicate Rows** | ${diff.before.duplicate_rows} | **${diff.after.duplicate_rows}** | Cleared |\n\n`;
                }

                if (diff.changes && diff.changes.length > 0) {
                    md += `**Automated Remediation Actions:**\n`;
                    diff.changes.forEach(c => {
                        md += `- ${c}\n`;
                    });
                    md += `\n`;
                }

                md += `## 4. Autonomous Feature Engineering & Transformation Explanations\n\n`;
                md += `### Strategy & Mathematical Intuition\n\n`;
                md += `${feExplanation}\n\n`;

                if (feTransformations.length > 0) {
                    md += `### Feature Transformation Registry & Explanations\n\n`;
                    md += `| Engineered Feature | Transformation Type | Source Column(s) | Mathematical Formula | Why Applied (Rationale) | Statistical Impact |\n`;
                    md += `|---|---|---|---|---|---|\n`;
                    feTransformations.forEach(t => {
                        const srcs = Array.isArray(t.source_columns) ? t.source_columns.join(', ') : (t.source_columns || '-');
                        md += `| \`${t.name || t.feature}\` | ${t.type || 'Custom'} | \`${srcs}\` | \`${t.formula || 'Custom'}\` | ${t.rationale || '-'} | ${t.impact || '-'} |\n`;
                    });
                    md += `\n`;
                }

                md += `## 5. Top Key Feature Drivers\n\n`;
                if (drivers.length > 0) {
                    md += `| Rank | Feature Dimension | Relative Contribution (%) |\n`;
                    md += `|---|---|---|\n`;
                    drivers.slice(0, 10).forEach((d, idx) => {
                        md += `| #${idx + 1} | \`${d.feature}\` | ${d.percentage}% |\n`;
                    });
                    md += `\n`;
                }

                md += `## 6. Strategic Business Recommendations\n\n`;
                if (recs.length > 0) {
                    recs.forEach((r, idx) => {
                        md += `${idx + 1}. **${r}**\n`;
                    });
                } else {
                    md += `- Deploy the champion model into the operational inference pipeline.\n`;
                }

                md += `\n---\n*Report generated autonomously by NexusDash AI Engine.*`;

                const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `nexus_agent_report_${targetCol}_${timestamp.slice(0, 10)}.md`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                addToast('Executive report downloaded (.md)', 'success');
            } else {
                // High-fidelity vector PDF generation
                const jspdfModule = await import('jspdf');
                const jsPDF = jspdfModule.jsPDF || jspdfModule.default?.jsPDF || jspdfModule.default;
                const doc = new jsPDF('p', 'mm', 'a4');
                const pageWidth = doc.internal.pageSize.getWidth();
                const pageHeight = doc.internal.pageSize.getHeight();
                const margin = 16;
                const contentWidth = pageWidth - margin * 2;
                let y = 16;

                const checkPageBreak = (neededHeight) => {
                    if (y + neededHeight > pageHeight - 16) {
                        doc.addPage();
                        y = 18;
                    }
                };

                // Header Top Bar
                doc.setFillColor(15, 23, 42); // slate-900
                doc.rect(margin, y, contentWidth, 24, 'F');

                doc.setTextColor(255, 255, 255);
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(14);
                doc.text('NEXUSDASH AUTONOMOUS AI AGENT', margin + 6, y + 9);

                doc.setFont('helvetica', 'normal');
                doc.setFontSize(8.5);
                doc.setTextColor(148, 163, 184); // slate-400
                doc.text('EXECUTIVE MACHINE LEARNING & DATA INTELLIGENCE DOSSIER', margin + 6, y + 15.5);

                doc.setTextColor(56, 189, 248); // sky-400
                doc.setFontSize(7.5);
                doc.text(`CONFIDENTIAL • RUN ID: ${String(runData.execution_id || 'AUTO-RUN').slice(0, 24)}`, margin + 6, y + 21);

                y += 29;

                // Meta Info Grid Card
                doc.setFillColor(248, 250, 252); // slate-50
                doc.setDrawColor(226, 232, 240); // slate-200
                doc.roundedRect(margin, y, contentWidth, 19, 2, 2, 'FD');

                doc.setFontSize(7.5);
                doc.setFont('helvetica', 'bold');
                doc.setTextColor(100, 116, 139);

                doc.text('TARGET VARIABLE', margin + 4, y + 5.5);
                doc.text('PROBLEM TYPE', margin + 55, y + 5.5);
                doc.text('PIPELINE TIME', margin + 105, y + 5.5);
                doc.text('DATE GENERATED', margin + 145, y + 5.5);

                doc.setFont('helvetica', 'normal');
                doc.setFontSize(8.5);
                doc.setTextColor(15, 23, 42);
                doc.text(String(targetCol).slice(0, 24), margin + 4, y + 13);
                doc.text(String(problemType).slice(0, 22), margin + 55, y + 13);
                doc.text(`${runData.total_duration_ms || 0} ms`, margin + 105, y + 13);
                doc.text(new Date().toLocaleDateString(), margin + 145, y + 13);

                y += 24;

                // Directive / User Goal
                if (queryText) {
                    checkPageBreak(16);
                    doc.setFontSize(8);
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(79, 70, 229); // indigo-600
                    doc.text('AGENT DIRECTIVE / GOAL:', margin, y);
                    y += 4.5;

                    doc.setFont('helvetica', 'italic');
                    doc.setFontSize(8.5);
                    doc.setTextColor(51, 65, 85);
                    const splitQuery = doc.splitTextToSize(`"${queryText}"`, contentWidth);
                    doc.text(splitQuery, margin, y);
                    y += splitQuery.length * 4.2 + 5;
                }

                // Section 1: Executive Summary
                checkPageBreak(28);
                doc.setFillColor(238, 242, 255); // indigo-50
                doc.setDrawColor(199, 210, 254);
                doc.roundedRect(margin, y, contentWidth, 6, 1, 1, 'FD');
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(9);
                doc.setTextColor(49, 46, 129); // indigo-900
                doc.text('1. EXECUTIVE SUMMARY & SYNTHESIS', margin + 4, y + 4.2);
                y += 9.5;

                doc.setFont('helvetica', 'normal');
                doc.setFontSize(8.5);
                doc.setTextColor(51, 65, 85);
                const summaryLines = doc.splitTextToSize(runData.executive_summary || 'Autonomous agent successfully optimized dataset and trained predictive model.', contentWidth);
                doc.text(summaryLines, margin, y);
                y += summaryLines.length * 4.2 + 6;

                // Section 2: Champion Model & Tournament
                checkPageBreak(38);
                doc.setFillColor(240, 253, 244); // emerald-50
                doc.setDrawColor(187, 247, 208);
                doc.roundedRect(margin, y, contentWidth, 6, 1, 1, 'FD');
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(9);
                doc.setTextColor(20, 83, 45); // emerald-900
                doc.text('2. CHAMPION MODEL & AUTOML TOURNAMENT', margin + 4, y + 4.2);
                y += 9.5;

                // Champion highlight card
                doc.setFillColor(248, 250, 252);
                doc.setDrawColor(203, 213, 225);
                doc.roundedRect(margin, y, contentWidth, 18, 2, 2, 'FD');

                doc.setFont('helvetica', 'bold');
                doc.setFontSize(9.5);
                doc.setTextColor(15, 23, 42);
                doc.text(`CROWNED CHAMPION: ${champ.name || 'Automated Best Model'}`, margin + 4, y + 6);

                doc.setFontSize(8);
                doc.setFont('helvetica', 'normal');
                doc.setTextColor(71, 85, 105);
                doc.text(`${champ.metric_name || 'Primary Metric'}:`, margin + 4, y + 13);
                doc.setFont('helvetica', 'bold');
                doc.setTextColor(16, 185, 129); // emerald-600
                doc.text(String(champ.metric_value ?? 'N/A'), margin + 34, y + 13);

                if (champ.secondary_metric_name) {
                    doc.setFont('helvetica', 'normal');
                    doc.setTextColor(71, 85, 105);
                    doc.text(`${champ.secondary_metric_name}:`, margin + 70, y + 13);
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(15, 23, 42);
                    doc.text(String(champ.secondary_metric_value ?? 'N/A'), margin + 98, y + 13);
                }

                doc.setFont('helvetica', 'normal');
                doc.setTextColor(71, 85, 105);
                doc.text(`Inference Latency:`, margin + 130, y + 13);
                doc.setFont('helvetica', 'bold');
                doc.text(`${champ.latency_ms || 0} ms`, margin + 158, y + 13);

                y += 23;

                // Tournament Leaderboard Mini Table
                if (runData.tournament && runData.tournament.length > 0) {
                    checkPageBreak(runData.tournament.length * 6 + 10);
                    doc.setFontSize(8);
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(71, 85, 105);
                    doc.text('Tournament Leaderboard (Ranked):', margin, y);
                    y += 4.5;

                    // Table Header
                    doc.setFillColor(241, 245, 249);
                    doc.rect(margin, y, contentWidth, 5, 'F');
                    doc.setFontSize(7.5);
                    doc.setTextColor(100, 116, 139);
                    doc.text('RANK & MODEL', margin + 4, y + 3.5);
                    doc.text(String(champ.metric_name || 'PRIMARY METRIC').toUpperCase(), margin + 80, y + 3.5);
                    doc.text('SECONDARY METRIC', margin + 120, y + 3.5);
                    doc.text('TRAIN TIME', margin + 155, y + 3.5);
                    y += 5;

                    runData.tournament.forEach((m, i) => {
                        checkPageBreak(6);
                        doc.setFont('helvetica', m.is_champion ? 'bold' : 'normal');
                        doc.setFontSize(7.5);
                        doc.setTextColor(m.is_champion ? 16 : 51, m.is_champion ? 185 : 65, m.is_champion ? 129 : 85);
                        doc.text(`#${i + 1}  ${m.model_name}${m.is_champion ? ' (CHAMPION)' : ''}`, margin + 4, y + 3.8);
                        doc.text(String(m.primary_metric), margin + 80, y + 3.8);
                        doc.text(String(m.secondary_metric || '-'), margin + 120, y + 3.8);
                        doc.text(`${m.latency_ms} ms`, margin + 155, y + 3.8);

                        doc.setDrawColor(241, 245, 249);
                        doc.line(margin, y + 4.8, margin + contentWidth, y + 4.8);
                        y += 5.2;
                    });
                    y += 4;
                }

                // Section 3: Data Quality & Self-Healing Audit
                if (diff.before && diff.after) {
                    checkPageBreak(38);
                    doc.setFillColor(243, 244, 246);
                    doc.setDrawColor(209, 213, 219);
                    doc.roundedRect(margin, y, contentWidth, 6, 1, 1, 'FD');
                    doc.setFont('helvetica', 'bold');
                    doc.setFontSize(9);
                    doc.setTextColor(31, 41, 55);
                    doc.text('3. DATA QUALITY & SELF-HEALING AUDIT', margin + 4, y + 4.2);
                    y += 9.5;

                    // Diff Grid
                    doc.setFillColor(248, 250, 252);
                    doc.rect(margin, y, contentWidth, 18, 'F');
                    doc.setFontSize(7.5);
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(100, 116, 139);
                    doc.text('HEALTH METRIC', margin + 4, y + 4.5);
                    doc.text('BEFORE CLEANING', margin + 65, y + 4.5);
                    doc.text('AFTER REMEDIATION', margin + 115, y + 4.5);
                    doc.text('DELTA / HEALTH', margin + 155, y + 4.5);

                    doc.setFont('helvetica', 'normal');
                    doc.setFontSize(7.5);
                    doc.setTextColor(30, 41, 59);

                    // Row 1: Quality Score
                    doc.text('Quality Score', margin + 4, y + 9.5);
                    doc.text(`${diff.before.quality_score}%`, margin + 65, y + 9.5);
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(16, 185, 129);
                    doc.text(`${diff.after.quality_score}%`, margin + 115, y + 9.5);
                    doc.text(`+${(diff.after.quality_score - diff.before.quality_score).toFixed(1)}%`, margin + 155, y + 9.5);

                    // Row 2: Missing Cells
                    doc.setFont('helvetica', 'normal');
                    doc.setTextColor(30, 41, 59);
                    doc.text('Missing Value Cells', margin + 4, y + 14.5);
                    doc.text(String(diff.before.missing_cells), margin + 65, y + 14.5);
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(16, 185, 129);
                    doc.text(String(diff.after.missing_cells), margin + 115, y + 14.5);
                    doc.text('100% Imputed', margin + 155, y + 14.5);

                    y += 22;

                    if (diff.changes && diff.changes.length > 0) {
                        checkPageBreak(diff.changes.length * 4.5 + 4);
                        doc.setFont('helvetica', 'bold');
                        doc.setFontSize(7.5);
                        doc.setTextColor(71, 85, 105);
                        doc.text('Remediation Actions Executed:', margin, y);
                        y += 4.5;
                        diff.changes.forEach(c => {
                            doc.setFont('helvetica', 'normal');
                            doc.setFontSize(7.5);
                            doc.setTextColor(100, 116, 139);
                            doc.text(`• ${c}`, margin + 2, y);
                            y += 4.2;
                        });
                        y += 3;
                    }
                }

                // Section 4: Autonomous Feature Engineering Explanations
                if (feTransformations.length > 0 || fe.total_synthetic > 0) {
                    checkPageBreak(38);
                    doc.setFillColor(243, 232, 255); // purple-100
                    doc.setDrawColor(216, 180, 254);
                    doc.roundedRect(margin, y, contentWidth, 6, 1, 1, 'FD');
                    doc.setFont('helvetica', 'bold');
                    doc.setFontSize(9);
                    doc.setTextColor(88, 28, 135); // purple-900
                    doc.text('4. AUTONOMOUS FEATURE ENGINEERING & RATIONALE', margin + 4, y + 4.2);
                    y += 9.5;

                    doc.setFont('helvetica', 'normal');
                    doc.setFontSize(7.5);
                    doc.setTextColor(51, 65, 85);
                    const feSummaryLines = doc.splitTextToSize(feExplanation, contentWidth);
                    doc.text(feSummaryLines, margin, y);
                    y += feSummaryLines.length * 4 + 4;

                    feTransformations.slice(0, 8).forEach((t) => {
                        checkPageBreak(19);
                        doc.setFillColor(248, 250, 252);
                        doc.setDrawColor(226, 232, 240);
                        doc.roundedRect(margin, y, contentWidth, 17, 1.5, 1.5, 'FD');

                        doc.setFont('helvetica', 'bold');
                        doc.setFontSize(8);
                        doc.setTextColor(15, 23, 42);
                        doc.text(String(t.name || t.feature), margin + 4, y + 4.5);

                        doc.setFont('helvetica', 'normal');
                        doc.setFontSize(7.5);
                        doc.setTextColor(126, 34, 206); // purple-700
                        doc.text(String(t.type || 'Transform'), margin + 65, y + 4.5);

                        doc.setTextColor(100, 116, 139);
                        doc.text(`Formula: ${String(t.formula || 'Custom')}`, margin + 122, y + 4.5);

                        doc.setFont('helvetica', 'italic');
                        doc.setFontSize(7.5);
                        doc.setTextColor(71, 85, 105);
                        const rLines = doc.splitTextToSize(`Why: ${t.rationale || 'Engineered to isolate higher-order variance.'}`, contentWidth - 8);
                        doc.text(rLines[0] || '', margin + 4, y + 9.5);

                        doc.setFont('helvetica', 'normal');
                        doc.setTextColor(16, 185, 129); // emerald-600
                        const impText = `Impact: ${t.impact || 'Supplied non-linear split planes.'}`;
                        const impLines = doc.splitTextToSize(impText, contentWidth - 8);
                        doc.text(impLines[0] || '', margin + 4, y + 14.5);

                        y += 20;
                    });
                    y += 2;
                }

                // Section 5: Top Feature Importance Drivers
                if (drivers.length > 0) {
                    checkPageBreak(drivers.slice(0, 6).length * 5.5 + 14);
                    doc.setFillColor(254, 243, 199); // amber-100
                    doc.setDrawColor(251, 191, 36);
                    doc.roundedRect(margin, y, contentWidth, 6, 1, 1, 'FD');
                    doc.setFont('helvetica', 'bold');
                    doc.setFontSize(9);
                    doc.setTextColor(120, 53, 15);
                    doc.text('5. KEY PREDICTIVE FEATURE DRIVERS', margin + 4, y + 4.2);
                    y += 9.5;

                    drivers.slice(0, 6).forEach((d, idx) => {
                        checkPageBreak(5.5);
                        doc.setFont('helvetica', 'bold');
                        doc.setFontSize(8);
                        doc.setTextColor(30, 41, 59);
                        doc.text(`#${idx + 1}  ${d.feature}`, margin + 4, y);

                        // Progress bar fill
                        const barMax = 65;
                        const barWidth = Math.max(2, (d.percentage / 100) * barMax);
                        doc.setFillColor(224, 231, 255);
                        doc.rect(margin + 75, y - 3, barMax, 3.5, 'F');
                        doc.setFillColor(99, 102, 241);
                        doc.rect(margin + 75, y - 3, barWidth, 3.5, 'F');

                        doc.setFont('helvetica', 'bold');
                        doc.setTextColor(79, 70, 229);
                        doc.text(`${d.percentage}%`, margin + 146, y);
                        y += 5.5;
                    });
                    y += 3;
                }

                // Section 6: Recommendations
                if (recs.length > 0) {
                    checkPageBreak(recs.length * 8 + 14);
                    doc.setFillColor(245, 243, 255); // purple-50
                    doc.setDrawColor(221, 214, 254);
                    doc.roundedRect(margin, y, contentWidth, 6, 1, 1, 'FD');
                    doc.setFont('helvetica', 'bold');
                    doc.setFontSize(9);
                    doc.setTextColor(88, 28, 135);
                    doc.text('6. ACTIONABLE STRATEGIC RECOMMENDATIONS', margin + 4, y + 4.2);
                    y += 9.5;

                    recs.forEach((r, idx) => {
                        checkPageBreak(12);
                        doc.setFont('helvetica', 'bold');
                        doc.setFontSize(8);
                        doc.setTextColor(99, 102, 241);
                        doc.text(`[Action ${idx + 1}]`, margin + 4, y);

                        doc.setFont('helvetica', 'normal');
                        doc.setFontSize(8);
                        doc.setTextColor(51, 65, 85);
                        const lines = doc.splitTextToSize(r, contentWidth - 8);
                        doc.text(lines, margin + 4, y + 4);
                        y += lines.length * 4 + 5;
                    });
                }

                // Footer pagination on all pages
                const totalPages = doc.internal.getNumberOfPages();
                for (let i = 1; i <= totalPages; i++) {
                    doc.setPage(i);
                    doc.setFontSize(7.5);
                    doc.setFont('helvetica', 'normal');
                    doc.setTextColor(148, 163, 184);
                    doc.text(
                        `NexusDash Autonomous AI Engine • Page ${i} of ${totalPages} • Confidential`,
                        margin,
                        pageHeight - 8
                    );
                }

                doc.save(`nexus_agent_executive_report_${targetCol}_${timestamp.slice(0, 10)}.pdf`);
                addToast('Executive PDF report downloaded successfully!', 'success');
            }
        } catch (err) {
            console.error('Failed to generate report:', err);
            addToast(`Failed to generate report: ${err.message}`, 'error');
        } finally {
            setIsGeneratingReport(false);
        }
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

                {/* Control Area: Natural Language Query & Prompt Command */}
                <div className="mt-8 pt-6 border-t border-slate-800/80 space-y-4">
                    {/* Dataset Scope Selector for Autonomous Agent */}
                    {uploadedDatasets && uploadedDatasets.length > 1 && (
                        <div className="flex items-center justify-between flex-wrap gap-2 p-2.5 bg-slate-950/70 border border-indigo-900/40 rounded-xl text-xs">
                            <div className="flex items-center gap-2">
                                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                                    <FolderKanban size={13} className="text-cyan-400" />
                                    Agent Dataset Scope:
                                </span>
                                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                                    isUnifiedMode 
                                        ? 'bg-purple-900/70 text-purple-300 border border-purple-700/60' 
                                        : 'bg-blue-900/70 text-blue-300 border border-blue-700/60'
                                }`}>
                                    {isUnifiedMode ? '✨ All Files Combined' : 'Single File Mode'}
                                </span>
                            </div>

                            <div className="flex items-center gap-1.5 flex-wrap">
                                {/* Option 1: Use All Together */}
                                <button
                                    type="button"
                                    onClick={async () => {
                                        try {
                                            await useAllFilesTogether('concat');
                                            addToast('✨ Agent switched to Unified Master Dataset (all files combined)', 'success');
                                        } catch (e) {
                                            addToast(e.message, 'error');
                                        }
                                    }}
                                    disabled={agentStatus === 'running'}
                                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all ${
                                        isUnifiedMode
                                            ? 'bg-purple-600 text-white shadow-sm ring-1 ring-purple-400'
                                            : 'bg-slate-900 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-700'
                                    }`}
                                >
                                    <Sparkles size={12} className={isUnifiedMode ? '' : 'text-purple-400'} />
                                    <span>All Files Combined</span>
                                </button>

                                {/* Option 2: Individual Files */}
                                {uploadedDatasets.filter(d => !d.isCombined).map(d => {
                                    const isThisActive = !isUnifiedMode && (d.id === activeDatasetId || d.isActive);
                                    return (
                                        <button
                                            key={d.id}
                                            type="button"
                                            onClick={async () => {
                                                try {
                                                    await useFileSeparately(d.id);
                                                    addToast(`Switched agent target dataset to '${d.name}'`, 'info');
                                                } catch (e) {
                                                    addToast(e.message, 'error');
                                                }
                                            }}
                                            disabled={agentStatus === 'running'}
                                            className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all truncate max-w-[180px] ${
                                                isThisActive
                                                    ? 'bg-blue-600 text-white shadow-sm ring-1 ring-blue-400'
                                                    : 'bg-slate-900 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-700'
                                            }`}
                                            title={d.name}
                                        >
                                            <FileText size={12} className={isThisActive ? '' : 'text-slate-400'} />
                                            <span className="truncate">{d.name}</span>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Primary Query Command Bar */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                                <Sparkles size={15} className="text-cyan-400 animate-pulse" />
                                <span>Write Query / Instruct Autonomous Agent:</span>
                            </label>
                            <span className="text-[11px] text-slate-400 font-mono hidden sm:inline">
                                Type any goal — the agent interprets targets & pipeline dynamically
                            </span>
                        </div>

                        <div className="relative flex items-center">
                            <input
                                type="text"
                                value={userQuery}
                                onChange={(e) => setUserQuery(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && agentStatus !== 'running' && dataSummary) {
                                        handleLaunchAgent();
                                    }
                                }}
                                disabled={agentStatus === 'running'}
                                placeholder="e.g. 'Predict Weekly_Sales and discover key revenue drivers' or 'Clean data and classify high churn customers'..."
                                className="w-full bg-slate-900/90 border-2 border-indigo-500/50 hover:border-indigo-400 focus:border-cyan-400 text-white placeholder-slate-500 text-sm md:text-base rounded-2xl pl-5 pr-36 py-3.5 shadow-inner focus:outline-none focus:ring-4 focus:ring-cyan-500/20 transition-all"
                            />
                            <button
                                type="button"
                                onClick={() => handleLaunchAgent()}
                                disabled={agentStatus === 'running' || !dataSummary}
                                className={`absolute right-2 px-5 py-2.5 rounded-xl font-bold text-xs md:text-sm flex items-center gap-1.5 shadow-lg transition-all ${
                                    agentStatus === 'running'
                                        ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                                        : !dataSummary
                                        ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                                        : 'bg-gradient-to-r from-indigo-500 via-purple-600 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white shadow-indigo-500/30 hover:scale-105 active:scale-95'
                                }`}
                            >
                                {agentStatus === 'running' ? (
                                    <>
                                        <RefreshCw size={15} className="animate-spin" />
                                        <span>Running...</span>
                                    </>
                                ) : (
                                    <>
                                        <Play size={15} fill="currentColor" />
                                        <span>Run Query</span>
                                    </>
                                )}
                            </button>
                        </div>

                        {/* Suggested Query Pills */}
                        {suggestedQueries.length > 0 && (
                            <div className="flex flex-wrap items-center gap-2 pt-1">
                                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Suggested Queries:</span>
                                {suggestedQueries.map((sq, idx) => (
                                    <button
                                        key={idx}
                                        type="button"
                                        disabled={agentStatus === 'running'}
                                        onClick={() => {
                                            setUserQuery(sq);
                                            handleLaunchAgent(sq);
                                        }}
                                        className="text-xs bg-slate-800/80 hover:bg-indigo-900/80 text-slate-300 hover:text-cyan-300 border border-slate-700 hover:border-cyan-500/50 px-2.5 py-1 rounded-lg transition shadow-sm"
                                    >
                                        ⚡ {sq}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Secondary Advanced Controls: Target, Presets, Speed */}
                    <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end pt-2 border-t border-slate-800/40">
                        {/* Preset Selector */}
                        <div className="md:col-span-5 space-y-1">
                            <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                                Or Quick Objective Preset
                            </label>
                            <select
                                value={selectedGoal}
                                onChange={(e) => {
                                    setSelectedGoal(e.target.value);
                                    setUserQuery(e.target.value);
                                }}
                                disabled={agentStatus === 'running'}
                                className="w-full bg-slate-800/90 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-400 transition"
                            >
                                {GOAL_PRESETS.map(g => (
                                    <option key={g.id} value={g.label}>{g.label}</option>
                                ))}
                            </select>
                        </div>

                        {/* Target Column */}
                        <div className="md:col-span-4 space-y-1">
                            <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                                <Database size={12} className="text-indigo-400" />
                                Target Column Override
                            </label>
                            <select
                                value={targetColumn}
                                onChange={(e) => setTargetColumn(e.target.value)}
                                disabled={agentStatus === 'running'}
                                className="w-full bg-slate-800/90 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-400 transition"
                            >
                                <option value="auto">⚡ Auto-Infer from Query / Dataset</option>
                                {dataSummary?.columns?.map(col => (
                                    <option key={col} value={col}>{col}</option>
                                ))}
                            </select>
                        </div>

                        {/* Execution Speed */}
                        <div className="md:col-span-3 space-y-1">
                            <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                                <Zap size={12} className="text-amber-400" />
                                Execution Speed
                            </label>
                            <div className="grid grid-cols-2 gap-1 bg-slate-800/90 p-1 rounded-xl border border-slate-700">
                                <button
                                    type="button"
                                    onClick={() => setSpeedMode('live')}
                                    className={`text-xs font-medium py-1 rounded-lg transition ${
                                        speedMode === 'live' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
                                    }`}
                                >
                                    Live Stream
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setSpeedMode('turbo')}
                                    className={`text-xs font-medium py-1 rounded-lg transition ${
                                        speedMode === 'turbo' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
                                    }`}
                                >
                                    Turbo
                                </button>
                            </div>
                        </div>
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
                        <div className="bg-gradient-to-br from-indigo-900 to-slate-900 text-white rounded-2xl p-5 shadow-lg border border-indigo-800/40 space-y-3.5">
                            <div className="flex items-center gap-2">
                                <Award size={20} className="text-amber-400" />
                                <h3 className="font-bold text-sm">Champion Artifact Ready</h3>
                            </div>
                            <p className="text-xs text-indigo-200">
                                {runData.champion_model.name} scored {runData.champion_model.metric_value} on {runData.champion_model.metric_name}.
                            </p>

                            {/* Prominent Download Report Button */}
                            <button
                                onClick={() => handleDownloadReport('pdf')}
                                disabled={isGeneratingReport}
                                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold py-2.5 px-3 rounded-xl flex items-center justify-center gap-2 transition shadow-md shadow-emerald-600/30 disabled:opacity-50 cursor-pointer"
                            >
                                {isGeneratingReport ? <Loader2 size={15} className="animate-spin" /> : <FileText size={15} />}
                                <span>Download Report (PDF)</span>
                            </button>

                            <div className="grid grid-cols-2 gap-2 pt-0.5">
                                <button
                                    onClick={handleDownloadModel}
                                    className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition"
                                >
                                    <Download size={14} />
                                    <span>Model (.pkl)</span>
                                </button>
                                <button
                                    onClick={() => handleDownloadReport('md')}
                                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition border border-slate-700"
                                >
                                    <FileDown size={14} className="text-cyan-400" />
                                    <span>Report (.md)</span>
                                </button>
                            </div>

                            <button
                                onClick={handleCopyReport}
                                className="w-full text-slate-400 hover:text-slate-200 text-[11px] font-mono py-1 flex items-center justify-center gap-1 transition"
                            >
                                {copiedBrief ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                                <span>{copiedBrief ? 'Copied brief to clipboard!' : 'Copy brief text'}</span>
                            </button>
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

                    {/* Executive Completion Action Banner */}
                    {runData && (
                        <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 border border-indigo-500/30 rounded-2xl p-4 md:p-5 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                            <div className="flex items-center gap-3.5">
                                <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shrink-0 shadow-lg shadow-emerald-500/10">
                                    <CheckCircle2 size={22} />
                                </div>
                                <div>
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <span className="text-white font-bold text-sm md:text-base">
                                            Autonomous Pipeline Complete
                                        </span>
                                        <span className="bg-emerald-500/20 text-emerald-300 text-[10px] font-mono px-2 py-0.5 rounded-full border border-emerald-500/30 font-semibold">
                                            {runData.champion_model?.name || 'Champion'} Crowned
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-400 mt-0.5">
                                        Target: <span className="text-indigo-300 font-mono font-medium">{runData.target_column}</span> • Score: <span className="text-emerald-400 font-bold">{runData.champion_model?.metric_value}</span> ({runData.champion_model?.metric_name}) • Duration: {runData.total_duration_ms}ms
                                    </p>
                                </div>
                            </div>

                            {/* Download Buttons Group */}
                            <div className="flex items-center gap-2 w-full md:w-auto flex-wrap">
                                <button
                                    onClick={() => handleDownloadReport('pdf')}
                                    disabled={isGeneratingReport}
                                    className="flex-1 md:flex-initial bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white text-xs font-bold py-2.5 px-4 rounded-xl flex items-center justify-center gap-2 transition shadow-lg shadow-indigo-600/30 disabled:opacity-50 cursor-pointer"
                                >
                                    {isGeneratingReport ? <Loader2 size={15} className="animate-spin" /> : <FileText size={15} />}
                                    <span>Download Report (PDF)</span>
                                </button>
                                <button
                                    onClick={() => handleDownloadReport('md')}
                                    title="Download report as Markdown document"
                                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold py-2.5 px-3 rounded-xl flex items-center justify-center gap-1.5 transition cursor-pointer"
                                >
                                    <FileDown size={14} className="text-cyan-400" />
                                    <span>Report (.md)</span>
                                </button>
                                <button
                                    onClick={handleDownloadModel}
                                    title="Download trained model serialized .pkl"
                                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold py-2.5 px-3 rounded-xl flex items-center justify-center gap-1.5 transition cursor-pointer"
                                >
                                    <Download size={14} className="text-amber-400" />
                                    <span>Model (.pkl)</span>
                                </button>
                            </div>

                            {/* Auto-routed Platform Links */}
                            <div className="w-full pt-3 border-t border-slate-800/80 flex items-center justify-between flex-wrap gap-2 text-xs">
                                <span className="text-slate-400 font-medium flex items-center gap-1.5">
                                    <Sparkles size={13} className="text-indigo-400" />
                                    Results auto-synced to destination pages:
                                </span>
                                <div className="flex items-center gap-2 flex-wrap">
                                    <button
                                        onClick={() => navigate('/clean')}
                                        className="bg-slate-800/90 hover:bg-slate-700 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-700 flex items-center gap-1.5 transition cursor-pointer"
                                    >
                                        <ShieldCheck size={13} className="text-emerald-400" />
                                        <span>Cleaning Page</span>
                                    </button>
                                    <button
                                        onClick={() => navigate('/features')}
                                        className="bg-slate-800/90 hover:bg-slate-700 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-700 flex items-center gap-1.5 transition cursor-pointer"
                                    >
                                        <Sparkles size={13} className="text-indigo-400" />
                                        <span>Feature Engineering</span>
                                    </button>
                                    <button
                                        onClick={() => navigate('/model')}
                                        className="bg-slate-800/90 hover:bg-slate-700 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-700 flex items-center gap-1.5 transition cursor-pointer"
                                    >
                                        <Award size={13} className="text-amber-400" />
                                        <span>Model Studio</span>
                                    </button>
                                    <button
                                        onClick={() => navigate('/predict')}
                                        className="bg-slate-800/90 hover:bg-slate-700 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-700 flex items-center gap-1.5 transition cursor-pointer"
                                    >
                                        <Zap size={13} className="text-cyan-400" />
                                        <span>Model Prediction</span>
                                    </button>
                                    <button
                                        onClick={() => navigate('/dashboard/custom')}
                                        className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-3 py-1.5 rounded-lg border border-indigo-500 flex items-center gap-1.5 transition cursor-pointer shadow-sm"
                                    >
                                        <LayoutDashboard size={13} />
                                        <span>AI Custom Dashboard ➔</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

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
                                    onClick={() => setActiveTab('engineering')}
                                    className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition shrink-0 ${
                                        activeTab === 'engineering'
                                            ? 'bg-white text-indigo-600 shadow-sm border border-slate-200'
                                            : 'text-slate-600 hover:text-slate-900'
                                    }`}
                                >
                                    <Sparkles size={15} className="text-amber-500" />
                                    <span>Feature Engineering</span>
                                    {runData.feature_engineering?.total_synthetic > 0 && (
                                        <span className="bg-indigo-100 text-indigo-700 text-[10px] px-1.5 py-0.5 rounded-full font-mono">
                                            +{runData.feature_engineering.total_synthetic}
                                        </span>
                                    )}
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

                                {/* TAB: Feature Engineering */}
                                {activeTab === 'engineering' && (
                                    <div className="space-y-6">
                                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                            <div>
                                                <h3 className="font-bold text-slate-800 text-base flex items-center gap-2">
                                                    <Sparkles size={18} className="text-indigo-600" />
                                                    Autonomous Feature Engineering & Synthesis
                                                </h3>
                                                <p className="text-xs text-slate-500 mt-0.5">
                                                    Non-linear interaction terms, temporal seasonality, skewness normalizations, and categorical encodings.
                                                </p>
                                            </div>
                                            {runData.feature_engineering?.total_synthetic !== undefined && (
                                                <span className="bg-indigo-50 text-indigo-700 border border-indigo-200 px-3 py-1 rounded-full text-xs font-bold font-mono self-start sm:self-auto">
                                                    +{runData.feature_engineering.total_synthetic} Synthetic Features
                                                </span>
                                            )}
                                        </div>

                                        {/* Strategy Overview Card */}
                                        <div className="bg-gradient-to-r from-indigo-50/90 via-purple-50/50 to-slate-50 border border-indigo-100/80 rounded-2xl p-5 space-y-2">
                                            <h4 className="text-xs font-bold text-indigo-950 uppercase tracking-wider flex items-center gap-2">
                                                <Bot size={15} className="text-indigo-600" />
                                                Agent Transformation Strategy & Intuition
                                            </h4>
                                            <p className="text-xs text-slate-700 leading-relaxed">
                                                {runData.feature_engineering?.explanation || (
                                                    "The Autonomous Feature Engineering Engine synthesized predictive signals to uncover non-linear relationships, normalize skewed feature spaces, and stabilize variance across model benchmarks."
                                                )}
                                            </p>
                                        </div>

                                        {/* Transformation Cards / Registry */}
                                        <div className="space-y-3">
                                            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                                                Synthesized Features & Explanations ({runData.feature_engineering?.transformations?.length || 0})
                                            </h4>

                                            {(!runData.feature_engineering?.transformations || runData.feature_engineering.transformations.length === 0) ? (
                                                <div className="bg-slate-50 border border-slate-200 rounded-xl p-8 text-center text-slate-500 text-xs">
                                                    All input features possessed optimal distributions. No additional non-linear expansions required for champion model.
                                                </div>
                                            ) : (
                                                <div className="grid grid-cols-1 gap-3.5">
                                                    {runData.feature_engineering.transformations.map((t, idx) => (
                                                        <div key={idx} className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm hover:border-indigo-300 transition space-y-2.5">
                                                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
                                                                <div className="flex items-center gap-2 flex-wrap">
                                                                    <span className="font-mono font-bold text-sm text-slate-900 bg-slate-100 px-2.5 py-0.5 rounded-lg border border-slate-200">
                                                                        {t.name || t.feature}
                                                                    </span>
                                                                    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-wider ${
                                                                        String(t.type || '').toLowerCase().includes('log') ? 'bg-purple-100 text-purple-700 border border-purple-200' :
                                                                        String(t.type || '').toLowerCase().includes('temporal') || String(t.type || '').toLowerCase().includes('periodicity') ? 'bg-emerald-100 text-emerald-700 border border-emerald-200' :
                                                                        String(t.type || '').toLowerCase().includes('ratio') || String(t.type || '').toLowerCase().includes('interaction') ? 'bg-cyan-100 text-cyan-700 border border-cyan-200' :
                                                                        'bg-amber-100 text-amber-700 border border-amber-200'
                                                                    }`}>
                                                                        {t.type || 'Engineered'}
                                                                    </span>
                                                                </div>
                                                                <div className="text-[11px] font-mono text-slate-500 flex items-center gap-2">
                                                                    <span>Formula: <code className="text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded font-bold">{t.formula || 'Custom'}</code></span>
                                                                    {t.source_columns && (
                                                                        <span>• Source: <code className="text-slate-700">{Array.isArray(t.source_columns) ? t.source_columns.join(', ') : t.source_columns}</code></span>
                                                                    )}
                                                                </div>
                                                            </div>

                                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-1">
                                                                <div className="bg-slate-50/80 rounded-lg p-2.5 border border-slate-100 space-y-1">
                                                                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                                                                        Why Applied (AI Rationale)
                                                                    </span>
                                                                    <p className="text-slate-700 leading-relaxed">
                                                                        {t.rationale || 'Engineered to isolate higher-order variance.'}
                                                                    </p>
                                                                </div>
                                                                <div className="bg-emerald-50/50 rounded-lg p-2.5 border border-emerald-100 space-y-1">
                                                                    <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700">
                                                                        Statistical & Model Impact
                                                                    </span>
                                                                    <p className="text-emerald-900 leading-relaxed font-medium">
                                                                        {t.impact || 'Supplied non-linear split planes to champion model.'}
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
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
                                        {/* Action Card inside Brief */}
                                        <div className="bg-gradient-to-r from-indigo-50/90 via-slate-50 to-indigo-50/50 border border-indigo-100 rounded-2xl p-4 md:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-sm">
                                            <div className="space-y-1">
                                                <h4 className="text-xs font-bold text-indigo-950 uppercase tracking-wider flex items-center gap-2">
                                                    <FileText size={16} className="text-indigo-600" />
                                                    Executive Intelligence Dossier Ready
                                                </h4>
                                                <p className="text-xs text-slate-600">
                                                    Export the complete autonomous agent dossier including data quality diffs, tournament benchmarks, feature driver attribution, and strategic recommendations.
                                                </p>
                                            </div>
                                            <div className="flex items-center gap-2 shrink-0">
                                                <button
                                                    onClick={() => handleDownloadReport('pdf')}
                                                    disabled={isGeneratingReport}
                                                    className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold py-2.5 px-4 rounded-xl flex items-center gap-2 transition shadow-md shadow-indigo-600/20 disabled:opacity-50 cursor-pointer"
                                                >
                                                    {isGeneratingReport ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />}
                                                    <span>Download PDF Report</span>
                                                </button>
                                                <button
                                                    onClick={() => handleDownloadReport('md')}
                                                    className="bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 text-xs font-semibold py-2.5 px-3.5 rounded-xl flex items-center gap-1.5 transition shadow-sm cursor-pointer"
                                                >
                                                    <FileDown size={14} className="text-slate-500" />
                                                    <span>Markdown (.md)</span>
                                                </button>
                                            </div>
                                        </div>

                                        <div className="prose prose-sm max-w-none text-slate-700">
                                            <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 leading-relaxed">
                                                <h4 className="text-sm font-bold text-slate-900 mb-2 flex items-center gap-2">
                                                    <Bot size={16} className="text-indigo-600" />
                                                    Executive Summary
                                                </h4>
                                                <p className="text-xs text-slate-600">{runData.executive_summary}</p>
                                            </div>
                                        </div>

                                        {/* Feature Engineering Highlights inside Brief */}
                                        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
                                            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center justify-between">
                                                <span className="flex items-center gap-2">
                                                    <Sparkles size={15} className="text-indigo-600" />
                                                    Autonomous Feature Engineering & Synthesis Insights
                                                </span>
                                                {runData.feature_engineering?.total_synthetic > 0 && (
                                                    <span className="bg-indigo-50 text-indigo-700 font-mono text-[11px] px-2 py-0.5 rounded-full border border-indigo-100">
                                                        +{runData.feature_engineering.total_synthetic} engineered signals
                                                    </span>
                                                )}
                                            </h4>
                                            <p className="text-xs text-slate-600 leading-relaxed">
                                                {runData.feature_engineering?.explanation || (
                                                    "The Autonomous Feature Engineering Engine synthesized predictive signals to uncover non-linear relationships, normalize skewed feature spaces, and stabilize variance across model benchmarks."
                                                )}
                                            </p>
                                            {runData.feature_engineering?.transformations && runData.feature_engineering.transformations.length > 0 && (
                                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5 pt-1">
                                                    {runData.feature_engineering.transformations.slice(0, 6).map((t, idx) => (
                                                        <div key={idx} className="bg-slate-50 rounded-lg p-2.5 border border-slate-100 space-y-1">
                                                            <div className="flex items-center justify-between">
                                                                <span className="font-mono font-bold text-[11px] text-slate-800 truncate">
                                                                    {t.name || t.feature}
                                                                </span>
                                                                <span className="text-[9px] font-semibold text-indigo-600 bg-indigo-50 px-1.5 py-0.2 rounded uppercase">
                                                                    {t.type?.split(' ')[0] || 'Engineered'}
                                                                </span>
                                                            </div>
                                                            <p className="text-[11px] text-slate-500 line-clamp-2">
                                                                {t.rationale}
                                                            </p>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
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
