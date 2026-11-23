import React, { useState, useRef, useEffect } from 'react';
import { 
    Layers, FileText, ChevronDown, Check, Sparkles, FolderKanban, 
    ArrowRight, Plus, RefreshCw, GitMerge 
} from 'lucide-react';
import { useData } from '../context/DataContext';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../context/ToastContext';

export default function DatasetScopeBar() {
    const {
        uploadedDatasets,
        activeDatasetId,
        isUnifiedMode,
        useAllFilesTogether,
        useFileSeparately,
        dataSummary,
        loading
    } = useData();

    const [dropdownOpen, setDropdownOpen] = useState(false);
    const dropdownRef = useRef(null);
    const navigate = useNavigate();
    const { addToast } = useToast();

    // Close dropdown on outside click
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    if (!uploadedDatasets || uploadedDatasets.length === 0) {
        return null;
    }

    const rawDatasets = uploadedDatasets.filter(d => !d.isCombined);
    const activeDataset = uploadedDatasets.find(d => d.id === activeDatasetId) || uploadedDatasets[0];

    const handleActivateAllTogether = async () => {
        try {
            setDropdownOpen(false);
            await useAllFilesTogether('concat');
            addToast(`✨ All ${rawDatasets.length} files combined into a unified master dataset!`, 'success');
        } catch (err) {
            addToast(`Failed to combine files: ${err.message}`, 'error');
        }
    };

    const handleSelectSeparateFile = async (id, name) => {
        try {
            setDropdownOpen(false);
            await useFileSeparately(id);
            addToast(`Switched to '${name}' (Separate mode)`, 'info');
        } catch (err) {
            addToast(`Failed to switch file: ${err.message}`, 'error');
        }
    };

    return (
        <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 border-b border-indigo-900/30 text-white px-4 md:px-6 py-2.5 shadow-md flex items-center justify-between flex-wrap gap-3 relative z-30">
            {/* Left: Current Active Dataset Identity & Scope */}
            <div className="flex items-center gap-3 min-w-0">
                <div className="flex items-center gap-2 shrink-0">
                    {isUnifiedMode ? (
                        <div className="p-1.5 bg-gradient-to-tr from-purple-500 to-indigo-500 text-white rounded-lg shadow-sm">
                            <Sparkles size={16} className="animate-pulse" />
                        </div>
                    ) : (
                        <div className="p-1.5 bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-lg">
                            <FileText size={16} />
                        </div>
                    )}
                    <div className="text-xs">
                        <span className="text-slate-400 uppercase font-semibold text-[10px] tracking-wider block">
                            Active Scope
                        </span>
                        <div className="flex items-center gap-1.5 font-bold text-white text-xs md:text-sm">
                            <span className="truncate max-w-[200px] md:max-w-[320px]">
                                {isUnifiedMode ? '✨ All Files Combined (Unified)' : (activeDataset?.name || 'Dataset')}
                            </span>
                            {dataSummary?.shape && (
                                <span className="text-[11px] font-normal text-slate-300 font-mono bg-slate-800/80 px-2 py-0.5 rounded-full border border-slate-700/60 hidden sm:inline">
                                    {dataSummary.shape[0]?.toLocaleString()} rows × {dataSummary.shape[1]} cols
                                </span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Scope Badge */}
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider hidden lg:inline-flex items-center gap-1 border ${
                    isUnifiedMode 
                        ? 'bg-purple-900/60 text-purple-300 border-purple-700/60' 
                        : 'bg-blue-900/60 text-blue-300 border-blue-700/60'
                }`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${isUnifiedMode ? 'bg-purple-400' : 'bg-blue-400'} animate-pulse`} />
                    {isUnifiedMode ? 'Combined Together' : 'Separate File'}
                </span>
            </div>

            {/* Right: Quick Scope Switching Controls & Selector */}
            <div className="flex items-center gap-2 ml-auto" ref={dropdownRef}>
                {/* 1-Click "Use All Together" Toggle Button */}
                {rawDatasets.length > 1 && (
                    <button
                        onClick={handleActivateAllTogether}
                        disabled={loading || isUnifiedMode}
                        title="Merge and analyze all uploaded files together as one master dataset"
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 shadow-sm ${
                            isUnifiedMode
                                ? 'bg-purple-600 text-white ring-2 ring-purple-400/50 cursor-default'
                                : 'bg-slate-800 hover:bg-purple-900/60 text-purple-300 hover:text-white border border-purple-700/40 hover:border-purple-500'
                        }`}
                    >
                        <Sparkles size={13} className={isUnifiedMode ? '' : 'text-purple-400'} />
                        <span className="hidden sm:inline">Use All Together</span>
                        <span className="sm:hidden">Together</span>
                    </button>
                )}

                {/* Dataset Selector Dropdown */}
                <div className="relative">
                    <button
                        onClick={() => setDropdownOpen(!dropdownOpen)}
                        disabled={loading}
                        className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700/80 border border-slate-700 text-slate-200 rounded-lg text-xs font-semibold flex items-center gap-2 transition-colors shadow-sm"
                    >
                        <FolderKanban size={14} className="text-blue-400" />
                        <span className="hidden sm:inline">Switch File</span>
                        <ChevronDown size={14} className={`transition-transform duration-200 ${dropdownOpen ? 'rotate-180' : ''}`} />
                    </button>

                    {dropdownOpen && (
                        <div className="absolute right-0 mt-2 w-72 md:w-80 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden py-2 z-50 text-slate-200 backdrop-blur-md">
                            <div className="px-3 py-1.5 border-b border-slate-800 text-[10px] font-bold uppercase tracking-wider text-slate-400 flex justify-between items-center">
                                <span>Choose Dataset Scope</span>
                                <span className="text-slate-500">{uploadedDatasets.length} total</span>
                            </div>

                            {/* Option 1: Use All Together */}
                            {rawDatasets.length > 1 && (
                                <button
                                    onClick={handleActivateAllTogether}
                                    className={`w-full px-3 py-2 text-left text-xs font-semibold flex items-center justify-between transition-colors border-b border-slate-800/80 ${
                                        isUnifiedMode
                                            ? 'bg-purple-950/60 text-purple-300'
                                            : 'hover:bg-slate-800 text-slate-300 hover:text-white'
                                    }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <Sparkles size={14} className="text-purple-400" />
                                        <div>
                                            <span className="block font-bold">✨ Use All Files Together</span>
                                            <span className="text-[10px] text-slate-400 font-normal">
                                                Unified union of all {rawDatasets.length} files
                                            </span>
                                        </div>
                                    </div>
                                    {isUnifiedMode && <Check size={14} className="text-purple-400" />}
                                </button>
                            )}

                            {/* Section: Individual Files */}
                            <div className="max-h-60 overflow-y-auto divide-y divide-slate-800/50 py-1">
                                <div className="px-3 py-1 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                                    Individual Files (Use Separately)
                                </div>
                                {uploadedDatasets.map(ds => {
                                    const isActive = !isUnifiedMode && (ds.id === activeDatasetId || ds.isActive);
                                    return (
                                        <button
                                            key={ds.id}
                                            onClick={() => handleSelectSeparateFile(ds.id, ds.name)}
                                            className={`w-full px-3 py-2 text-left text-xs flex items-center justify-between transition-colors ${
                                                isActive
                                                    ? 'bg-blue-950/60 text-blue-300 font-bold'
                                                    : 'hover:bg-slate-800 text-slate-300'
                                            }`}
                                        >
                                            <div className="min-w-0 pr-2">
                                                <div className="truncate text-xs flex items-center gap-1.5">
                                                    <FileText size={13} className="text-slate-400 shrink-0" />
                                                    <span className="truncate">{ds.name}</span>
                                                </div>
                                                <div className="text-[10px] text-slate-400 mt-0.5 font-mono">
                                                    {ds.shape ? `${ds.shape[0]?.toLocaleString()} rows × ${ds.shape[1]} cols` : 'Ready to inspect'}
                                                </div>
                                            </div>
                                            {isActive && <Check size={14} className="text-blue-400 shrink-0" />}
                                        </button>
                                    );
                                })}
                            </div>

                            {/* Footer link to Upload & Combine */}
                            <div className="pt-2 px-2 border-t border-slate-800 flex items-center justify-between gap-1">
                                <button
                                    onClick={() => {
                                        setDropdownOpen(false);
                                        navigate('/load');
                                    }}
                                    className="w-full text-center py-1.5 text-xs text-blue-400 hover:text-blue-300 hover:bg-slate-800 rounded-lg transition-colors font-semibold flex items-center justify-center gap-1.5"
                                >
                                    <Plus size={13} />
                                    Manage / Upload More Files
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

