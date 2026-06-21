import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Sparkles, X, Bot, RefreshCw } from 'lucide-react';
import { chatWithAI } from '../services/api';
import { useData } from '../context/DataContext';

export default function AIAssistantWidget() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: 'Hello! I am **Nexus AI**, your data intelligence assistant. ⚡\n\nHow can I help you with your data workflow, cleaning, or machine learning model today?'
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showBadge, setShowBadge] = useState(true);
    const messagesEndRef = useRef(null);
    const { dataSummary } = useData();

    // Scroll to bottom when messages change
    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, isLoading]);

    // Format markdown-like bolding for visual display
    const formatMessage = (text) => {
        if (!text) return '';
        // Simple formatter for **bold** and `code` blocks
        let formatted = text;
        // Escape HTML
        formatted = formatted
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
        
        // Bold **text**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Code `block`
        formatted = formatted.replace(/`(.*?)`/g, '<code class="bg-slate-800 text-pink-400 px-1.5 py-0.5 rounded font-mono text-xs">$1</code>');
        // Newlines to br
        formatted = formatted.replace(/\n/g, '<br />');
        return formatted;
    };

    const handleSend = async (e, textToSend = null) => {
        if (e) e.preventDefault();
        const content = textToSend || input;
        if (!content.trim() || isLoading) return;

        // Add user message
        const userMsg = { role: 'user', content: content.trim() };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            // Build system prompt with dataset context if available
            let systemInstruction = "You are Nexus AI, a helpful enterprise data intelligence assistant for the NexusDash platform. Help the user analyze their datasets, understand metrics, and build machine learning workflows. Be concise, expert, and friendly.";
            if (dataSummary) {
                systemInstruction += `\n\nActive Dataset Context:\n- Shape: ${dataSummary.shape[0]} rows, ${dataSummary.shape[1]} columns\n- Columns: ${dataSummary.columns.join(', ')}\n- Missing Values: ${JSON.stringify(dataSummary.missing_values)}\n- Types: ${JSON.stringify(dataSummary.dtypes)}`;
            }

            // Call Gemini via Express Backend
            // We pass the full history (up to last 10 messages)
            const chatHistory = [...messages, userMsg].slice(-10);
            const res = await chatWithAI(chatHistory, systemInstruction);
            
            setMessages(prev => [...prev, { role: 'assistant', content: res.reply }]);
        } catch (error) {
            console.error("AI Assistant Error:", error);
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                content: "I'm sorry, I encountered an issue connecting to my brain. Please verify that your Gemini API key is configured correctly in the environment." 
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const triggerSuggestion = (text) => {
        handleSend(null, text);
    };

    const resetChat = () => {
        setMessages([
            {
                role: 'assistant',
                content: 'Chat history reset. How can I assist you with your data analysis?'
            }
        ]);
    };

    const suggestions = [
        "Explain the active dataset structure",
        "How do I handle missing values?",
        "Explain classification vs regression model selection",
        "What are the best chart types for categorical data?"
    ];

    return (
        <div className="fixed bottom-6 right-6 z-50 font-sans">
            {/* Floating Action Button */}
            {!isOpen && (
                <button
                    onClick={() => {
                        setIsOpen(true);
                        setShowBadge(false);
                    }}
                    className="relative flex items-center justify-center w-14 h-14 rounded-full bg-gradient-to-tr from-indigo-600 via-indigo-500 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-xl hover:shadow-indigo-500/30 transform hover:scale-105 transition-all duration-300 group"
                >
                    <MessageSquare size={24} className="group-hover:rotate-12 transition-transform duration-200" />
                    
                    {/* Badge */}
                    {showBadge && (
                        <span className="absolute -top-1 -right-1 flex h-4 w-4">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-4 w-4 bg-emerald-500 text-[9px] text-white font-bold items-center justify-center">✦</span>
                        </span>
                    )}

                    {/* Tooltip */}
                    <span className="absolute right-16 scale-0 group-hover:scale-100 transition-all duration-200 origin-right bg-slate-900 text-white text-xs font-semibold px-3 py-1.5 rounded-lg whitespace-nowrap border border-slate-800 shadow-md">
                        Ask Nexus AI ✦
                    </span>
                </button>
            )}

            {/* Chat Panel */}
            {isOpen && (
                <div className="flex flex-col w-96 max-w-[calc(100vw-2rem)] h-[550px] max-h-[calc(100vh-6rem)] bg-slate-950/95 border border-slate-800/80 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-xl animate-in fade-in slide-in-from-bottom-4 duration-300">
                    
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 bg-slate-900/60 border-b border-slate-800/80">
                        <div className="flex items-center gap-2">
                            <div className="p-1.5 bg-indigo-500/10 rounded-lg text-indigo-400">
                                <Bot size={20} />
                            </div>
                            <div>
                                <h3 className="font-bold text-white text-sm flex items-center gap-1.5">
                                    Nexus AI Assistant
                                    <span className="flex h-2 w-2 relative">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                                    </span>
                                </h3>
                                <p className="text-[10px] text-indigo-400 font-semibold uppercase tracking-wider">AI Assistant</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={resetChat}
                                title="Reset Chat"
                                className="text-slate-400 hover:text-white p-1.5 hover:bg-slate-800/50 rounded-lg transition-colors"
                            >
                                <RefreshCw size={14} />
                            </button>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="text-slate-400 hover:text-white p-1.5 hover:bg-slate-800/50 rounded-lg transition-colors"
                            >
                                <X size={16} />
                            </button>
                        </div>
                    </div>

                    {/* Messages Body */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">
                        {messages.map((msg, i) => (
                            <div
                                key={i}
                                className={`flex gap-2.5 max-w-[85%] ${
                                    msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
                                }`}
                            >
                                {msg.role !== 'user' && (
                                    <div className="h-7 w-7 rounded-full bg-indigo-600/10 flex items-center justify-center text-indigo-400 text-xs shrink-0 font-bold border border-indigo-500/20">
                                        N
                                    </div>
                                )}
                                <div
                                    className={`p-3 rounded-2xl text-xs leading-relaxed ${
                                        msg.role === 'user'
                                            ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-tr-none'
                                            : 'bg-slate-900/60 border border-slate-800/50 text-slate-200 rounded-tl-none'
                                    }`}
                                >
                                    <div
                                        dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                                        className="space-y-1.5"
                                    />
                                </div>
                            </div>
                        ))}

                        {/* Loading Indicator */}
                        {isLoading && (
                            <div className="flex gap-2.5 max-w-[85%]">
                                <div className="h-7 w-7 rounded-full bg-indigo-600/10 flex items-center justify-center text-indigo-400 text-xs shrink-0 font-bold border border-indigo-500/20">
                                    N
                                </div>
                                <div className="bg-slate-900/60 border border-slate-800/50 p-3.5 rounded-2xl rounded-tl-none text-slate-400 flex items-center gap-1">
                                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Dataset context mini banner */}
                    {dataSummary && (
                        <div className="px-4 py-1.5 bg-indigo-500/5 border-t border-b border-slate-800/40 text-[10px] text-indigo-400/90 flex items-center justify-between">
                            <span className="font-medium">Active dataset context loaded</span>
                            <span className="opacity-70 font-mono text-[9px]">({dataSummary.shape[0]}x{dataSummary.shape[1]})</span>
                        </div>
                    )}

                    {/* Suggestion Chips */}
                    {messages.length === 1 && (
                        <div className="p-3 bg-slate-900/20 border-t border-slate-900/60">
                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-2 px-1">Suggested Questions</p>
                            <div className="flex flex-col gap-1.5 max-h-24 overflow-y-auto">
                                {suggestions.map((text, idx) => (
                                    <button
                                        key={idx}
                                        onClick={() => triggerSuggestion(text)}
                                        className="text-left text-[11px] text-slate-400 hover:text-white px-2.5 py-1.5 rounded-lg bg-slate-900/55 hover:bg-slate-800/60 border border-slate-800/30 transition-all duration-200"
                                    >
                                        ✦ {text}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Input Footer Form */}
                    <form onSubmit={handleSend} className="p-3 bg-slate-950 border-t border-slate-900 flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask me anything..."
                            disabled={isLoading}
                            className="flex-1 bg-slate-900 border border-slate-800 focus:border-indigo-500 focus:outline-none rounded-xl px-3 py-2 text-xs text-white placeholder-slate-500 transition-colors disabled:opacity-50"
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl transition-all disabled:opacity-40 disabled:hover:bg-indigo-600 shadow-lg shadow-indigo-600/10 flex items-center justify-center shrink-0"
                        >
                            <Send size={14} />
                        </button>
                    </form>
                </div>
            )}
        </div>
    );
}
