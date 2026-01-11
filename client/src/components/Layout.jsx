import React, { useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Database, Home, BarChart2, Cpu, Wrench, Settings, Menu, X } from 'lucide-react';
import clsx from 'clsx';
import logo from '../assets/logo.png';

const SidebarItem = ({ to, icon: Icon, label, onClick }) => (
    <NavLink
        to={to}
        onClick={onClick}
        className={({ isActive }) =>
            clsx(
                'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group',
                isActive
                    ? 'bg-primary text-white shadow-lg shadow-primary/30'
                    : 'text-slate-500 hover:bg-white hover:text-primary hover:shadow-sm'
            )
        }
    >
        <Icon size={20} />
        <span className="font-medium">{label}</span>
    </NavLink>
);

const Sidebar = ({ isOpen, onClose }) => {
    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 md:hidden backdrop-blur-sm"
                    onClick={onClose}
                />
            )}

            {/* Sidebar Container */}
            <div className={clsx(
                "w-64 h-screen bg-slate-50 border-r border-slate-200 flex flex-col p-4 fixed left-0 top-0 z-50 transition-transform duration-300 md:translate-x-0 shadow-xl md:shadow-none",
                isOpen ? "translate-x-0" : "-translate-x-full"
            )}>
                <div className="flex items-center justify-between mb-8 px-2">
                    <div className="flex items-center gap-3">
                        <img src={logo} alt="NexusDash Logo" className="w-10 h-10 object-contain rounded-lg shadow-sm" />
                        <div>
                            <h1 className="font-bold text-slate-800 text-lg leading-tight">NexusDash</h1>
                            <p className="text-xs text-slate-500 font-medium">Pro Platform</p>
                        </div>
                    </div>
                    {/* Close button for mobile */}
                    <button onClick={onClose} className="md:hidden text-slate-400 hover:text-slate-600">
                        <X size={24} />
                    </button>
                </div>

                <nav className="flex-1 space-y-1">
                    <SidebarItem to="/" icon={Home} label="Overview" onClick={onClose} />
                    <SidebarItem to="/load" icon={Database} label="Data Source" onClick={onClose} />
                    <SidebarItem to="/clean" icon={Wrench} label="Data Cleaning" onClick={onClose} />
                    <SidebarItem to="/eda" icon={BarChart2} label="Exploratory Analysis" onClick={onClose} />
                    <SidebarItem to="/model" icon={Cpu} label="Machine Learning" onClick={onClose} />
                </nav>

                <div className="mt-auto pt-4 border-t border-slate-200">
                    <SidebarItem to="/settings" icon={Settings} label="Settings" onClick={onClose} />
                </div>
            </div>
        </>
    );
};

export default function Layout() {
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    return (
        <div className="min-h-screen bg-white text-slate-900">
            {/* Mobile Header */}
            <div className="md:hidden fixed top-0 left-0 right-0 h-16 bg-white border-b border-slate-200 z-30 flex items-center px-4 justify-between">
                <div className="flex items-center gap-3">
                    <button onClick={() => setIsSidebarOpen(true)} className="text-slate-600 hover:text-indigo-600 transition-colors">
                        <Menu size={24} />
                    </button>
                    <span className="font-bold text-slate-800 text-lg">NexusDash</span>
                </div>
            </div>

            <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

            <div className="md:ml-64 min-h-screen pt-16 md:pt-0 transition-all duration-300">
                <Outlet />
            </div>
        </div>
    );
}
