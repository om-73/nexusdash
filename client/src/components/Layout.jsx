import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Database, Home, BarChart2, Cpu, Wrench, Settings } from 'lucide-react';
import clsx from 'clsx';

const SidebarItem = ({ to, icon: Icon, label }) => (
    <NavLink
        to={to}
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

const Sidebar = () => {
    return (
        <div className="w-64 h-screen bg-slate-50 border-r border-slate-200 flex flex-col p-4 fixed left-0 top-0">
            <div className="flex items-center gap-3 px-4 py-4 mb-8">
                <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-primary/30">
                    DA
                </div>
                <div>
                    <h1 className="font-bold text-slate-800 text-lg leading-tight">NexusDash</h1>
                    <p className="text-xs text-slate-500 font-medium">Pro Platform</p>
                </div>
            </div>

            <nav className="flex-1 space-y-1">
                <SidebarItem to="/" icon={Home} label="Overview" />
                <SidebarItem to="/load" icon={Database} label="Data Source" />
                <SidebarItem to="/clean" icon={Wrench} label="Data Cleaning" />
                <SidebarItem to="/eda" icon={BarChart2} label="Exploratory Analysis" />
                <SidebarItem to="/model" icon={Cpu} label="Machine Learning" />
            </nav>

            <div className="mt-auto pt-4 border-t border-slate-200">
                <SidebarItem to="/settings" icon={Settings} label="Settings" />
            </div>
        </div>
    );
};

export default function Layout() {
    return (
        <div className="min-h-screen bg-white text-slate-900">
            <Sidebar />
            <div className="ml-64 min-h-screen">
                <Outlet />
            </div>
        </div>
    );
}
