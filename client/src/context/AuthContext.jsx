import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../services/api';
// Removing external dependency to prevent potential import crashes
// import { jwtDecode } from 'jwt-decode';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem('token'));
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Helper to decode JWT manually
    const parseJwt = (token) => {
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(window.atob(base64).split('').map(function (c) {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            }).join(''));
            return JSON.parse(jsonPayload);
        } catch (e) {
            return null;
        }
    };

    useEffect(() => {
        const initAuth = async () => {
            if (token) {
                try {
                    const decoded = parseJwt(token);
                    if (decoded && decoded.exp * 1000 < Date.now()) {
                        await performAutoLogin();
                    } else if (decoded) {
                        setUser(decoded);
                        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
                    } else {
                        await performAutoLogin();
                    }
                } catch (e) {
                    await performAutoLogin();
                }
            } else {
                await performAutoLogin();
            }
            setLoading(false);
        };
        initAuth();
    }, [token]);

    const login = async (email, password) => {
        try {
            // Because api points to /api header, we need to go to /auth/login
            // Wait, api base is /api. So /auth/login becomes /api/auth/login. Correct.
            const res = await api.post('/auth/login', { email, password });
            const { token, user } = res.data;

            localStorage.setItem('token', token);
            setToken(token);
            setUser(user);
            api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            return true;
        } catch (error) {
            console.error(error);
            throw error;
        }
    };

    const performAutoLogin = async () => {
        try {
            console.log("Auto-logging in as default admin...");
            await login('admin@nexusdash.com', 'admin123');
            setError(null);
        } catch (err) {
            console.error("Auto-login failed:", err);
            setError("Failed to connect to the server. Please check your internet connection or try again later.");
            setLoading(false);
        }
    };

    const register = async (email, password) => {
        try {
            await api.post('/auth/register', { email, password });
            return true;
        } catch (error) {
            throw error;
        }
    };

    // ... login/register/logout methods ...

    const logout = () => {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
        delete api.defaults.headers.common['Authorization'];
    };

    if (loading) {
        return <div className="min-h-screen flex items-center justify-center bg-slate-50 text-slate-500">Loading...</div>;
    }

    if (error && !user) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 p-4">
                <div className="bg-white p-8 rounded-xl shadow-lg max-w-md w-full text-center">
                    <h2 className="text-xl font-bold text-red-600 mb-2">Connection Failed</h2>
                    <p className="text-slate-600 mb-6">{error}</p>
                    <button
                        onClick={() => window.location.reload()}
                        className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
                    >
                        Retry Connection
                    </button>

                </div>
            </div>
        );
    }

    return (
        <AuthContext.Provider value={{ user, login, register, logout, loading, isAuthenticated: !!user }}>
            {children}
        </AuthContext.Provider>
    );
};
