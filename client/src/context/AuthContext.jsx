import React, { createContext, useContext, useState, useEffect } from 'react';
import { api, deleteFile } from '../services/api';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

// Fast JWT parser without external dependency
const parseJwt = (token) => {
    try {
        if (!token) return null;
        const base64Url = token.split('.')[1];
        if (!base64Url) return null;
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(window.atob(base64).split('').map(function (c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        return JSON.parse(jsonPayload);
    } catch (e) {
        return null;
    }
};

// Check token synchronously on script evaluation for 0ms initial render delay
const getInitialAuth = () => {
    try {
        const savedToken = localStorage.getItem('token');
        if (savedToken) {
            const decoded = parseJwt(savedToken);
            if (decoded && decoded.exp * 1000 > Date.now()) {
                api.defaults.headers.common['Authorization'] = `Bearer ${savedToken}`;
                return { token: savedToken, user: decoded, isValid: true };
            }
        }
    } catch (e) {
        console.warn('Error reading stored auth:', e);
    }
    return { token: null, user: null, isValid: false };
};

// Deduplicate concurrent auto-logins (e.g. from React StrictMode)
let inFlightLoginPromise = null;

export const AuthProvider = ({ children }) => {
    const initialAuth = React.useMemo(() => getInitialAuth(), []);
    const [user, setUser] = useState(initialAuth.user);
    const [token, setToken] = useState(initialAuth.token);
    const [loading, setLoading] = useState(!initialAuth.isValid);
    const [error, setError] = useState(null);

    const login = async (email, password) => {
        try {
            const res = await api.post('/auth/login', { email, password });
            const { token: newToken, user: newUser } = res.data;

            localStorage.setItem('token', newToken);
            setToken(newToken);
            setUser(newUser);
            api.defaults.headers.common['Authorization'] = `Bearer ${newToken}`;
            setError(null);
            return true;
        } catch (error) {
            console.error(error);
            throw error;
        }
    };

    const performAutoLogin = async () => {
        if (inFlightLoginPromise) return inFlightLoginPromise;
        inFlightLoginPromise = (async () => {
            try {
                await login('admin@nexusdash.com', 'admin123');
                setError(null);
            } catch (err) {
                console.error("Auto-login failed:", err);
                setError("Failed to connect to the server. Please check your connection or try again.");
            } finally {
                setLoading(false);
                inFlightLoginPromise = null;
            }
        })();
        return inFlightLoginPromise;
    };

    useEffect(() => {
        if (!initialAuth.isValid) {
            performAutoLogin();
        }
    }, [initialAuth.isValid]);

    const register = async (email, password) => {
        try {
            await api.post('/auth/register', { email, password });
            return true;
        } catch (error) {
            throw error;
        }
    };

    const logout = () => {
        const activeFile = sessionStorage.getItem('active_session_file');
        if (activeFile) {
            deleteFile(activeFile).catch(() => {});
            sessionStorage.removeItem('active_session_file');
        }
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
        delete api.defaults.headers.common['Authorization'];
    };

    if (loading && !user) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-slate-50 text-slate-500">
                <div className="flex flex-col items-center gap-3">
                    <div className="w-8 h-8 rounded-full border-4 border-slate-200 border-t-primary animate-spin"></div>
                    <span className="text-sm font-medium text-slate-600">Connecting to platform...</span>
                </div>
            </div>
        );
    }

    if (error && !user) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 p-4">
                <div className="bg-white p-8 rounded-xl shadow-lg max-w-md w-full text-center">
                    <h2 className="text-xl font-bold text-red-600 mb-2">Connection Failed</h2>
                    <p className="text-slate-600 mb-6">{error}</p>
                    <button
                        onClick={() => {
                            setError(null);
                            setLoading(true);
                            performAutoLogin();
                        }}
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
