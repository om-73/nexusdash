import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../services/api';
import { jwtDecode } from 'jwt-decode';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem('token'));
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const initAuth = async () => {
            if (token) {
                try {
                    const decoded = jwtDecode(token);
                    if (decoded.exp * 1000 < Date.now()) {
                        await performAutoLogin();
                    } else {
                        setUser(decoded);
                        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
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

    const performAutoLogin = async () => {
        try {
            console.log("Auto-logging in as default admin...");
            // Use the default admin credentials we seeded in server
            await login('admin@nexusdash.com', 'admin123');
        } catch (err) {
            console.error("Auto-login failed:", err);
            setLoading(false);
        }
    };

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

    const register = async (email, password) => {
        try {
            await api.post('/auth/register', { email, password });
            return true;
        } catch (error) {
            throw error;
        }
    };

    const logout = () => {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
        delete api.defaults.headers.common['Authorization'];
    };

    return (
        <AuthContext.Provider value={{ user, login, register, logout, loading, isAuthenticated: !!user }}>
            {!loading && children}
        </AuthContext.Provider>
    );
};
