import React, { createContext, useContext, useState } from 'react';

const DataContext = createContext();

import { getState } from '../services/api';
import { useAuth } from './AuthContext';

export const DataProvider = ({ children }) => {
    const { isAuthenticated } = useAuth();
    const [dataSummary, setDataSummary] = useState(null);
    const [dataPreview, setDataPreview] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Re-hydrate state on mount or login
    React.useEffect(() => {
        if (isAuthenticated) {
            // If we already have data, don't re-fetch immediately to avoid race conditions
            // or overwriting with null if backend is resetting.
            // However, we might want to validate it? For now, trust local state if present.
            if (!dataSummary) {
                const fetchState = async () => {
                    try {
                        const res = await getState();
                        if (res) {
                            setDataSummary(res);
                            if (res.preview) setDataPreview(res.preview);
                        }
                    } catch (err) {
                        console.warn("Failed to restore state", err);
                    }
                };
                fetchState();
            }
        } else {
            setDataSummary(null);
        }
    }, [isAuthenticated, dataSummary]);

    const refreshData = async () => {
        setLoading(true);
        try {
            const res = await getState();
            if (res) {
                setDataSummary(res);
                if (res.preview) setDataPreview(res.preview);
            }
        } catch (err) {
            console.warn("Failed to refresh state", err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <DataContext.Provider value={{
            dataSummary, setDataSummary,
            dataPreview, setDataPreview,
            loading, setLoading,
            error, setError,
            refreshData
        }}>
            {children}
        </DataContext.Provider>
    );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useData = () => {
    const context = useContext(DataContext);
    if (!context) {
        throw new Error('useData must be used within a DataProvider');
    }
    return context;
};
