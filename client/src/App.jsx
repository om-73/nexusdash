import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import { DataProvider } from './context/DataContext';
import { AuthProvider } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';
import ProtectedRoute from './components/ProtectedRoute';

import DataLoad from './pages/DataLoad';
import Dashboard from './pages/Dashboard';
import DataClean from './pages/DataClean';
import EDA from './pages/EDA';
import Model from './pages/Model';
import CustomDashboard from './pages/CustomDashboard';
import CustomDashboard from './pages/CustomDashboard';

function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <DataProvider>
          <BrowserRouter>
            <Routes>
              {/* Login removed for auto-auth demo mode */}
              {/* <Route path="/login" element={<Login />} /> */}

              <Route element={<ProtectedRoute />}>
                <Route path="/" element={<Layout />}>
                  <Route index element={<Dashboard />} />
                  <Route path="load" element={<DataLoad />} />
                  <Route path="clean" element={<DataClean />} />
                  <Route path="eda" element={<EDA />} />
                  <Route path="model" element={<Model />} />
                  <Route path="dashboard/custom" element={<CustomDashboard />} />
                  <Route path="settings" element={<div className="p-8">Settings</div>} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Route>
              </Route>
            </Routes>
          </BrowserRouter>
        </DataProvider>
      </ToastProvider>
    </AuthProvider>
  );
}

export default App;
