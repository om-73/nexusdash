import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import { DataProvider } from './context/DataContext';
import { AuthProvider } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';
import ProtectedRoute from './components/ProtectedRoute';

const DataLoad = lazy(() => import('./pages/DataLoad'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const DataClean = lazy(() => import('./pages/DataClean'));
const FeatureEngineering = lazy(() => import('./pages/FeatureEngineering'));
const EDA = lazy(() => import('./pages/EDA'));
const Model = lazy(() => import('./pages/Model'));
const ModelPrediction = lazy(() => import('./pages/ModelPrediction'));
const Workflow = lazy(() => import('./pages/Workflow'));
const CustomDashboard = lazy(() => import('./pages/CustomDashboard'));

function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <DataProvider>
          <BrowserRouter>
            <Suspense fallback={<div className="flex items-center justify-center p-8 h-screen w-full"><div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div></div>}>
              <Routes>
                <Route element={<ProtectedRoute />}>
                  <Route path="/" element={<Layout />}>
                    <Route index element={<Dashboard />} />
                    <Route path="load" element={<DataLoad />} />
                    <Route path="clean" element={<DataClean />} />
                    <Route path="features" element={<FeatureEngineering />} />
                    <Route path="eda" element={<EDA />} />
                    <Route path="model" element={<Model />} />
                    <Route path="predict" element={<ModelPrediction />} />
                    <Route path="workflow" element={<Workflow />} />
                    <Route path="dashboard/custom" element={<CustomDashboard />} />
                    <Route path="settings" element={<div className="p-8">Settings</div>} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                  </Route>
                </Route>
              </Routes>
            </Suspense>
          </BrowserRouter>
        </DataProvider>
      </ToastProvider>
    </AuthProvider>
  );
}

export default App;
