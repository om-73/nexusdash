const express = require('express');
const router = express.Router();
const dataController = require('../controllers/dataController');
const { authenticate } = require('../middleware/authMiddleware');
const auditLogger = require('../middleware/auditLogger');

// Protect all routes
router.use(authenticate);

router.post('/load', auditLogger('Load Data'), dataController.processUpload);
router.get('/state', dataController.getState);
router.post('/clean', auditLogger('Clean Data'), dataController.cleanData);
router.get('/eda', dataController.getEDA);
router.post('/train', auditLogger('Train Model'), dataController.trainModel);
router.get('/export', dataController.exportData);
router.post('/connect', auditLogger('Connect DB'), dataController.connectDatabase);
router.post('/undo', dataController.undoAction);
router.post('/redo', dataController.redoAction);
router.get('/quality', dataController.getDataQuality);
router.get('/pipeline', dataController.getPipeline);
router.get('/eda-summary', dataController.getEDASummary);
router.get('/recommend', dataController.getChartRecommendations);
router.post('/query', dataController.queryMachineLearning);
router.get('/kpi', dataController.getKPIs);
router.get('/model/download', dataController.downloadModel);

// Observability & Contracts
router.post('/register', dataController.registerDataset);
router.get('/health/:id', dataController.checkHealth);
router.post('/contracts', dataController.saveContract);
router.post('/validate-contract', dataController.validateContract);

// Compliance (PII)
router.get('/scan-pii', dataController.scanPII);
router.post('/mask-data', auditLogger('Mask Data'), dataController.maskData);

// Intelligence & AI
router.post('/ai/chat', dataController.aiChat);
router.post('/ai/configure', dataController.configureAI);
router.post('/analyze-drivers', dataController.analyzeDrivers);

// Analytics & Engineering
router.post('/query/builder', dataController.queryBuilder);
router.post('/features/register', dataController.registerFeature);
router.get('/features', dataController.getFeatures);
router.post('/notebook', dataController.executeNotebook);

// Time Travel & History
router.get('/history/snapshots/:id', dataController.getSnapshotHistory);
router.post('/history/revert', dataController.revertToSnapshot);
router.post('/history/diff', dataController.diffSnapshots);

// Custom Dashboards
router.post('/dashboards/save', dataController.saveDashboard);
router.get('/dashboards', dataController.listDashboards);
router.get('/dashboards/:id', dataController.getDashboard);

module.exports = router;
