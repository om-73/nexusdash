const express = require('express');
const router = express.Router();
const agentController = require('../controllers/agentController');
const { authenticate } = require('../middleware/authMiddleware');
const auditLogger = require('../middleware/auditLogger');

// All agent routes protected by authentication
router.use(authenticate);

router.post('/run', auditLogger('Autonomous Agent Run'), agentController.runAgent);
router.post('/sample', auditLogger('Load Agent Sample'), agentController.loadSample);
router.post('/plan', agentController.getPlan);

module.exports = router;
