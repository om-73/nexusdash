const fs = require('fs');
const path = require('path');

const AUDIT_FILE = path.join(__dirname, '../audit.log');

const auditLogger = (action) => {
    return (req, res, next) => {
        // We log AFTER the response finishes to capture status code, but for simplicity here we log request
        const user = req.user ? req.user.email : 'Anonymous';
        const timestamp = new Date().toISOString();
        const ip = req.ip;

        const logEntry = `[${timestamp}] User: ${user} | Action: ${action} | IP: ${ip} | Path: ${req.originalUrl}\n`;

        // Append to file asynchronously
        fs.appendFile(AUDIT_FILE, logEntry, (err) => {
            if (err) console.error('Failed to write to audit log:', err);
        });

        next();
    };
};

module.exports = auditLogger;
