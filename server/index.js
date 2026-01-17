const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request Logger
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    next();
});

const { setupCleanupSchedule } = require('./utils/fileCleanup');

// Uploads directory
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}
app.use('/uploads', express.static(uploadDir));

// Initialize Auto-Cleanup
setupCleanupSchedule(uploadDir);

// Initialize Auth (Seed default user)
const { initializeDefaultUser } = require('./utils/authUtils');
initializeDefaultUser();

// Routes
app.use('/api/auth', require('./routes/authRoutes'));
app.use('/api/upload', require('./routes/uploadRoutes'));
app.use('/api/data', require('./routes/dataRoutes'));

// python process management removed to avoid conflict with start-dev.sh
// The Python engine is started separately (e.g. by start-dev.sh or Docker)
/*
const { spawn } = require('child_process');

// Logs buffer
let recentLogs = "Initializing...";

// Start Python Engine Programmatically
const pythonCwd = path.join(__dirname, '../data_engine');
// Use absolute path to Docker's main Python (where pip packages are)
const pythonProcess = spawn('/usr/local/bin/python', ['-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'], {
    cwd: pythonCwd,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
});

pythonProcess.stdout.on('data', (data) => {
    const chunk = data.toString();
    console.log('[Python]', chunk);
    recentLogs += chunk;
    if (recentLogs.length > 5000) recentLogs = recentLogs.slice(-5000);
});

pythonProcess.stderr.on('data', (data) => {
    const chunk = data.toString();
    console.error('[Python ERR]', chunk);
    recentLogs += chunk;
    if (recentLogs.length > 5000) recentLogs = recentLogs.slice(-5000);
});

pythonProcess.on('close', (code) => {
    const msg = `\n[FATAL] Python process exited with code ${code}`;
    console.error(msg);
    recentLogs += msg;
});
*/

// Logs API
app.get('/api/health', async (req, res) => {
    try {
        const pythonUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
        const axios = require('axios');
        const pythonRes = await axios.get(`${pythonUrl}/`);
        res.json({
            status: 'healthy',
            server: 'online',
            python: pythonRes.status === 200 ? 'connected' : 'error',
            python_url: pythonUrl
        });
    } catch (error) {
        // Debug Frontend Path
        const clientDir = path.join(__dirname, '../client');
        let clientFiles = [];
        try {
            if (fs.existsSync(clientDir)) {
                clientFiles = fs.readdirSync(clientDir);
                const distPath = path.join(clientDir, 'dist');
                if (fs.existsSync(distPath)) {
                    clientFiles.push('dist_contents: ' + fs.readdirSync(distPath).join(', '));
                } else {
                    clientFiles.push('NO DIST FOLDER');
                }
            } else {
                clientFiles.push('NO CLIENT FOLDER');
            }
        } catch (e) {
            clientFiles.push('Error reading client dir: ' + e.message);
        }

        res.status(200).json({
            status: 'degraded',
            server: 'online',
            python: 'disconnected',
            details: error.message,
            recent_logs: recentLogs,
            client_debug: clientFiles
        });
    }
});

// Manual Cleanup Endpoint
app.post('/api/cleanup', async (req, res) => {
    try {
        console.log('[Manual Cleanup] Cleanup requested via API');
        const uploadDirPath = path.join(__dirname, 'uploads');
        
        fs.readdir(uploadDirPath, async (err, files) => {
            if (err) {
                return res.status(500).json({ error: 'Failed to read upload directory' });
            }

            const now = Date.now();
            const FILE_RETENTION_TIME = process.env.FILE_RETENTION_HOURS ? 
                (parseInt(process.env.FILE_RETENTION_HOURS) * 60 * 60 * 1000) : 
                (24 * 60 * 60 * 1000);

            let deletedFiles = [];
            let totalSize = 0;

            files.forEach(file => {
                if (file === '.gitkeep') return;

                const filePath = path.join(uploadDirPath, file);
                fs.stat(filePath, (err, stats) => {
                    if (err) return;

                    const age = now - stats.mtimeMs;
                    if (age > FILE_RETENTION_TIME) {
                        fs.unlink(filePath, (err) => {
                            if (!err) {
                                console.log(`[Manual Cleanup] Deleted: ${file}`);
                                deletedFiles.push(file);
                                totalSize += stats.size;
                            }
                        });
                    }
                });
            });

            // Send response after processing
            setTimeout(() => {
                res.json({
                    message: 'Cleanup completed',
                    deletedCount: deletedFiles.length,
                    deletedFiles: deletedFiles,
                    freedSpaceMB: (totalSize / (1024 * 1024)).toFixed(2)
                });
            }, 500);
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Serve static files from the React client
app.use(express.static(path.join(__dirname, '../client/dist')));

// Handle React routing, return all requests to React app
// Handle React routing, return all requests to React app
app.get(/(.*)/, (req, res) => {
    const indexPath = path.join(__dirname, '../client/dist', 'index.html');
    if (fs.existsSync(indexPath)) {
        res.sendFile(indexPath);
    } else {
        console.error(`Frontend build not found at: ${indexPath}`);
        res.status(500).send(`
            <html>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #e11d48;">Deployment Error: Frontend Not Found</h1>
                    <p>The backend is running, but the frontend build files are missing.</p>
                    <p><strong>Debugging Tips for Render:</strong></p>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Ensure your <strong>Build Command</strong> is set to: <code>npm run build</code></li>
                        <li>Check if the build logs show "vite build" output.</li>
                        <li>path looked for: ${indexPath}</li>
                    </ul>
                </body>
            </html>
        `);
    }
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
