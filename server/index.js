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

// Health Check (Infrastructure Debugging)
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
        res.status(503).json({
            status: 'degraded',
            server: 'online',
            python: 'disconnected',
            details: error.message
        });
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
