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

// Routes
// Routes
app.use('/api/auth', require('./routes/authRoutes'));
app.use('/api/upload', require('./routes/uploadRoutes')); // Uploads might need protection too, but sticking to data for now
app.use('/api/data', require('./routes/dataRoutes')); // We will apply protection inside dataRoutes to allow granular control or just apply globally
// Or apply globally:
// const { authenticate } = require('./middleware/authMiddleware');
// app.use('/api/data', authenticate, require('./routes/dataRoutes'));

app.get('/', (req, res) => {
    res.send('Data Analysis Platform API is running');
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
