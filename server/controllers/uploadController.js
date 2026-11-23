const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { deleteFileImmediately } = require('../utils/fileCleanup');

// Configure Multer Storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, '../uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const uploadHandler = multer({
    storage: storage,
    limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        const mimeType = file.mimetype;

        console.log('[Debug] File upload validation - Name:', file.originalname, 'MIME:', mimeType);

        // Accept common CSV/Excel and Zip MIME types
        const acceptedMimes = [
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/csv',
            'application/csv',
            'text/plain', // Some systems send CSV as text/plain
            'application/zip',
            'application/x-zip-compressed'
        ];

        const acceptedExtensions = ['.csv', '.xlsx', '.xls', '.zip'];

        if (acceptedExtensions.includes(ext) && acceptedMimes.includes(mimeType)) {
            console.log('[Info] File accepted based on MIME and extension');
            return cb(null, true);
        }

        // Fallback: accept based on extension alone for CSV/ZIP
        if (ext === '.csv' || ext === '.zip') {
            console.log('[Info] File accepted based on extension');
            return cb(null, true);
        }

        const error = `File type not supported. Accepted: CSV, XLSX, XLS, ZIP. Got: ${ext} (${mimeType})`;
        console.error('[Error]', error);
        cb(new Error(error));
    }
}).any(); // Accepts single or multiple files under any field name

exports.uploadFile = (req, res) => {
    uploadHandler(req, res, (err) => {
        if (err) {
            console.error('[Error] Upload validation failed:', err.message);
            return res.status(400).json({ error: err.message });
        }
        
        const files = req.files || (req.file ? [req.file] : []);
        if (files.length === 0) {
            console.error('[Error] No file provided in request');
            return res.status(400).json({ error: 'Please select at least one file to upload' });
        }

        const uploadedFiles = files.map(file => ({
            filePath: file.path,
            filename: file.filename,
            originalName: file.originalname,
            size: file.size
        }));

        const primary = uploadedFiles[0];
        const responseData = {
            message: files.length > 1 ? `${files.length} files uploaded successfully` : 'File uploaded successfully',
            files: uploadedFiles,
            count: files.length,
            // Backwards compatibility for single-file consumers:
            filePath: primary.filePath,
            filename: primary.filename,
            originalName: primary.originalName,
            size: primary.size
        };
        console.log('[Debug] Upload Success:', responseData);
        res.json(responseData);
    });
};

exports.cleanupFile = async (req, res) => {
    try {
        let filePath = req.body?.filePath;
        if (!filePath && typeof req.body === 'string') {
            try {
                filePath = JSON.parse(req.body).filePath;
            } catch (e) {}
        }
        if (!filePath && req.query?.filePath) {
            filePath = req.query.filePath;
        }

        if (!filePath) {
            return res.status(400).json({ error: 'filePath parameter is missing' });
        }
        
        // Prevent Path Traversal
        const uploadDir = path.resolve(__dirname, '../uploads');
        const resolvedPath = path.resolve(filePath);
        if (!resolvedPath.startsWith(uploadDir)) {
            console.error('[Security] Attempted directory traversal deletion:', filePath);
            return res.status(403).json({ error: 'Access denied: File must reside in uploads directory' });
        }

        if (fs.existsSync(resolvedPath)) {
            await deleteFileImmediately(resolvedPath);
            return res.json({ message: 'File deleted automatically' });
        }
        res.json({ message: 'File already deleted or not found' });
    } catch (err) {
        console.error('[Error] Cleanup failed:', err.message);
        res.status(500).json({ error: 'Failed to delete file' });
    }
};
