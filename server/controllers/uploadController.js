const multer = require('multer');
const path = require('path');
const fs = require('fs');

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

const upload = multer({
    storage: storage,
    limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        const mimeType = file.mimetype;
        
        console.log('[Debug] File upload validation - Name:', file.originalname, 'MIME:', mimeType);
        
        // Accept common CSV/Excel MIME types
        const acceptedMimes = [
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/csv',
            'application/csv',
            'text/plain' // Some systems send CSV as text/plain
        ];
        
        const acceptedExtensions = ['.csv', '.xlsx', '.xls'];
        
        if (acceptedExtensions.includes(ext) && acceptedMimes.includes(mimeType)) {
            console.log('[Info] File accepted based on MIME and extension');
            return cb(null, true);
        }
        
        // Fallback: accept based on extension alone for CSV
        if (ext === '.csv') {
            console.log('[Info] File accepted based on .csv extension');
            return cb(null, true);
        }

        const error = `File type not supported. Accepted: CSV, XLSX, XLS. Got: ${ext} (${mimeType})`;
        console.error('[Error]', error);
        cb(new Error(error));
    }
}).single('file'); // 'file' is the key name

exports.uploadFile = (req, res) => {
    upload(req, res, (err) => {
        if (err) {
            console.error('[Error] Upload validation failed:', err.message);
            return res.status(400).json({ error: err.message });
        }
        if (!req.file) {
            console.error('[Error] No file provided in request');
            return res.status(400).json({ error: 'Please select a file to upload' });
        }

        // Return file info
        const responseData = {
            message: 'File uploaded successfully',
            filePath: req.file.path,
            filename: req.file.filename,
            originalName: req.file.originalname,
            size: req.file.size
        };
        console.log('[Debug] Upload Success:', responseData);
        res.json(responseData);
    });
};;
