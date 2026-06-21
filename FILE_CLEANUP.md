# Automatic File Cleanup Configuration

## Overview
The application automatically deletes uploaded CSV/Excel files after a specified retention period to save disk space and maintain server cleanliness.

## Features

### 1. **Automatic Scheduled Cleanup**
- Runs automatically every **1 hour**
- Deletes files older than the retention period (default: 24 hours)
- Logs deleted files with size information

### 2. **Configurable Retention Time**
- Set via environment variable: `FILE_RETENTION_HOURS`
- Default: 24 hours
- Can be modified in `server/.env`

### 3. **Manual Cleanup Endpoint**
- Endpoint: `POST /api/cleanup`
- Allows immediate cleanup on demand
- Returns detailed cleanup statistics

## Configuration

### Setting Retention Time

**File:** `server/.env`

```env
FILE_RETENTION_HOURS=24
```

**Options:**
- `FILE_RETENTION_HOURS=1` → Delete files older than 1 hour
- `FILE_RETENTION_HOURS=24` → Delete files older than 1 day (default)
- `FILE_RETENTION_HOURS=72` → Delete files older than 3 days
- `FILE_RETENTION_HOURS=168` → Delete files older than 1 week

## Usage

### Automatic Cleanup (No Action Required)
Files are automatically cleaned up based on the configured retention time. Logs show:
```
[Cleanup] Deleted old file: file-1234567890-123456789.csv (25.5h old, 245.67KB)
[Cleanup] Cleanup complete: deleted 3 files (0.72MB freed)
```

### Manual Cleanup (On Demand)

**Using cURL:**
```bash
curl -X POST http://localhost:5001/api/cleanup
```

**Response Example:**
```json
{
  "message": "Cleanup completed",
  "deletedCount": 3,
  "deletedFiles": [
    "file-1768642357228-989258383.csv",
    "file-1768642335194-33421504.csv",
    "file-1768642329761-240159335.csv"
  ],
  "freedSpaceMB": "0.72"
}
```

## File Storage Location
- **Upload Directory:** `server/uploads/`
- **Protected Files:** `.gitkeep` (never deleted)

## Logging

### Server Logs
Check `backend.log` for cleanup activity:
```bash
tail -f backend.log | grep Cleanup
```

### Cleanup Log Examples
```
[Cleanup] Initialized schedule for /path/to/uploads
[Cleanup] File retention time: 24 hours
[Cleanup] Running scheduled cleanup...
[Cleanup] Deleted old file: file-xxx-xxx.csv (25.5h old, 245.67KB)
[Cleanup] Cleanup complete: deleted 1 files (0.24MB freed)
```

## Advanced Options

### Immediate File Deletion After Processing
If you want files deleted immediately after they're processed:

1. Edit `server/controllers/dataController.js`
2. Add this import:
   ```javascript
   const { deleteFileImmediately } = require('../utils/fileCleanup');
   ```

3. In the `processUpload` function, after successful processing:
   ```javascript
   // After successful data load
   await deleteFileImmediately(file_path);
   console.log('File deleted immediately after processing');
   ```

## Best Practices

1. **Development:** Set `FILE_RETENTION_HOURS=1` for quick cleanup during testing
2. **Production:** Set `FILE_RETENTION_HOURS=72` or higher for safer retention
3. **Monitor:** Check logs regularly to understand cleanup patterns
4. **Manual Cleanup:** Run `POST /api/cleanup` before deployments to free space

## Troubleshooting

### Files Not Being Deleted
- Check if files are being accessed by the Python engine
- Verify `FILE_RETENTION_HOURS` is set correctly in `.env`
- Check server logs: `tail -100 backend.log | grep Cleanup`

### Permission Denied Errors
- Ensure Node process has write permissions to `server/uploads/`
- Fix permissions: `chmod 755 server/uploads`

### Cleanup Not Running
- Verify the cleanup schedule initialized: Check logs for `[Cleanup] Initialized schedule`
- Restart the server to reinitialize cleanup
- Check if any processes are still accessing the files

## Related Files
- `server/utils/fileCleanup.js` - Cleanup logic
- `server/index.js` - Cleanup initialization and endpoint
- `server/.env` - Configuration
