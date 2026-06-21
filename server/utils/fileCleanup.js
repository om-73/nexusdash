const fs = require('fs');
const path = require('path');
const fsp = fs.promises;

const ONE_HOUR = 60 * 60 * 1000;
const TWENTY_FOUR_HOURS = 24 * 60 * 60 * 1000;
const FILE_RETENTION_TIME = process.env.FILE_RETENTION_HOURS ? 
    (parseInt(process.env.FILE_RETENTION_HOURS) * 60 * 60 * 1000) : 
    TWENTY_FOUR_HOURS;

/**
 * Setup recurring cleanup task for upload directory using fs.promises
 * @param {string} uploadDir - Path to upload directory
 */
const setupCleanupSchedule = (uploadDir) => {
    console.log(`[Cleanup] Initialized schedule for ${uploadDir}`);
    console.log(`[Cleanup] File retention time: ${FILE_RETENTION_TIME / (60 * 60 * 1000)} hours`);

    const runCleanup = async () => {
        console.log('[Cleanup] Running scheduled cleanup...');
        try {
            if (!fs.existsSync(uploadDir)) return;
            const files = await fsp.readdir(uploadDir);
            const now = Date.now();
            let deletedCount = 0;
            let totalSize = 0;

            for (const file of files) {
                if (file === '.gitkeep') continue;

                const filePath = path.join(uploadDir, file);
                try {
                    const stats = await fsp.stat(filePath);
                    const age = now - stats.mtimeMs;
                    const ageHours = (age / (60 * 60 * 1000)).toFixed(2);

                    if (age > FILE_RETENTION_TIME) {
                        const sizeKB = (stats.size / 1024).toFixed(2);
                        await fsp.unlink(filePath);
                        console.log(`[Cleanup] Deleted old file: ${file} (${ageHours}h old, ${sizeKB}KB)`);
                        deletedCount++;
                        totalSize += stats.size;
                    }
                } catch (err) {
                    console.error(`[Cleanup] Failed to process file ${file}:`, err.message);
                }
            }

            if (deletedCount > 0) {
                const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
                console.log(`[Cleanup] Cleanup complete: deleted ${deletedCount} files (${totalSizeMB}MB freed)`);
            }
        } catch (err) {
            console.error('[Cleanup] Scheduled cleanup failed:', err);
        }
    };

    runCleanup(); // Run once on startup
    setInterval(runCleanup, ONE_HOUR); // Run every hour
};

/**
 * Immediate file deletion (e.g. for user cleanups)
 * @param {string} filePath - Absolute path to target file
 */
const deleteFileImmediately = async (filePath) => {
    try {
        await fsp.unlink(filePath);
        console.log(`[Cleanup] Immediately deleted file: ${filePath}`);
    } catch (err) {
        console.error(`[Cleanup] Failed to immediately delete ${filePath}:`, err.message);
        throw err;
    }
};

module.exports = { setupCleanupSchedule, deleteFileImmediately };
