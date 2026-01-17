const fs = require('fs');
const path = require('path');

const ONE_HOUR = 60 * 60 * 1000;
const TWENTY_FOUR_HOURS = 24 * 60 * 60 * 1000;
const FILE_RETENTION_TIME = process.env.FILE_RETENTION_HOURS ? 
    (parseInt(process.env.FILE_RETENTION_HOURS) * 60 * 60 * 1000) : 
    TWENTY_FOUR_HOURS;

const setupCleanupSchedule = (uploadDir) => {
    console.log(`[Cleanup] Initialized schedule for ${uploadDir}`);
    console.log(`[Cleanup] File retention time: ${FILE_RETENTION_TIME / (60 * 60 * 1000)} hours`);

    // Run cleanup immediately on start, then every hour
    const runCleanup = () => {
        console.log('[Cleanup] Running scheduled cleanup...');
        fs.readdir(uploadDir, (err, files) => {
            if (err) {
                console.error('[Cleanup] Failed to read directory:', err);
                return;
            }

            const now = Date.now();
            let deletedCount = 0;
            let totalSize = 0;

            files.forEach(file => {
                if (file === '.gitkeep') return; // Skip gitkeep

                const filePath = path.join(uploadDir, file);
                fs.stat(filePath, (err, stats) => {
                    if (err) return;

                    const age = now - stats.mtimeMs;
                    const ageHours = (age / (60 * 60 * 1000)).toFixed(2);

                    if (age > FILE_RETENTION_TIME) {
                        const sizeKB = (stats.size / 1024).toFixed(2);
                        fs.unlink(filePath, (err) => {
                            if (err) {
                                console.error(`[Cleanup] Failed to delete ${file}:`, err);
                            } else {
                                console.log(`[Cleanup] Deleted old file: ${file} (${ageHours}h old, ${sizeKB}KB)`);
                                deletedCount++;
                                totalSize += stats.size;
                            }
                        });
                    }
                });
            });

            // Log summary after a small delay
            setTimeout(() => {
                if (deletedCount > 0) {
                    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
                    console.log(`[Cleanup] Cleanup complete: deleted ${deletedCount} files (${totalSizeMB}MB freed)`);
                }
            }, 1000);
        });
    };

    runCleanup(); // Run once on startup
    setInterval(runCleanup, ONE_HOUR); // Run every hour
};

// Immediate file deletion function (for after processing)
const deleteFileImmediately = (filePath) => {
    return new Promise((resolve, reject) => {
        fs.unlink(filePath, (err) => {
            if (err) {
                console.error(`[Cleanup] Failed to immediately delete ${filePath}:`, err);
                reject(err);
            } else {
                console.log(`[Cleanup] Immediately deleted file: ${filePath}`);
                resolve();
            }
        });
    });
};

module.exports = { setupCleanupSchedule, deleteFileImmediately };
