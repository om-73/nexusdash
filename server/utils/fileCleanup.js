const fs = require('fs');
const path = require('path');

const ONE_HOUR = 60 * 60 * 1000;
const TWENTY_FOUR_HOURS = 24 * 60 * 60 * 1000;

const setupCleanupSchedule = (uploadDir) => {
    console.log(`[Cleanup] Initialized schedule for ${uploadDir}`);

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

            files.forEach(file => {
                if (file === '.gitkeep') return; // Skip gitkeep

                const filePath = path.join(uploadDir, file);
                fs.stat(filePath, (err, stats) => {
                    if (err) return;

                    if (now - stats.mtimeMs > TWENTY_FOUR_HOURS) {
                        fs.unlink(filePath, (err) => {
                            if (err) console.error(`[Cleanup] Failed to delete ${file}:`, err);
                            else {
                                console.log(`[Cleanup] Deleted old file: ${file}`);
                                deletedCount++;
                            }
                        });
                    }
                });
            });
        });
    };

    runCleanup(); // Run once on startup
    setInterval(runCleanup, ONE_HOUR);
};

module.exports = { setupCleanupSchedule };
