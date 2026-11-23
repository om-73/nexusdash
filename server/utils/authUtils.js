const fs = require('fs');
const path = require('path');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const USERS_FILE = path.join(__dirname, '../models/users.json');
const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-key-change-this';

// In-memory cache to avoid blocking synchronous disk I/O on every request
let cachedUsers = null;

// Password verification cache to avoid expensive bcryptjs compute on repeated logins
const passwordCache = new Map();

// Read users with in-memory caching
exports.getUsers = () => {
    if (cachedUsers) return cachedUsers;
    try {
        if (!fs.existsSync(USERS_FILE)) {
            cachedUsers = [];
            return cachedUsers;
        }
        const data = fs.readFileSync(USERS_FILE, 'utf8');
        cachedUsers = JSON.parse(data);
        return cachedUsers;
    } catch (err) {
        return [];
    }
};

// Write users and invalidate/update cache
exports.saveUsers = (users) => {
    cachedUsers = users;
    fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));
};

// Find user by email
exports.findUserByEmail = (email) => {
    const users = exports.getUsers();
    return users.find(u => u.email === email);
};

// Create user
exports.createUser = async (email, password, role = 'viewer') => {
    const users = exports.getUsers();
    if (users.find(u => u.email === email)) {
        throw new Error('User already exists');
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = {
        id: Date.now().toString(),
        email,
        password: hashedPassword,
        role,
        createdAt: new Date().toISOString()
    };

    users.push(newUser);
    exports.saveUsers(users);
    // Warm password cache for the new user
    passwordCache.set(`${password}:${hashedPassword}`, true);
    return { id: newUser.id, email: newUser.email, role: newUser.role };
};

// Verify Password with fast cache lookup
exports.verifyPassword = async (password, hash) => {
    const cacheKey = `${password}:${hash}`;
    if (passwordCache.has(cacheKey)) {
        return passwordCache.get(cacheKey);
    }
    const isValid = await bcrypt.compare(password, hash);
    if (passwordCache.size > 200) {
        // Simple prune
        const firstKey = passwordCache.keys().next().value;
        passwordCache.delete(firstKey);
    }
    passwordCache.set(cacheKey, isValid);
    return isValid;
};

// Generate Token
exports.generateToken = (user) => {
    return jwt.sign(
        { id: user.id, email: user.email, role: user.role },
        JWT_SECRET,
        { expiresIn: '24h' }
    );
};

// Initialize Default User (For Demo/Ephemeral Env)
exports.initializeDefaultUser = async () => {
    try {
        const users = exports.getUsers();
        const adminEmail = 'admin@nexusdash.com';

        if (!users.find(u => u.email === adminEmail)) {
            console.log('[Auth] Seeding default admin user...');
            await exports.createUser(adminEmail, 'admin123', 'admin');
            console.log('[Auth] Default admin created: admin@nexusdash.com / admin123');
        }
    } catch (err) {
        console.error('[Auth] Failed to seed default user:', err);
    }
};
