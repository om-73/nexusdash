const fs = require('fs');
const path = require('path');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const USERS_FILE = path.join(__dirname, '../models/users.json');
const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-key-change-this';

// Read users
exports.getUsers = () => {
    try {
        if (!fs.existsSync(USERS_FILE)) return [];
        const data = fs.readFileSync(USERS_FILE, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        return [];
    }
};

// Write users
exports.saveUsers = (users) => {
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
    return { id: newUser.id, email: newUser.email, role: newUser.role };
};

// Verify Password
exports.verifyPassword = async (password, hash) => {
    return await bcrypt.compare(password, hash);
};

// Generate Token
exports.generateToken = (user) => {
    return jwt.sign(
        { id: user.id, email: user.email, role: user.role },
        JWT_SECRET,
        { expiresIn: '24h' }
    );
};
