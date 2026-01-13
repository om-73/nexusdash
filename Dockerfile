# Base image: Python 3.11 (stable, good for AI libs)
FROM python:3.11-slim

# Install system dependencies (curl for Node setup, build-essential for some python libs)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (Version 20.x)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# --- Python Setup ---
# Copy Python requirements first for caching
COPY data_engine/requirements.txt ./data_engine/requirements.txt

# Install Python dependencies
WORKDIR /app/data_engine
RUN pip install --no-cache-dir -r requirements.txt

# --- Node Setup ---
# Go back to root
WORKDIR /app

# Copy root package.json for npm install
COPY package.json ./

# Copy server package files (if any specific, but we used root)
COPY client/package.json ./client/
# (Server deps are in root package.json now, so just installing root is enough for server)

# Install Node dependencies
RUN npm install

# --- Application Code ---
# Copy the rest of the application
COPY . .

# Build Frontend
WORKDIR /app/client
RUN npm install
RUN npm run build

# Go back to root
WORKDIR /app

# Ensure uploads directory exists and is writable
RUN mkdir -p server/uploads && chmod 777 server/uploads

# Expose ports (Render only listens on one, usually PORT env var, but we expose internal 8000 too)
EXPOSE 5001
EXPOSE 8000

# Environment variables
ENV NODE_ENV=production
ENV PYTHON_ENGINE_URL=http://127.0.0.1:8000

# Start command (Run both services)
# We use the script we defined in package.json
CMD ["npm", "start"]
