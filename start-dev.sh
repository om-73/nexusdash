#!/bin/bash
export PATH=$PATH:/usr/local/bin:/opt/homebrew/bin

# Kill any stale lingering dev servers
lsof -ti:8000,5001,5173 | xargs kill -9 2>/dev/null || true
sleep 1

# Start Python Data Engine
echo "Starting Data Engine..."
cd "data_engine"
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > ../python.log 2>&1 &
cd ..

# Start Node.js Backend
echo "Starting Node Backend..."
cd "server"
nohup node index.js > ../backend.log 2>&1 &
cd ..

# Start React Frontend
echo "Starting React Frontend..."
cd "client"
nohup npm run dev -- --host > ../frontend.log 2>&1 &
cd ..

echo "All services started in detached background."
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:5001"
echo "Data Engine: http://localhost:8000"
echo "Logs are being written to python.log, backend.log, and frontend.log"
