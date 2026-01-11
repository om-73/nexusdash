#!/bin/bash

# Start Python Data Engine
echo "Starting Data Engine..."
cd "data_engine"
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../python.log 2>&1 &
PID_PYTHON=$!
cd ..

# Start Node.js Backend
echo "Starting Node Backend..."
cd "server"
node index.js > ../backend.log 2>&1 &
PID_NODE=$!
cd ..

# Start React Frontend
echo "Starting React Frontend..."
cd "client"
npm run dev -- --host > ../frontend.log 2>&1 &
PID_REACT=$!
cd ..

echo "All services started."
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:5001"
echo "Data Engine: http://localhost:8000"
echo "Logs are being written to python.log, backend.log, and frontend.log"

# Trap SIGINT to kill all background processes
trap "kill $PID_PYTHON $PID_NODE $PID_REACT; exit" SIGINT

wait
