#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Start the Trading Service (FastAPI) using Uvicorn
# We add the current directory to PYTHONPATH to ensure imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Apex Trading Service on port 8000..."
./.venv/bin/uvicorn apex_integration.trading_service:app --host 0.0.0.0 --port 8000
