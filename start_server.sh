#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Start the Trading Service (FastAPI) using Uvicorn
# We add the current directory to PYTHONPATH to ensure imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Web Monitor (app.py) which will launch Trading Service..."
./.venv/bin/python -u app.py
