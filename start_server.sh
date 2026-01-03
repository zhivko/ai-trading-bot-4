#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Start the application
python app.py
