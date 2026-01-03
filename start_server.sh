#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Start the application using the virtual environment directly
./.venv/bin/python app.py
