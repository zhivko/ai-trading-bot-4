#!/bin/bash

# Configuration
SERVICE_NAME="ai-trading-bot-4"

echo "Showing logs for service: $SERVICE_NAME"
echo "Press Ctrl+C to exit log view."
echo "----------------------------------------"

# Show logs following (-f)
journalctl -u $SERVICE_NAME -f
