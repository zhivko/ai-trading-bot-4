#!/bin/bash

# Configuration
SERVICE_NAME="ai-trading-bot-4"

echo "Restarting Systemd Service: $SERVICE_NAME..."

# Restart the service
sudo systemctl restart $SERVICE_NAME

if [ $? -eq 0 ]; then
    echo "Service restarted successfully!"
    echo "Showing service status..."
    sudo systemctl status $SERVICE_NAME --no-pager
else
    echo "Failed to restart service."
    exit 1
fi
