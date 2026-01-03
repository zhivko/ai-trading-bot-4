#!/bin/bash

# Configuration
APP_DIR="/opt/ai-trading-bot"
REPO_URL="https://github.com/zhivko/ai-trading-bot-4" # User: Replace this with your actual Git repo URL
SERVICE_NAME="ai-trading-bot-4"
USER_NAME=$USER

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Setup...${NC}"

# 1. System Dependencies
echo -e "${GREEN}[1/5] Installing System Dependencies...${NC}"
sudo apt update
sudo apt install -y python3-pip python3-venv git dos2unix

# 2. Clone/Update Repository
echo -e "${GREEN}[2/5] Setting up Repository...${NC}"

# Check if directory exists
if [ -d "$APP_DIR" ]; then
    echo "Directory exists. Pulling latest changes..."
    cd "$APP_DIR"
    sudo git pull
    # Fix ownership in case sudo git pull changed it
    sudo chown -R "$USER_NAME:$USER_NAME" "$APP_DIR"
else
    echo "Cloning repository..."
    if [ "$REPO_URL" == "YOUR_REPO_URL_HERE" ]; then
        read -p "Enter your Git Repository URL: " INPUT_REPO_URL
        if [ -z "$INPUT_REPO_URL" ]; then
            echo "Error: Repository URL is required."
            exit 1
        fi
        REPO_URL="$INPUT_REPO_URL"
    fi
    sudo git clone "$REPO_URL" "$APP_DIR"
    sudo chown -R "$USER_NAME:$USER_NAME" "$APP_DIR"
    cd "$APP_DIR"
fi

# 3. Python Virtual Environment
echo -e "${GREEN}[3/5] Setting up Virtual Environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Install dependencies
echo "Installing Python dependencies..."
./.venv/bin/pip install --upgrade pip
# Install CPU-only torch to save resources on server, unless GPU is required
./.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/pip install -r requirements.txt

# 4. Environment Variables
echo -e "${GREEN}[4/5] Checking Configuration...${NC}"
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    # You might want to provide a .env.example file in your repo
    echo "Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        touch .env
    fi
    echo "Please edit $APP_DIR/.env with your secrets."
fi

# Make start script executable
# Convert scripts to Unix line endings
dos2unix start_server.sh
chmod +x start_server.sh

# 5. Systemd Service
echo -e "${GREEN}[5/5] Configuring Systemd Service...${NC}"

SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

sudo bash -c "cat > $SERVICE_FILE" <<EOL
[Unit]
Description=AI Trading Bot Service
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$APP_DIR
Environment="PYTHONPATH=$APP_DIR"
ExecStart=$APP_DIR/.venv/bin/uvicorn apex_integration.trading_service:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload
echo "Enabling service..."
sudo systemctl enable $SERVICE_NAME

echo -e "${GREEN}Setup Complete!${NC}"
echo "To start the service, run: sudo systemctl start $SERVICE_NAME"
echo "To check status, run: sudo systemctl status $SERVICE_NAME"
echo "To view logs, run: journalctl -u $SERVICE_NAME -f"
