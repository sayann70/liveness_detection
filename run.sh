#!/bin/bash

# AuraSense - Development Runner & Orchestrator
# Color output helpers
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}    AuraSense - Employee Activity Monitoring System   ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. Start/Verify MySQL Server
echo -e "${YELLOW}[1/4] Checking MySQL status...${NC}"
if ! pgrep -x "mysqld" > /dev/null; then
    echo -e "${YELLOW}MySQL not running. Starting via brew services...${NC}"
    brew services start mysql
    sleep 3
else
    echo -e "${GREEN}[✓] MySQL server is already running.${NC}"
fi

# 2. Database Import
echo -e "${YELLOW}[2/4] Applying SQL DDL Schema...${NC}"
/opt/homebrew/bin/mysql -u root -e "CREATE DATABASE IF NOT EXISTS employee_activity_db;"
if [ $? -eq 0 ]; then
    /opt/homebrew/bin/mysql -u root employee_activity_db < schema.sql
    echo -e "${GREEN}[✓] Database schema successfully verified/imported.${NC}"
else
    echo -e "${RED}[✗] Failed to configure MySQL database database connection.${NC}"
fi

# 3. Running Diagnostics Verification
echo -e "${YELLOW}[3/4] Running diagnostic verification...${NC}"
./venv/bin/python validate_setup.py
if [ $? -ne 0 ]; then
    echo -e "${RED}[✗] Diagnostics failed. Please resolve dependencies or database issues.${NC}"
    exit 1
fi

# 4. Launch Server
echo -e "${YELLOW}[4/4] Starting FastAPI Uvicorn Server...${NC}"
echo -e "${GREEN}Dashboard will be available at: http://localhost:8000/${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the monitoring system.${NC}"
echo ""

# Wait a second, and launch dashboard in browser
(sleep 2 && open http://localhost:8000/) &

cd backend
../venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
