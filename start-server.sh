#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Change to the script's directory so all relative paths are correct
cd "$SCRIPT_DIR" || exit

# Activate the virtual environment if it exists in the script's directory
if [ -d "venv" ]; then
    echo "Activating virtual environment from $SCRIPT_DIR/venv..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment 'venv' not found in $SCRIPT_DIR."
fi

# Start the FastAPI server
echo "Starting BETSY agent server from $SCRIPT_DIR..."
python main.py