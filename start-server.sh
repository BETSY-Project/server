#!/bin/bash

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the FastAPI server
echo "Starting BETSY agent server..."
python api.py