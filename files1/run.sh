#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Check if data exists
if [ ! -f "data/newDataset.csv" ]; then
    echo "Error: data/newDataset.csv not found!"
    echo "Please place your dataset in the data/ directory"
    exit 1
fi

# Run the analysis
echo "Starting RNN Malware Detection Analysis..."
python3 rnn_fileless_malware_detection.py

# Deactivate virtual environment
deactivate

echo ""
echo "Analysis complete! Check the outputs/ directory for results."
