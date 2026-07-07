#!/bin/bash
set -e

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example -- fill in your PROJECT_ID before running."
else
    echo ".env already exists, skipping."
fi

mkdir -p test_input test_output

echo "Setup complete. Run: source venv/bin/activate"
