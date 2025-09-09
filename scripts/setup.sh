#!/bin/bash
echo "Installing project..."

if ! command -v uv &> /dev/null; then
    echo "Install uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv sync --dev
echo "Install finished !"
echo "To activate environment: source .venv/bin/activate"
