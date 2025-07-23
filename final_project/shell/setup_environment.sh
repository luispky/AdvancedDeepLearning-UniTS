#!/bin/bash

# Exit on any error
set -e

echo "=== Project Environment Setup ==="

# Go to project root (assumed one level up from script)
cd "$(dirname "$0")/.."
echo "Project root: $(pwd)"

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found in project root!"
    exit 1
fi

echo "✅ Found pyproject.toml"

# Check if uv is already installed
if ! command -v uv &>/dev/null; then
    echo "📦 uv not found, installing..."
    # Install uv and clean up the installer script
    curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh
    sh install_uv.sh
    rm install_uv.sh
    echo "✅ uv installed successfully!"
else
    echo "✅ uv is already installed ($(uv --version))"
fi

# Install dependencies and create virtual environment
echo "📦 Installing dependencies with uv..."
if uv sync --no-dev; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    echo "💡 This is just an environment setup, not a package"
    exit 1
fi

# Check if virtual environment was created
if [ -f ".venv/bin/activate" ]; then
    echo "✅ Virtual environment created at: $(pwd)/.venv"
    echo ""
    echo "🎉 Environment setup complete!"
    echo ""
    echo "To activate the environment, run:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "You can now run project scripts!"
else
    echo "❌ Virtual environment was not created properly"
    echo "💡 Check the output above for errors"
    exit 1
fi
