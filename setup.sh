#!/bin/bash

echo "🔧 Options Outbreak Dashboard Setup Script"
echo "------------------------------------------"

# Check Python version
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 is not installed. Please install it first."
    exit 1
fi

# Create virtual environment
echo "📁 Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment. Install python3-venv first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

# Install numpy first to ensure compatibility
echo "📦 Installing numpy..."
pip install numpy==1.26.4

# Install core dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Optional torch install (redundancy)
if ! python -c "import torch" &> /dev/null; then
    echo "📦 Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Prompt for NewsAPI Key
echo ""
echo "🔑 Please enter your NewsAPI.org API key:"
read -p "API Key: " NEWS_API_KEY

# Prompt for Polygon.io API Key
echo ""
echo "🔑 Please enter your Polygon.io API key:"
read -p "API Key: " POLYGON_API_KEY

# Ask user about debug mode
echo ""
read -p "🐞 Enable debug mode for development? (y/n): " DEBUG_RESPONSE
if [[ "$DEBUG_RESPONSE" =~ ^[Yy]$ ]]; then
    DEBUG_MODE=true
else
    DEBUG_MODE=false
fi

# Write to .env
echo "📝 Writing .env file..."
ENV_FILE=".env"
touch $ENV_FILE

# Strip old keys if present
grep -v "^NEWSAPI_KEY=" $ENV_FILE | grep -v "^DEBUG_MODE=" | grep -v "^POLYGON_API_KEY=" > temp_env && mv temp_env $ENV_FILE

echo "NEWSAPI_KEY=$NEWS_API_KEY" >> $ENV_FILE
echo "POLYGON_API_KEY=$POLYGON_API_KEY" >> $ENV_FILE
echo "DEBUG_MODE=$DEBUG_MODE" >> $ENV_FILE
echo "✅ .env updated with API keys and debug setting."

# Final instructions
echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the dashboard:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: python app.py"
echo "3. Visit http://127.0.0.1:8050 in your browser"
echo ""

exec "$SHELL"