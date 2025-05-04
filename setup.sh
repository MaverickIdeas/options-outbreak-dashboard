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

# Install core dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# If torch is missing, install CPU version explicitly (optional redundancy)
if ! python -c "import torch" &> /dev/null; then
    echo "📦 Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Prompt for API key
echo ""
echo "🔑 Please enter your NewsAPI.org API key:"
read -p "API Key: " API_KEY

# Write or update .env file
echo "📝 Writing .env file..."
ENV_FILE=".env"
touch $ENV_FILE

# Remove old key if exists
grep -v "^NEWSAPI_KEY=" $ENV_FILE > temp_env && mv temp_env $ENV_FILE

# Write new key
echo "NEWSAPI_KEY=$API_KEY" >> $ENV_FILE
echo "✅ .env updated with your API key."

# Done
echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the dashboard:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: python app.py"
echo "3. Visit http://127.0.0.1:8050 in your browser"
echo ""

exec "$SHELL"
