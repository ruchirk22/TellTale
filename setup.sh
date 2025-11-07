#!/bin/bash
# Setup script for M3 MacBook Pro
set -e # Exit on error

echo "Setting up AI Interview Cheating Detection System..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with MPS support (Metal for M3)
echo "Installing PyTorch with Metal support..."
pip install torch torchvision torchaudio

# Install core dependencies from requirements.txt
echo "Installing core dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data (punkt, stopwords, averaged_perceptron_tagger)..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Pre-download Whisper model (base model as specified in .env)
echo "Downloading Whisper 'base' model (this may take a few minutes)..."
python -c "import whisper; whisper.load_model('base')"

# Create necessary directories
echo "Creating directories (data, results, temp, config, models, tests, modules)..."
mkdir -p data/legitimate data/cheating results temp config models tests/ modules/

# Create .env file
echo "Creating .env file..."
cat > .env << 'EOF'
# Environment Configuration
LOG_LEVEL=INFO
PYTORCH_ENABLE_MPS_FALLBACK=1
WHISPER_MODEL=base
OUTPUT_DIR=./results
EOF

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To test the installation and MPS (Metal) support, run:"
echo "python -c 'import torch; print(f\"MPS Available: {torch.backends.mps.is_available()}\")'"