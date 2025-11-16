# Setup script for Windows
# This script sets up the AI Interview Cheating Detection System on Windows

Write-Host "Setting up AI Interview Cheating Detection System..."

# Check Python version
try {
    $python_version = python --version 2>&1
    Write-Host "Python version: $python_version"
}
catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment 'venv'..."
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU or CUDA)
Write-Host "Installing PyTorch..."
Write-Host "By default, this installs CPU version. For GPU support:"
Write-Host "  - CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
Write-Host "  - Continuing with default (CPU)..."
pip install torch torchvision torchaudio

# Install core dependencies from requirements.txt
Write-Host "Installing core dependencies..."
pip install -r requirements.txt

# Ensure Visual C++ Build Tools (needed for some packages like webrtcvad)
Write-Host "Checking for Microsoft Visual C++ Build Tools (required for some native wheels)..."
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($null -eq $cl) {
    Write-Host "Microsoft Visual C++ Build Tools not found on PATH." -ForegroundColor Yellow
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($null -ne $winget) {
        Write-Host "winget detected. You can install the Build Tools using winget (requires admin)."
        $install = Read-Host "Install Visual C++ Build Tools now? [Y/n]"
        if ($install -eq "Y" -or $install -eq "y" -or $install -eq "") {
            Write-Host "Attempting to install Visual C++ Build Tools via winget..."
            Write-Host "This may prompt for elevation."
            winget install --id Microsoft.VisualStudio.2022.BuildTools -e
            Write-Host "If the installation failed or you declined elevation, please install the 'Build Tools for Visual Studio' manually:" -ForegroundColor Yellow
            Write-Host "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        } else {
            Write-Host "Skipping automatic installation. If you encounter build errors (e.g., for 'webrtcvad'), install Build Tools manually:" -ForegroundColor Yellow
            Write-Host "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        }
    } else {
        Write-Host "winget not found. To build native extensions you must install the Visual C++ Build Tools manually:" -ForegroundColor Yellow
        Write-Host "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        Write-Host "Alternatively, use Miniconda/Anaconda and install troublesome packages from conda-forge to avoid compilation." -ForegroundColor Yellow
    }
} else {
    Write-Host "Visual C++ Build Tools detected (cl.exe found)."
}

# Download spaCy model
Write-Host "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Download NLTK data
Write-Host "Downloading NLTK data (punkt, stopwords, averaged_perceptron_tagger)..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Pre-download Whisper model (base model as specified in .env)
Write-Host "Downloading Whisper 'base' model (this may take a few minutes)..."
python -c "import whisper; whisper.load_model('base')"

# Create necessary directories
Write-Host "Creating directories (data, results, temp, config, models, tests, modules)..."
$directories = @("data/legitimate", "data/cheating", "results", "temp", "config", "models", "tests", "modules")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Create .env file
Write-Host "Creating .env file..."
@"
# Environment Configuration
LOG_LEVEL=INFO
PYTORCH_ENABLE_MPS_FALLBACK=0
WHISPER_MODEL=base
OUTPUT_DIR=./results
"@ | Out-File -FilePath .env -Encoding UTF8

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future, run:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To test the PyTorch installation, run:"
Write-Host "  python -c 'import torch; print(f""PyTorch Version: {torch.__version__}""); print(f""CUDA Available: {torch.cuda.is_available()}"")'"
Write-Host ""
Write-Host "To run the main analysis, use:"
Write-Host "  python main.py <audio_file> --output <output_dir>"
