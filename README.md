# TellTale – Interview Integrity Analyzer

TellTale is a local-first pipeline that inspects recorded interview responses for signs of cheating. It combines audio preprocessing, speech-to-text transcription, prosodic analysis, disfluency profiling, and AI-generated text detection to estimate the probability that a response was read or produced by an assistant model.

## Core Capabilities

- **Audio conditioning** – normalizes audio, removes silence, gathers signal health statistics, and (optionally) segments by speaker.
- **Word-aligned transcription** – runs OpenAI Whisper locally for text, word timings, and pause durations.
- **Prosody extraction** – uses openSMILE (eGeMAPSv02) and Praat/parselmouth to measure pitch, intensity, and contour dynamics that often differ between read vs. spontaneous speech.
- **Disfluency scoring** – flags fillers, repetitions, and self-corrections that are common in genuine speech but rare in scripted delivery.
- **AI text heuristics** – estimates how likely the transcript was produced by a language model via GPT-2 perplexity, burstiness, and linguistic markers.
- **Risk aggregation** – fuses the above signals with configurable weights to produce an overall cheating probability and supporting evidence.

## Repository Layout

```bash
Code/
    main.py                 # CLI entrypoint for full analysis runs
    modules/                # Feature-specific components (audio, prosody, NLP, etc.)
    config/                 # YAML configuration & scoring thresholds
    data/                   # Place interview audio here
    models/                 # Storage for finetuned or cached models
    results/                # JSON + text reports generated per run
    temp/                   # Working directory for scratch files & test audio
    tests/                  # Reserved for unit/integration tests
    requirements.txt        # Python dependency lock (install via pip)
    setup.sh                # Bootstrap script for macOS (creates venv, downloads models)
```

## Getting Started

### Prerequisites

- macOS with Python 3.10+ (developed on Apple Silicon/M3 with Metal acceleration)
- Xcode Command Line Tools (`xcode-select --install`)
- ~15 GB free disk space for models and caches (Whisper, GPT-2, spaCy, etc.)

### One-Line Setup

```bash
cd Code
# macOS / Linux
bash setup.sh
# Windows (PowerShell)
# .\setup.ps1
```

The script creates `venv/`, installs dependencies (PyTorch w/ MPS support, Whisper, spaCy, etc.), downloads language models, seeds `.env`, and scaffolds the standard folder structure.

### Manual Environment Setup

If you prefer manual steps:

```bash
cd Code
# Create venv
python -m venv venv
# Activate virtual environment:
# macOS / Linux
source venv/bin/activate
# Windows PowerShell
# .\venv\Scripts\Activate.ps1
# Windows CMD
# .\venv\Scripts\activate.bat

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; [__import__('nltk').download(pkg) for pkg in ['punkt','stopwords','averaged_perceptron_tagger']]"
python -c "import whisper; whisper.load_model('base')"  # caches the model
```

## Running an Analysis

1. Activate the environment:
    - macOS / Linux: `source venv/bin/activate`
    - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
    - Windows CMD: `.\venv\Scripts\activate.bat`
2. Place an audio file (WAV/MP3/M4A) in `temp/` or provide an absolute path.
3. Run the CLI:

    ```bash
    python main.py temp/test_audio.wav --output results/ --model base
    ```

    The run produces a timestamped folder under `results/` containing:
    - `full_analysis_results.json` – detailed metrics from each module
    - `summary_report.txt` – human-readable verdict, evidence, and transcript snapshot

### Module Smoke Tests

Each module exposes a `python -m modules.<name>` script that exercises core functionality with the sample `temp/test_audio.wav` file. Use these when debugging individual stages (audio preprocessing, prosody extraction, transcription, etc.).

## Configuration & Tuning

- `config/config.yaml` – global defaults for sample rate, Whisper model size, pitch bounds, device selection, and logging level.
- `config/thresholds.yaml` – scoring weights, risk boundaries, and trigger thresholds for prosody, pause timing, and disfluency features. Adjust these to calibrate the system after collecting labeled interviews.
- `.env` – runtime defaults (log level, model choice, output directory) created by `setup.sh`.

## Risk Scoring Logic

`InterviewCheatingDetector._calculate_detection_scores` evaluates four major signal groups:

- **Prosody abnormality** – low pitch variation & high monotonicity suggest reading.
- **Pause abnormality** – unnaturally few/regular pauses can indicate scripted delivery.
- **Disfluency absence** – lack of fillers, repetitions, or self-corrections is suspicious.
- **AI content probability** – low perplexity, low burstiness, and formal language raise flags.

`_make_final_assessment` applies weights (default 25/20/25/20/10) to produce a cheating probability and assembles textual evidence for the summary report.

## Data Management

- Keep raw or labeled interview audio under `data/legitimate/` and `data/cheating/` for experimentation.
- Intermediate recordings and scratch artifacts should stay in `temp/` (already git-ignored).
- Model checkpoints or fine-tuned weights belong in `models/` (also ignored by git).

## Development Workflow

- **Linting/formatting** – not enforced yet; consider adding `ruff`/`black`.
- **Testing** – `tests/` is currently empty. When adding unit tests, run them via `pytest` inside the virtual environment.
- **Logging** – defaults to console INFO. To persist logs, set `logging.log_file` in `config.yaml` and update `logging.basicConfig` accordingly.

## Troubleshooting

- **Whisper crashes on MPS** – word timestamps fail on MPS due to float64 support. The transcriber forces CPU execution for reliability. Remove the override only if the upstream bug is resolved.
- **parselmouth Sound errors** – ensure audio arrays are float64 before creating `parselmouth.Sound` objects (already handled in `ProsodyExtractor`).
- **opensmile not found** – on first run, openSMILE may need to download configuration files; verify installation path or reinstall the Python wheel.
- **GPU memory issues** – set `PYTORCH_ENABLE_MPS_FALLBACK=1` (already in `.env`) or run the analysis fully on CPU by editing `config.yaml`.

---

For questions or contributions, open an issue in the repository or reach out to the maintainers noted in `Docs/`.
