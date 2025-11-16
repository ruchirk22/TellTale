import whisper
import torch
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    Handles audio transcription using the local OpenAI Whisper model,
    optimized for M3 MPS (Metal).
    """

    def __init__(self, model_size: str = "base", language: str = "en"):
        """
        Initializes the WhisperTranscriber.

        Args:
            model_size: The size of the Whisper model (e.g., "tiny", "base", "small").
            language: The language for transcription.
        """
        self.model_size = model_size
        self.language = language
        # self.device = self._get_device() # Buggy with word timestamps on MPS

        # --- FIX for MPS float64 error ---
        # The whisper `add_word_timestamps` function has a known bug on MPS
        # where it tries to cast a tensor to float64, which MPS doesn't support.
        # Forcing CPU ensures word timestamping works, even if transcription is slightly slower.
        self.device = "cpu"
        # --- End of FIX ---
        
        logger.info(f"Loading Whisper model '{model_size}' onto device '{self.device}'...")
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info("Whisper model loaded successfully.")

    def _get_device(self) -> str:
        """Detects and returns the optimal device (MPS, CUDA, or CPU)."""
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal) device found.")
            return "mps"
        if torch.cuda.is_available():
            logger.info("CUDA device found.")
            return "cuda"
        logger.info("No GPU found, using CPU.")
        return "cpu"

    def transcribe(self, audio_path: str, word_timestamps: bool = True) -> Dict[str, Any]:
        """
        Transcribes the given audio file.

        Args:
            audio_path: Path to the audio file.
            word_timestamps: Whether to include word-level timestamps.

        Returns:
            The full transcription result dictionary from Whisper.
        """
        try:
            logger.info(f"Starting transcription for: {audio_path}")
            # The 'fp16=False' is important for MPS compatibility with some models
            transcribe_options = {
                "word_timestamps": word_timestamps,
                "language": self.language,
                # "fp16": False if self.device == "mps" else True # Not needed for CPU
            }
            
            result = self.model.transcribe(audio_path, **transcribe_options)
            
            logger.info("Transcription complete.")
            return result
        
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            raise

    def get_words_with_pauses(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a Whisper result to create a list of words, calculating
        the pause duration *before* each word.

        Returns:
            A list of word objects, e.g.:
            [{'word': 'Hello', 'start': 0.5, 'end': 0.8, 'pause_before': 0.2}, ...]
        """
        words_with_pauses = []
        last_word_end = 0.0

        for segment in result.get("segments", []):
            if "words" not in segment:
                logger.warning("No 'words' in segment, skipping pause calculation for it.")
                continue

            for word_info in segment["words"]:
                word = word_info["word"]
                start = word_info["start"]
                end = word_info["end"]
                
                # Calculate pause duration before this word
                pause_before = round(start - last_word_end, 3)
                
                words_with_pauses.append({
                    "word": word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "pause_before": pause_before
                })
                
                last_word_end = end
        
        logger.info(f"Processed {len(words_with_pauses)} words with pauses.")
        return words_with_pauses

    def get_transcript_text(self, result: Dict[str, Any]) -> str:
        """Returns the full plain text from a transcription result."""
        return result.get("text", "").strip()


class TranscriptAnalyzer:
    """
    Analyzes transcription results to extract features like
    speaking rate and pause statistics.
    """

    @staticmethod
    def calculate_speaking_rate(result: Dict[str, Any]) -> Tuple[float, List[float]]:
        """
        Calculates the overall speaking rate in Words Per Minute (WPM).

        Args:
            result: The Whisper transcription result.

        Returns:
            A tuple of (overall_wpm, segment_wpm_list).
        """
        full_text = result.get("text", "")
        if not full_text:
            return 0.0, []

        total_words = len(full_text.split())
        total_duration_minutes = 0.0
        
        segment_wpm_list = []
        
        segments = result.get("segments", [])
        if not segments:
            return 0.0, []
            
        # Calculate total duration from segments
        if segments:
            total_duration_minutes = (segments[-1]["end"] - segments[0]["start"]) / 60.0

        # Calculate WPM for each segment
        for segment in segments:
            segment_text = segment.get("text", "")
            segment_words = len(segment_text.split())
            segment_duration_min = (segment["end"] - segment["start"]) / 60.0
            
            if segment_duration_min > 0:
                segment_wpm = segment_words / segment_duration_min
                segment_wpm_list.append(segment_wpm)

        overall_wpm = 0.0
        if total_duration_minutes > 0:
            overall_wpm = total_words / total_duration_minutes

        logger.info(f"Speaking Rate: {overall_wpm:.2f} WPM (Overall)")
        return overall_wpm, segment_wpm_list

    @staticmethod
    def get_pause_statistics(words_with_pauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates statistics on pause durations.

        Micropause: 0.1s - 0.5s (natural word boundaries)
        Macropause: > 0.5s (thinking, sentence boundaries)

        Args:
            words_with_pauses: The list generated by get_words_with_pauses.

        Returns:
            A dictionary of pause statistics.
        """
        pauses = [word["pause_before"] for word in words_with_pauses if word["pause_before"] > 0.01]
        
        if not pauses:
            return {
                "mean_pause": 0, "median_pause": 0, "std_pause": 0,
                "max_pause": 0, "micropause_count": 0, "macropause_count": 0,
                "micropause_ratio": 0
            }

        micropauses = [p for p in pauses if 0.1 <= p <= 0.5]
        macropauses = [p for p in pauses if p > 0.5]

        total_pauses = len(pauses)
        stats = {
            "mean_pause": round(np.mean(pauses), 3),
            "median_pause": round(np.median(pauses), 3),
            "std_pause": round(np.std(pauses), 3),
            "max_pause": round(np.max(pauses), 3),
            "micropause_count": len(micropauses),
            "macropause_count": len(macropauses),
            "micropause_ratio": len(micropauses) / total_pauses if total_pauses > 0 else 0
        }
        
        logger.info(f"Pause Stats: {stats['micropause_count']} micro, "
                    f"{stats['macropause_count']} macro.")
        return stats

if __name__ == "__main__":
    """
    Provides a simple test run for this module.
    Uses the same test audio file as audio_processor.
    
     Instructions:
     1. Activate your virtual environment:
         - macOS / Linux: `source venv/bin/activate`
         - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
         - Windows CMD: `.\venv\Scripts\activate.bat`
     2. Make sure your 'temp/test_audio.wav' file exists.
     3. Run this module from the 'TellTale/Code/' directory:
         `python -m modules.transcription`
    """
    logger.info("Running Transcription module test...")
    
    test_audio_path = "temp/test_audio.wav"

    try:
        # 1. Initialize Transcriber
        # We use 'base' as it was downloaded by our setup script.
        transcriber = WhisperTranscriber(model_size="base")

        # 2. Transcribe Audio
        result = transcriber.transcribe(test_audio_path)
        
        full_text = transcriber.get_transcript_text(result)
        logger.info(f"--- Full Transcript ---\n{full_text}\n-------------------------")

        # 3. Get Words with Pauses
        words = transcriber.get_words_with_pauses(result)
        if words:
            logger.info(f"First 5 words: {words[:5]}")
        
        # 4. Analyze Transcript
        analyzer = TranscriptAnalyzer()
        
        wpm, _ = analyzer.calculate_speaking_rate(result)
        logger.info(f"Overall WPM: {wpm:.2f}")
        
        pause_stats = analyzer.get_pause_statistics(words)
        logger.info(f"Pause Statistics: {pause_stats}")
        
        logger.info("Transcription module test complete.")

    except FileNotFoundError:
        logger.error(f"Test file not found at '{test_audio_path}'.")
        logger.error("Please ensure 'temp/test_audio.wav' exists.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)