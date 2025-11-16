import librosa
import numpy as np
import webrtcvad
import logging
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles loading, preprocessing, and Voice Activity Detection (VAD)
    for audio files to prepare them for analysis.
    """

    def __init__(self, target_sr: int = 16000, mono: bool = True):
        """
        Initializes the AudioProcessor.

        Args:
            target_sr: The target sample rate. 16000 Hz is required for webrtcvad.
            mono: Whether to convert the audio to mono.
        """
        self.target_sr = target_sr
        self.mono = mono
        if target_sr not in (8000, 16000, 32000, 48000):
            logger.error(f"Invalid sample rate for VAD: {target_sr}. Must be 8, 16, 32, or 48 kHz.")
            raise ValueError("Invalid sample rate for VAD.")
        
        # Initialize VAD with aggressiveness level 3 (most aggressive)
        self.vad = webrtcvad.Vad(3)
        logger.info(f"AudioProcessor initialized with SR={target_sr} Hz.")

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int, float]:
        """
        Loads an audio file, converts it to the target sample rate and mono.

        Args:
            audio_path: Path to the audio file (WAV, MP3, M4A, etc.).

        Returns:
            A tuple of (audio_signal, sample_rate, duration_in_seconds).
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=self.mono)
            duration = librosa.get_duration(y=audio, sr=sr)
            logger.info(f"Loaded audio: {audio_path} ({duration:.2f}s, {sr}Hz)")
            return audio, sr, duration
        
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}", exc_info=True)
            raise

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalizes audio amplitude to the range [-1, 1].

        Args:
            audio: The audio signal as a numpy array.

        Returns:
            The normalized audio signal.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalized_audio = audio / max_val
            return normalized_audio
        logger.warning("Audio file is silent, returning as is.")
        return audio

    def _get_speech_segments(self, audio: np.ndarray, sr: int, 
                             frame_duration_ms: int = 30, 
                             padding_duration_ms: int = 300) -> List[Tuple[float, float]]:
        """
        Helper function to get (start, end) timestamps of speech segments.
        """
        if sr != self.target_sr:
            raise ValueError(f"Audio SR ({sr}) doesn't match VAD SR ({self.target_sr})")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms")

        # Convert audio to 16-bit PCM bytes
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Calculate frame size and padding
        frame_size_samples = int(sr * frame_duration_ms / 1000)
        padding_frames = int(padding_duration_ms / frame_duration_ms)
        
        num_frames = len(audio_int16) // frame_size_samples
        if num_frames == 0:
            logger.warning("Audio is too short for VAD.")
            return []

        # Get VAD output for each frame
        speech_frames = []
        for i in range(num_frames):
            start = i * frame_size_samples
            end = start + frame_size_samples
            frame_bytes = audio_int16[start:end].tobytes()
            
            # Ensure frame is correct length (webrtcvad is strict)
            if len(frame_bytes) != frame_size_samples * 2: # 2 bytes per int16
                break 
                
            try:
                is_speech = self.vad.is_speech(frame_bytes, sr)
                speech_frames.append(is_speech)
            except Exception as e:
                logger.warning(f"VAD failed for frame {i}: {e}")
                speech_frames.append(False)

        # Add padding
        padded_speech_frames = speech_frames.copy()
        for i in range(len(speech_frames)):
            if speech_frames[i]:
                start_pad = max(0, i - padding_frames)
                end_pad = min(len(speech_frames), i + padding_frames + 1)
                for j in range(start_pad, end_pad):
                    padded_speech_frames[j] = True

        # Convert frames to time segments
        segments = []
        in_speech = False
        start_time = 0.0
        frame_duration_sec = frame_duration_ms / 1000.0

        for i, is_speech in enumerate(padded_speech_frames):
            if is_speech and not in_speech:
                start_time = i * frame_duration_sec
                in_speech = True
            elif not is_speech and in_speech:
                end_time = i * frame_duration_sec
                segments.append((start_time, end_time))
                in_speech = False
        
        # If still in speech at the end
        if in_speech:
            end_time = len(padded_speech_frames) * frame_duration_sec
            segments.append((start_time, end_time))

        return segments

    def remove_silence(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Removes silence from an audio signal using VAD.

        Args:
            audio: The audio signal.
            sr: The sample rate.

        Returns:
            A tuple of (audio_with_silence_removed, speech_segments_timestamps).
        """
        speech_segments = self._get_speech_segments(audio, sr)
        
        if not speech_segments:
            logger.warning("No speech detected by VAD.")
            return np.array([]), []

        # Concatenate audio from speech segments
        audio_no_silence = np.concatenate([
            audio[int(start * sr):int(end * sr)]
            for start, end in speech_segments
        ])
        
        logger.info(f"Removed silence. Original duration: {len(audio)/sr:.2f}s, "
                    f"New duration: {len(audio_no_silence)/sr:.2f}s")
        return audio_no_silence, speech_segments

    def separate_speakers(self, audio: np.ndarray, sr: int) -> Tuple[List, List]:
        """
        Placeholder for speaker separation.
        In this basic implementation, we assume all detected speech
        belongs to the candidate.
        
        A full implementation would use pyannote.audio here.

        Args:
            audio: The audio signal.
            sr: The sample rate.

        Returns:
            A tuple of (interviewer_segments, candidate_segments).
            For now, interviewer_segments is empty.
        """
        logger.warning("Using placeholder speaker separation. "
                       "Treating all speech as 'candidate'.")
        
        # Get speech segments
        _, speech_segments_ts = self.remove_silence(audio, sr)
        
        candidate_segments = []
        for start, end in speech_segments_ts:
            segment_audio = audio[int(start * sr):int(end * sr)]
            candidate_segments.append({
                "start": start,
                "end": end,
                "audio": segment_audio
            })
            
        interviewer_segments = []
        return interviewer_segments, candidate_segments

    def extract_environment_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Extracts non-speech segments (silence, background noise) from audio.

        Args:
            audio: The audio signal.
            sr: The sample rate.

        Returns:
            A list of numpy arrays, each containing a segment of background noise.
        """
        speech_segments = self._get_speech_segments(audio, sr)
        
        if not speech_segments:
            logger.info("No speech detected, returning full audio as environment audio.")
            return [audio]

        background_segments = []
        last_end_time = 0.0

        for start, end in speech_segments:
            gap_duration = start - last_end_time
            if gap_duration > 0.1: # Only capture gaps > 100ms
                bg_start_sample = int(last_end_time * sr)
                bg_end_sample = int(start * sr)
                background_segments.append(audio[bg_start_sample:bg_end_sample])
            last_end_time = end

        # Check for gap after last speech segment
        if last_end_time < len(audio) / sr:
            bg_start_sample = int(last_end_time * sr)
            background_segments.append(audio[bg_start_sample:])
            
        logger.info(f"Extracted {len(background_segments)} background noise segments.")
        return background_segments

    def get_audio_stats(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculates basic statistics for a given audio signal.

        Args:
            audio: The audio signal.
            sr: The sample rate.

        Returns:
            A dictionary of audio statistics.
        """
        duration = librosa.get_duration(y=audio, sr=sr)
        if duration == 0:
            return {
                'duration': 0, 'sample_rate': sr, 'num_samples': 0,
                'rms_energy': 0, 'max_amplitude': 0, 'zero_crossing_rate': 0
            }

        rms_energy = np.sqrt(np.mean(audio**2))
        max_amplitude = np.max(np.abs(audio))
        zero_crossings = librosa.zero_crossings(audio)
        zero_crossing_rate = np.mean(zero_crossings)

        return {
            'duration': duration,
            'sample_rate': sr,
            'num_samples': len(audio),
            'rms_energy': float(rms_energy),
            'max_amplitude': float(max_amplitude),
            'zero_crossing_rate': float(zero_crossing_rate)
        }

if __name__ == "__main__":
    """
    Provides a simple test run for this module.
    You will need a test audio file to run this.
    
     Instructions:
     1. Activate your virtual environment:
         - macOS / Linux: `source venv/bin/activate`
         - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
         - Windows CMD: `.\venv\Scripts\activate.bat`
     2. Place a test audio file (e.g., 'test.wav') in the 'TellTale/Code/temp/' folder.
     3. Run this module from the 'TellTale/Code/' directory:
         `python -m modules.audio_processor`
    """
    logger.info("Running AudioProcessor module test...")
    
    # Define a test file path. We'll look in the 'temp' folder.
    test_audio_path = "temp/test_audio.wav" # You must create this file!

    try:
        # 1. Initialize Processor
        processor = AudioProcessor()

        # 2. Load Audio
        audio, sr, duration = processor.load_audio(test_audio_path)
        logger.info(f"Loaded audio. Duration: {duration:.2f}s")

        # 3. Normalize Audio
        normalized_audio = processor.normalize_audio(audio)
        logger.info(f"Normalized audio. New max amplitude: {np.max(np.abs(normalized_audio)):.2f}")

        # 4. Get Stats on Original Audio
        stats_original = processor.get_audio_stats(normalized_audio, sr)
        logger.info(f"Original Audio Stats: {stats_original}")

        # 5. Remove Silence
        audio_no_silence, segments = processor.remove_silence(normalized_audio, sr)
        if len(audio_no_silence) > 0:
            logger.info(f"Silence removed. {len(segments)} speech segments found.")
            stats_no_silence = processor.get_audio_stats(audio_no_silence, sr)
            logger.info(f"VAD Audio Stats: {stats_no_silence}")
        else:
            logger.warning("No audio returned after VAD.")

        # 6. Extract Environment Audio
        env_segments = processor.extract_environment_audio(normalized_audio, sr)
        logger.info(f"Extracted {len(env_segments)} environment/background segments.")

        # 7. Separate Speakers (Placeholder)
        _, candidate_segments = processor.separate_speakers(normalized_audio, sr)
        logger.info(f"Found {len(candidate_segments)} 'candidate' speech segments.")
        
        logger.info("AudioProcessor module test complete.")

    except FileNotFoundError:
        logger.error(f"Test file not found at '{test_audio_path}'.")
        logger.error("Please place a test audio file (WAV, MP3, etc.) at that location "
                     "and rename it to 'test_audio.wav'.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)