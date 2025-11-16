import parselmouth
import opensmile
import numpy as np
import pandas as pd
import logging
import tempfile
import os
import soundfile as sf
from scipy.stats import linregress
from typing import Dict, Any, List
import librosa # <-- Import librosa

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProsodyExtractor:
    """
    Extracts prosodic and acoustic features from audio files using
    openSMILE (eGeMAPSv02) and Praat/parselmouth (Pitch, Intensity, etc.).
    """

    def __init__(self, pitch_floor: float = 75.0, pitch_ceiling: float = 500.0):
        """
        Initializes the ProsodyExtractor.

        Args:
            pitch_floor: Minimum pitch (Hz) to consider (Praat setting).
            pitch_ceiling: Maximum pitch (Hz) to consider (Praat setting).
        """
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        
        # Initialize openSMILE
        # eGeMAPSv02 provides 88 summary features (functionals)
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
                loglevel=0 # Suppress C++ logging
            )
            logger.info("openSMILE (eGeMAPSv02, Functionals) initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize openSMILE: {e}")
            self.smile = None

    def extract_all_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Main method to extract all prosodic features from an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            A dictionary containing all extracted features.
        """
        all_features = {}

        # --- FIX: Load audio once with librosa to ensure compatibility ---
        try:
            # Load with a consistent sample rate. Praat/openSMILE will resample if needed,
            # but loading it correctly first is key.
            # We use a non-standard SR (like 22050) just for loading,
            # then pass the signal and its SR to the tools.
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            logger.info(f"Successfully loaded audio with librosa (SR: {sr})")
        except Exception as e:
            logger.error(f"librosa failed to load audio, cannot proceed: {e}")
            return {}
        # --- End of FIX ---

        # 1. Extract openSMILE features
        try:
            opensmile_features = self._extract_opensmile_features(audio, sr) # Pass signal
            all_features.update(opensmile_features)
        except Exception as e:
            logger.error(f"openSMILE feature extraction failed: {e}", exc_info=True)

        # 2. Extract Praat features
        try:
            praat_features = self._extract_praat_features(audio, sr) # Pass signal
            all_features.update(praat_features)
        except Exception as e:
            logger.error(f"Praat feature extraction failed: {e}", exc_info=True)
            
        # 3. Analyze pitch contour from Praat data
        if "pitch_values" in all_features and "pitch_times" in all_features:
            try:
                contour_features = self._analyze_pitch_contour(
                    all_features["pitch_values"],
                    all_features["pitch_times"]
                )
                all_features.update(contour_features)
            except Exception as e:
                logger.error(f"Pitch contour analysis failed: {e}", exc_info=True)
        
        # Clean up intermediate arrays if they exist
        all_features.pop("pitch_values", None)
        all_features.pop("pitch_times", None)
        
        logger.info(f"Successfully extracted {len(all_features)} prosodic features.")
        return all_features

    def _extract_opensmile_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extracts eGeMAPSv02 summary features using openSMILE from an audio signal.
        """
        if self.smile is None:
            logger.error("openSMILE is not initialized. Skipping extraction.")
            return {}
            
        logger.info("Extracting openSMILE features...")
        # process_file returns a DataFrame, we take the first row
        # --- FIX: Use process_signal instead of process_file ---
        features_df = self.smile.process_signal(audio, sr)
        # --- End of FIX ---
        
        # Convert to a dictionary and rename keys to be more Python-friendly
        features = features_df.iloc[0].to_dict()
        renamed_features = {f"os_{key}": val for key, val in features.items()}
        
        logger.info(f"Extracted {len(renamed_features)} openSMILE features.")
        return renamed_features

    def _extract_praat_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extracts core features (pitch, intensity, formants) using parselmouth
        from an audio signal.
        """
        logger.info("Extracting Praat (parselmouth) features...")
        features: Dict[str, Any] = {}

        try:
            # --- FIX: Load Sound from numpy array instead of file path ---
            snd = parselmouth.Sound(
                values=audio.astype(np.float64),
                sampling_frequency=float(sr)
            )
            # --- End of FIX ---
        except Exception as e:
            logger.error(f"parselmouth failed to create sound from array: {e}", exc_info=True)
            return {}

        # 1. Pitch (Fundamental Frequency)
        pitch = None
        try:
            pitch = snd.to_pitch(pitch_floor=self.pitch_floor, pitch_ceiling=self.pitch_ceiling)
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values == 0] = np.nan # Use nan for unvoiced frames
            
            valid_pitch_values = pitch_values[~np.isnan(pitch_values)]
            
            if len(valid_pitch_values) > 0:
                features['praat_pitch_mean'] = np.mean(valid_pitch_values)
                features['praat_pitch_std'] = np.std(valid_pitch_values)
                features['praat_pitch_min'] = np.min(valid_pitch_values)
                features['praat_pitch_max'] = np.max(valid_pitch_values)
                features['praat_pitch_cv'] = np.std(valid_pitch_values) / np.mean(valid_pitch_values)
            else:
                logger.warning("No valid pitch found in audio.")
            
            # Store for contour analysis
            features['pitch_values'] = pitch_values
            features['pitch_times'] = pitch.xs()
            
        except Exception as e:
            logger.error(f"Praat pitch extraction failed: {e}", exc_info=True)

        # 2. Intensity
        try:
            intensity = snd.to_intensity()
            intensity_values = intensity.values[0] # Get 1D array
            
            features['praat_intensity_mean'] = np.mean(intensity_values)
            features['praat_intensity_std'] = np.std(intensity_values)
        except Exception as e:
            logger.error(f"Praat intensity extraction failed: {e}", exc_info=True)

        # 3. Voice Quality (Jitter, Shimmer, HNR)
        if pitch is not None:
            try:
                # Create a PointProcess for jitter/shimmer
                point_process = parselmouth.praat.call(pitch, "To PointProcess")
                
                # Jitter
                jitter_local = parselmouth.praat.call(
                    point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
                )
                features['praat_jitter_local'] = jitter_local

                # Shimmer
                shimmer_local = parselmouth.praat.call(
                    point_process, "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
                )
                features['praat_shimmer_local'] = shimmer_local

                # Harmonics-to-Noise Ratio (HNR)
                hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                hnr_mean = parselmouth.praat.call(hnr, "Get mean", 0.0, 0.0)
                features['praat_hnr_mean'] = hnr_mean
                
            except Exception as e:
                # This can fail on very short or unvoiced audio
                logger.warning(f"Praat voice quality (jitter/shimmer/HNR) extraction failed: {e}")
        else:
            logger.warning("Skipping voice quality extraction because pitch is unavailable.")

        logger.info(f"Extracted {len(features)} Praat features.")
        return features

    def _analyze_pitch_contour(self, pitch_values: np.ndarray, 
                                 time_values: np.ndarray) -> Dict[str, float]:
        """
        Calculates advanced pitch contour features.
        
        Args:
            pitch_values: Array of pitch values (with NaNs).
            time_values: Corresponding time array.
            
        Returns:
            Dictionary of contour features.
        """
        valid_indices = ~np.isnan(pitch_values)
        if np.sum(valid_indices) < 2: # Need at least 2 points for stats
            logger.warning("Not enough valid pitch data for contour analysis.")
            return {}

        pitch_valid = pitch_values[valid_indices]
        time_valid = time_values[valid_indices]

        features = {}

        # 1. Pitch Declination (slope of pitch over time)
        try:
            slope, intercept, r_value, p_value, std_err = linregress(time_valid, pitch_valid)
            features['pitch_declination'] = slope
        except Exception as e:
            logger.error(f"Pitch declination calculation failed: {e}")

        # 2. Pitch Monotonicity (how often pitch goes up vs. down)
        try:
            diffs = np.diff(pitch_valid)
            ups = np.sum(diffs > 0)
            downs = np.sum(diffs < 0)
            total_changes = ups + downs
            if total_changes > 0:
                # 1.0 = perfectly monotonic (always up or always down)
                # 0.0 = perfectly varied (up-down-up-down)
                monotonicity = np.abs(ups - downs) / total_changes
                features['pitch_monotonicity'] = monotonicity
            else:
                features['pitch_monotonicity'] = 0.0
        except Exception as e:
            logger.error(f"Pitch monotonicity calculation failed: {e}")

        return features

    # Note: Rhythm features (PVI, syllable rate) are complex
    # and often part of openSMILE's eGeMAPS set.
    # For example, `os_Rhythm_PostVoc_Mean` is included.
    # We will rely on openSMILE for rhythm for now.

if __name__ == "__main__":
    """
    Provides a simple test run for this module.
    
     Instructions:
     1. Activate your virtual environment:
         - macOS / Linux: `source venv/bin/activate`
         - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
         - Windows CMD: `.\venv\Scripts\activate.bat`
     2. Make sure your 'temp/test_audio.wav' file exists.
     3. Run this module from the 'TellTale/Code/' directory:
         `python -m modules.prosody_extractor`
    """
    logger.info("Running ProsodyExtractor module test...")

    test_audio_path = "temp/test_audio.wav"

    try:
        # 1. Initialize Extractor
        extractor = ProsodyExtractor()

        # 2. Extract Features
        if extractor.smile is not None:
            features = extractor.extract_all_features(test_audio_path)
            
            logger.info(f"--- Extracted {len(features)} Features ---")
            
            # Print a few key features
            key_features = [
                'praat_pitch_mean', 'praat_pitch_std', 'praat_pitch_cv',
                'pitch_declination', 'pitch_monotonicity',
                'praat_jitter_local', 'praat_hnr_mean',
                'os_F0semitoneFrom27.5Hz_Stddev_sma3' # openSMILE's pitch std
            ]
            
            for key in key_features:
                if key in features:
                    logger.info(f"{key}: {features[key]:.4f}")
                else:
                    logger.warning(f"Key feature '{key}' not found.")
            
            logger.info("ProsodyExtractor module test complete.")
        else:
            logger.error("Test failed: openSMILE could not be initialized.")

    except FileNotFoundError:
        logger.error(f"Test file not found at '{test_audio_path}'.")
        logger.error("Please ensure 'temp/test_audio.wav' exists.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)