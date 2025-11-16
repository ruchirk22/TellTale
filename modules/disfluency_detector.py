import re
import logging
import numpy as np
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DisfluencyDetector:
    """
    Analyzes a transcript text to detect and quantify speech disfluencies,
    which are strong indicators of spontaneous (vs. scripted) speech.
    """

    def __init__(self):
        """
        Initializes the detector with regex patterns for common disfluencies.
        """
        # Regex to find filled pauses (um, uh, ah, er)
        # \b = word boundary, + = one or more occurrences
        self.filled_pause_pattern = re.compile(
            r'\b(um|uh|ah|er|uhm|uhh|ehm)\b', re.IGNORECASE
        )
        
        # Regex for common discourse markers (not always disfluencies, but
        # common in spontaneous speech and rare in AI scripts)
        self.discourse_marker_pattern = re.compile(
            r'\b(well|so|like|you know|i mean|actually|basically)\b', re.IGNORECASE
        )
        
        # Regex for self-corrections
        self.correction_pattern = re.compile(
            r'\b(or rather|sorry|no wait|let me rephrase)\b', re.IGNORECASE
        )
        
        # Regex for false starts (a word fragment followed by a dash)
        # This is highly dependent on the transcriber (e.g., "I went to the st-")
        # Whisper doesn't typically output this, so we'll rely on repetitions.
        # self.false_start_pattern = re.compile(r'\b\w+-\b')
        
        logger.info("DisfluencyDetector initialized.")

    def analyze_transcript(self, text: str, words_with_pauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes a transcript for various disfluencies.

        Args:
            text: The full transcript text.
            words_with_pauses: A list of word dicts from WhisperTranscriber.

        Returns:
            A dictionary of disfluency features.
        """
        features = {}
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            logger.warning("Transcript is empty, skipping disfluency analysis.")
            return {}

        # 1. Filled Pauses
        filled_pauses = self.filled_pause_pattern.findall(text)
        features['filled_pause_count'] = len(filled_pauses)
        features['filled_pause_rate'] = (len(filled_pauses) / total_words) * 100 # per 100 words

        # 2. Discourse Markers
        discourse_markers = self.discourse_marker_pattern.findall(text)
        features['discourse_marker_count'] = len(discourse_markers)

        # 3. Self-Corrections
        self_corrections = self.correction_pattern.findall(text)
        features['self_correction_count'] = len(self_corrections)

        # 4. Repetitions
        repetition_features = self._detect_repetitions(words_with_pauses)
        features.update(repetition_features)

        # 5. False Starts (placeholder, as Whisper doesn't provide fragments)
        features['false_start_count'] = 0 # Placeholder

        # 6. Overall Disfluency Score
        total_disfluencies = (
            features['filled_pause_count'] + 
            features['total_repetitions'] +
            features['self_correction_count'] +
            features['false_start_count']
        )
        
        features['total_disfluencies'] = total_disfluencies
        features['disfluency_density'] = (total_disfluencies / total_words) * 100 # per 100 words

        # Calculate score (0-100) based on density
        features['disfluency_score'] = self._calculate_disfluency_score(features)
        
        return features

    def _detect_repetitions(self, words_list: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Detects immediate (e.g., "the the") and near (e.g., "I I think")
        word repetitions.
        """
        words = [w['word'].strip().lower().strip(".,?!") for w in words_list]
        immediate_reps = 0
        near_reps = 0 # Repetition within 3 words
        
        for i in range(1, len(words)):
            if not words[i]: # Skip empty strings
                continue
                
            # Immediate repetition: "the the"
            if words[i] == words[i-1]:
                immediate_reps += 1
            
            # Near repetition: "I I think" or "I was I was"
            if i > 1 and words[i] == words[i-2]:
                near_reps += 1

        return {
            'immediate_word_rep': immediate_reps,
            'near_word_rep': near_reps,
            'total_repetitions': immediate_reps + near_reps
        }

    def _calculate_disfluency_score(self, features: Dict[str, Any]) -> float:
        """
        Calculates a "naturalness" score from 0-100 based on disfluency rates.
        High score = natural. Low score = suspiciously perfect.
        """
        score = 0
        
        # 1. Filled Pauses (most important)
        pause_rate = features['filled_pause_rate']
        if pause_rate == 0:
            score += 0 # Red flag
        elif pause_rate < 1.0: # 1 per 100 words
            score += 10 # Suspiciously low
        elif pause_rate < 3.0: # 1-3 per 100 words
            score += 30 # Natural range
        else: # > 3 per 100 words
            score += 25 # Also natural (maybe nervous)

        # 2. Repetitions
        rep_rate = (features['total_repetitions'] / (len(features) + 1)) * 100
        if rep_rate == 0:
            score += 10 # A bit suspicious
        elif rep_rate < 2.0:
            score += 30 # Natural
        else:
            score += 20 # Very disfluent

        # 3. Self-Corrections
        if features['self_correction_count'] > 0:
            score += 20 # Good sign of spontaneity

        # 4. Discourse Markers
        if features['discourse_marker_count'] > 2:
            score += 20 # Good sign of spontaneous thought
        
        # Max score is 100 (30 + 30 + 20 + 20)
        return min(score, 100.0)


class DisfluencyComparator:
    """
    Assesses spontaneity based on expected disfluency distributions.
    As defined in the Master Prompt.
    """
    
    # Expected values per 100 words
    SPONTANEOUS_EXPECTED = {
        'filled_pause_rate': (1.0, 5.0), # e.g., 1-5 'ums' per 100 words
        'repetition_rate': (0.5, 3.0),
        'disfluency_density': (2.0, 10.0)
    }

    SCRIPTED_EXPECTED = {
        'filled_pause_rate': (0.0, 0.5),
        'repetition_rate': (0.0, 0.2),
        'disfluency_density': (0.0, 1.0)
    }

    @classmethod
    def assess_spontaneity(cls, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares features against expected distributions for
        spontaneous vs. scripted speech.
        """
        pause_rate = features.get('filled_pause_rate', 0)
        disfluency_density = features.get('disfluency_density', 0)
        
        # Simple scoring: how many features are in the "scripted" range?
        scripted_indicators = 0
        
        if (pause_rate >= cls.SCRIPTED_EXPECTED['filled_pause_rate'][0] and
            pause_rate <= cls.SCRIPTED_EXPECTED['filled_pause_rate'][1]):
            scripted_indicators += 1
            
        if (disfluency_density >= cls.SCRIPTED_EXPECTED['disfluency_density'][0] and
            disfluency_density <= cls.SCRIPTED_EXPECTED['disfluency_density'][1]):
            scripted_indicators += 1

        # How many features are in the "spontaneous" range?
        spontaneous_indicators = 0
        
        if (pause_rate >= cls.SPONTANEOUS_EXPECTED['filled_pause_rate'][0] and
            pause_rate <= cls.SPONTANEOUS_EXPECTED['filled_pause_rate'][1]):
            spontaneous_indicators += 1
            
        if (disfluency_density >= cls.SPONTANEOUS_EXPECTED['disfluency_density'][0] and
            disfluency_density <= cls.SPONTANEOUS_EXPECTED['disfluency_density'][1]):
            spontaneous_indicators += 1

        assessment = "inconclusive"
        if scripted_indicators == 2:
            assessment = "scripted"
        elif spontaneous_indicators == 2:
            assessment = "spontaneous"
        elif scripted_indicators > spontaneous_indicators:
            assessment = "likely scripted"
        elif spontaneous_indicators > scripted_indicators:
            assessment = "likely spontaneous"

        # Probability (0 = spontaneous, 1 = scripted)
        prob_scripted = 0.5
        if assessment == "scripted":
            prob_scripted = 0.9
        elif assessment == "likely scripted":
            prob_scripted = 0.7
        elif assessment == "likely spontaneous":
            prob_scripted = 0.3
        elif assessment == "spontaneous":
            prob_scripted = 0.1

        return {
            "assessment": assessment,
            "prob_scripted": prob_scripted,
            "disfluency_score_naturalness": features.get('disfluency_score', 0)
        }


if __name__ == "__main__":
    """
    Provides a simple test run for this module.
    No external files are needed.
    
     Instructions:
     1. Activate your virtual environment:
         - macOS / Linux: `source venv/bin/activate`
         - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
         - Windows CMD: `.\venv\Scripts\activate.bat`
     2. Run this module from the 'TellTale/Code/' directory:
         `python -m modules.disfluency_detector`
    """
    logger.info("Running DisfluencyDetector module test...")
    
    detector = DisfluencyDetector()

    # 1. Test "Natural" Spontaneous Text
    logger.info("\n--- Test 1: Natural Speech ---")
    natural_text = ("Um, so I think, uh, the main advantage is, well, "
                    "actually the, the approach. I mean, I guess I'd say "
                    "it's like... you know, it's flexible.")
    
    # Mock words_with_pauses for repetition analysis
    natural_words = [{"word": w} for w in natural_text.lower().split()]
    
    natural_features = detector.analyze_transcript(natural_text, natural_words)
    natural_assessment = DisfluencyComparator.assess_spontaneity(natural_features)
    
    logger.info(f"Natural Text Features: {natural_features}")
    logger.info(f"Natural Text Assessment: {natural_assessment}")

    # 2. Test "Scripted" (AI-generated) Text
    logger.info("\n--- Test 2: Scripted/AI Speech ---")
    scripted_text = ("The primary advantage is its inherent flexibility. "
                     "Furthermore, the architecture facilitates robust scalability "
                     "and simplifies the integration process.")
    
    scripted_words = [{"word": w} for w in scripted_text.lower().split()]

    scripted_features = detector.analyze_transcript(scripted_text, scripted_words)
    scripted_assessment = DisfluencyComparator.assess_spontaneity(scripted_features)

    logger.info(f"Scripted Text Features: {scripted_features}")
    logger.info(f"Scripted Text Assessment: {scripted_assessment}")
    
    logger.info("\nDisfluencyDetector module test complete.")