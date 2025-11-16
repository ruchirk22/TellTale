import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml # <-- Import YAML
from typing import Dict, Any

# --- Import Our Modules ---
from modules.audio_processor import AudioProcessor
from modules.transcription import WhisperTranscriber, TranscriptAnalyzer
from modules.prosody_extractor import ProsodyExtractor
from modules.disfluency_detector import DisfluencyDetector, DisfluencyComparator
from modules.ai_content_detector import AIContentDetector

# --- Global Config Loading ---
CONFIG_PATH = Path(__file__).parent / "config/config.yaml"
THRESHOLDS_PATH = Path(__file__).parent / "config/thresholds.yaml"

def load_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {path}: {e}")
        return {}

CONFIG = load_config(CONFIG_PATH)
THRESHOLDS = load_config(THRESHOLDS_PATH)

# --- Set up logging ---
log_level = CONFIG.get('logging', {}).get('level', 'INFO')
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewCheatingDetector:
    """
    Main pipeline to integrate all analysis modules and generate a final
    cheating detection assessment.
    """

    def __init__(self, whisper_model: str = "base"):
        """
        Initializes all analyzer modules.
        
        Args:
            whisper_model: The size of the Whisper model to use.
        """
        logger.info(f"Initializing InterviewCheatingDetector with Whisper model '{whisper_model}'...")
        try:
            # Load settings from CONFIG
            audio_config = CONFIG.get('audio', {})
            prosody_config = CONFIG.get('prosody', {})
            ai_config = CONFIG.get('ai_detection', {})
            
            self.audio_processor = AudioProcessor(
                target_sr=audio_config.get('sample_rate', 16000)
            )
            self.transcriber = WhisperTranscriber(
                model_size=whisper_model
            )
            self.transcript_analyzer = TranscriptAnalyzer()
            self.prosody_extractor = ProsodyExtractor(
                pitch_floor=prosody_config.get('pitch_floor', 75),
                pitch_ceiling=prosody_config.get('pitch_ceiling', 500)
            )
            self.disfluency_detector = DisfluencyDetector()
            self.ai_content_detector = AIContentDetector(
                device=ai_config.get('device', 'cpu') # Use device from config
            )
            
            # Load thresholds
            self.thresholds = THRESHOLDS
            
            logger.info("All modules initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}", exc_info=True)
            raise

    def analyze_interview(self, audio_path: str, output_dir: str = None) -> Dict:
        """
        Runs the full analysis pipeline on a single audio file.
        
        Args:
            audio_path: The path to the interview audio file.
            output_dir: Optional. Directory to save detailed results and report.
            
        Returns:
            A dictionary containing the full analysis results.
        """
        logger.info(f"\n{'='*60}\nANALYZING INTERVIEW: {audio_path}\n{'='*60}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Results will be saved to: {output_dir}")
            
        results = {
            'metadata': {
                'audio_path': str(audio_path),
                'analysis_timestamp': datetime.now().isoformat(),
            },
            'audio_info': {},
            'transcript': {},
            'features': {
                'prosody': {},
                'timing': {},
                'disfluency': {},
                'ai_content': {}
            },
            'scores': {},
            'assessment': {}
        }

        try:
            # === STAGE 1: Audio Processing ===
            logger.info("--- STAGE 1: Audio Processing ---")
            audio, sr, duration = self.audio_processor.load_audio(audio_path)
            normalized_audio = self.audio_processor.normalize_audio(audio)
            results['audio_info'] = self.audio_processor.get_audio_stats(normalized_audio, sr)
            logger.info(f"Audio loaded and normalized. Duration: {duration:.2f}s")

            # === STAGE 2: Transcription & Timing ===
            logger.info("--- STAGE 2: Transcription & Timing Analysis ---")
            transcript_result = self.transcriber.transcribe(audio_path)
            full_text = self.transcriber.get_transcript_text(transcript_result)
            words_with_pauses = self.transcriber.get_words_with_pauses(transcript_result)
            
            results['transcript'] = {
                'full_text': full_text,
                'words_with_pauses': words_with_pauses # For debugging
            }
            
            wpm, _ = self.transcript_analyzer.calculate_speaking_rate(transcript_result)
            pause_stats = self.transcript_analyzer.get_pause_statistics(words_with_pauses)
            
            results['features']['timing'] = {
                'speaking_rate_wpm': wpm,
                **pause_stats # Unpack all stats
            }
            logger.info(f"Transcription complete. WPM: {wpm:.2f}")
            logger.info(f"Pause stats: {pause_stats}")

            # === STAGE 3: Disfluency Analysis ===
            logger.info("--- STAGE 3: Disfluency Analysis ---")
            disfluency_features = self.disfluency_detector.analyze_transcript(
                full_text, words_with_pauses
            )
            disfluency_assessment = DisfluencyComparator.assess_spontaneity(disfluency_features)
            
            results['features']['disfluency'] = disfluency_features
            results['assessment']['spontaneity'] = disfluency_assessment
            logger.info(f"Disfluency assessment: {disfluency_assessment['assessment']}")

            # === STAGE 4: Prosody Analysis ===
            logger.info("--- STAGE 4: Prosody Analysis ---")
            prosody_features = self.prosody_extractor.extract_all_features(audio_path)
            results['features']['prosody'] = prosody_features
            logger.info(f"Extracted {len(prosody_features)} prosodic features.")

            # === STAGE 5: AI Content Analysis ===
            logger.info("--- STAGE 5: AI Content Analysis ---")
            if full_text:
                ai_content_features = self.ai_content_detector.analyze_text(full_text)
                results['features']['ai_content'] = ai_content_features
                logger.info(f"AI Content Probability: {ai_content_features.get('ai_content_probability', 0):.2%}")
            else:
                logger.warning("Transcript is empty, skipping AI content analysis.")
                results['features']['ai_content'] = {}

            # === STAGE 6: Scoring & Final Assessment ===
            logger.info("--- STAGE 6: Final Scoring & Assessment ---")
            scores = self._calculate_detection_scores(results['features'])
            results['scores'] = scores
            
            assessment = self._make_final_assessment(scores)
            results['assessment']['final'] = assessment
            
            logger.info(f"FINAL ASSESSMENT: {assessment['risk_level']}")
            logger.info(f"FINAL PROBABILITY: {assessment['cheating_probability']:.2%}")

            # === STAGE 7: Save Results ===
            if output_dir:
                self._save_results(results, output_dir)

            logger.info(f"\n{'='*60}\nANALYSIS COMPLETE.\n{'='*60}")
            return results

        except Exception as e:
            logger.error(f"Error in main analysis pipeline: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_detection_scores(self, features: Dict) -> Dict[str, float]:
        """
        Calculates individual (0-1) abnormality scores for each feature set.
        Score of 1.0 = 100% abnormal/suspicious.
        Score of 0.0 = 100% normal/natural.
        """
        scores = {}
        t_prosody = self.thresholds.get('prosody', {})
        t_pause = self.thresholds.get('pause', {})
        t_disfluency = self.thresholds.get('disfluency', {})

        # 1. Prosody Abnormality (Reading vs. Spontaneous)
        prosody_feats = features.get('prosody', {})
        score_components = []
        
        pitch_cv = prosody_feats.get('praat_pitch_cv', 0.20)
        if pitch_cv < t_prosody.get('pitch_cv_suspicious', 0.15): 
            score_components.append(0.8)
        elif pitch_cv < t_prosody.get('pitch_cv_somewhat', 0.25): 
            score_components.append(0.4)
        else: 
            score_components.append(0.0)
        
        monotonicity = prosody_feats.get('pitch_monotonicity', 0.5)
        if monotonicity > t_prosody.get('monotonicity_suspicious', 0.7): 
            score_components.append(0.9)
        elif monotonicity > t_prosody.get('monotonicity_somewhat', 0.5): 
            score_components.append(0.5)
        else: 
            score_components.append(0.0)
        
        scores['prosody_abnormality'] = np.mean(score_components)

        # 2. Pause Abnormality (Unnatural timing)
        timing_feats = features.get('timing', {})
        score_components = []
        
        micropause_count = timing_feats.get('micropause_count', 5)
        if micropause_count < t_pause.get('micropause_suspicious', 3): 
            score_components.append(0.8)
        elif micropause_count < t_pause.get('micropause_somewhat', 7): 
            score_components.append(0.4)
        else: 
            score_components.append(0.0)

        pause_std = timing_feats.get('std_pause', 0.15)
        if pause_std < t_pause.get('pause_std_suspicious', 0.1): 
            score_components.append(0.7)
        else: 
            score_components.append(0.0)
        
        scores['pause_abnormality'] = np.mean(score_components)

        # 3. Disfluency Absence (Suspiciously "perfect" speech)
        disfluency_score = features.get('disfluency', {}).get('disfluency_score', 50)
        if disfluency_score < t_disfluency.get('score_suspicious_1', 10): 
            scores['disfluency_absence'] = 0.9
        elif disfluency_score < t_disfluency.get('score_suspicious_2', 20): 
            scores['disfluency_absence'] = 0.7
        elif disfluency_score < t_disfluency.get('score_suspicious_3', 40): 
            scores['disfluency_absence'] = 0.3
        else: 
            scores['disfluency_absence'] = 0.0

        # 4. AI Content Probability (Direct from module)
        scores['ai_content_probability'] = features.get('ai_content', {}).get('ai_content_probability', 0.0)

        # 5. Rate Consistency (Placeholder)
        scores['rate_consistency'] = 0.0

        return scores

    def _make_final_assessment(self, scores: Dict) -> Dict[str, Any]:
        """
        Applies weights and thresholds to generate the final verdict.
        """
        weights = self.thresholds.get('weights', {})
        levels = self.thresholds.get('risk_levels', {})

        cheating_probability = sum(
            scores.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        
        # Determine risk level
        if cheating_probability >= levels.get('HIGH', 0.75): 
            risk_level = "HIGH"
        elif cheating_probability >= levels.get('MEDIUM-HIGH', 0.60): 
            risk_level = "MEDIUM-HIGH"
        elif cheating_probability >= levels.get('MEDIUM', 0.45): 
            risk_level = "MEDIUM"
        elif cheating_probability >= levels.get('LOW-MEDIUM', 0.30): 
            risk_level = "LOW-MEDIUM"
        else: 
            risk_level = "LOW"
        
        # Identify key evidence
        evidence = []
        if scores.get('prosody_abnormality', 0) > 0.6:
            evidence.append("Abnormal prosodic patterns (monotonous, flat) suggest reading.")
        if scores.get('pause_abnormality', 0) > 0.6:
            evidence.append("Pause patterns are unnatural (too few or too regular).")
        if scores.get('disfluency_absence', 0) > 0.7:
            evidence.append("Speech is suspiciously fluent (lacks normal 'um's' or repetitions).")
        if scores.get('ai_content_probability', 0) > 0.7:
            evidence.append("Transcript content has high probability of being AI-generated.")
            
        if not evidence:
            evidence.append("No significant indicators of cheating detected.")

        return {
            'cheating_probability': cheating_probability,
            'risk_level': risk_level,
            'evidence': evidence,
            'score_breakdown': scores
        }

    def _save_results(self, results: Dict, output_dir: Path):
        """Saves the full results JSON and a summary text report."""
        
        # 1. Save Full JSON
        json_path = output_dir / "full_analysis_results.json"
        try:
            with open(json_path, 'w') as f:
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer): return int(obj)
                        if isinstance(obj, np.floating): return float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        return super(NumpyEncoder, self).default(obj)
                
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Full JSON results saved to: {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON results: {e}")

        # 2. Save Summary Report
        report_path = output_dir / "summary_report.txt"
        assessment = results.get('assessment', {}).get('final', {})
        
        try:
            with open(report_path, 'w') as f:
                f.write(f"{'='*50}\n")
                f.write(" AI INTERVIEW CHEATING DETECTION REPORT\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Audio File: {results.get('metadata', {}).get('audio_path')}\n")
                f.write(f"Analysis Date: {results.get('metadata', {}).get('analysis_timestamp')}\n\n")
                
                f.write(f"--- FINAL ASSESSMENT ---\n")
                f.write(f"Risk Level: {assessment.get('risk_level', 'N/A')}\n")
                f.write(f"Cheating Probability: {assessment.get('cheating_probability', 0):.2%}\n\n")
                
                f.write(f"--- KEY EVIDENCE ---\n")
                for item in assessment.get('evidence', ['N/A']):
                    f.write(f"- {item}\n")
                
                f.write("\n--- SCORE BREAKDOWN (1.0 = 100% Suspicious) ---\n")
                for key, val in assessment.get('score_breakdown', {}).items():
                    f.write(f"- {key}: {val:.3f}\n")
                    
                f.write(f"\n--- TRANSCRIPT ---\n")
                f.write(results.get('transcript', {}).get('full_text', 'No transcript available.'))
                f.write("\n")

            logger.info(f"Summary report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save summary report: {e}")

# --- This is how we run it from the command line ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TellTale: AI Interview Cheating Detection System"
    )
    parser.add_argument(
        "audio_file", 
        help="Path to the interview audio file (e.g., temp/test_audio.wav)"
    )
    parser.add_argument(
        "--output", "-o", 
        default="results/", 
        help="Directory to save the analysis reports (default: 'results/')"
    )
    parser.add_argument(
        "--model", "-m", 
        default=CONFIG.get('transcription', {}).get('whisper_model', 'base'),
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help="Whisper model size to use (default: 'base')"
    )
    args = parser.parse_args()

    # Create a unique output directory for this run
    file_stem = Path(args.audio_file).stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output) / f"{file_stem}_{run_timestamp}"

    # Initialize and run the detector
    try:
        detector = InterviewCheatingDetector(whisper_model=args.model)
        detector.analyze_interview(args.audio_file, output_dir=run_output_dir)
        
        print(f"\nAnalysis complete. View report in: {run_output_dir}")
        
    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)