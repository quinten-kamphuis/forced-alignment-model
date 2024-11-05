from cog import BasePredictor, Input, Path
import torch
import torchaudio
import inflect
import unicodedata
from typing import Dict, List
import re
from pathlib import Path as PathLib


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Download and load the MMS model
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model(with_star=False).to(self.device)
        
        # Get labels and dictionary
        self.labels = self.bundle.get_labels(star=None)
        self.dictionary = self.bundle.get_dict(star=None)
        
        # Add space and basic ASCII letters to dictionary
        self.dictionary[" "] = len(self.dictionary)
        for c in "abcdefghijklmnopqrstuvwxyz":
            if c not in self.dictionary:
                self.dictionary[c] = len(self.dictionary)

    def is_supported_character(self, char: str) -> bool:
        """Check if a character is supported by the forced alignment dictionary"""
        return char.lower() in self.dictionary or char.isspace()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for alignment"""
        p = inflect.engine()
        
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        
        # Convert numbers to words
        words = []
        for word in text.split():
            if word.isdigit():
                words.append(p.number_to_words(word))
            elif any(char.isdigit() for char in word):
                parts = re.split("(\d+)", word)
                converted = []
                for part in parts:
                    if part.isdigit():
                        converted.append(p.number_to_words(part))
                    elif part:
                        converted.append(part)
                words.append(" ".join(converted))
            else:
                words.append(word)
        
        text = " ".join(words)
        text = text.replace("'", "")
        
        # Keep only supported characters
        cleaned_chars = []
        for char in text:
            if self.is_supported_character(char):
                cleaned_chars.append(char.lower())
            else:
                cleaned_chars.append(" ")
        
        return " ".join("".join(cleaned_chars).split())

    def preprocess_script(self, script: str) -> tuple[List[str], List[str]]:
        """Preprocess script for alignment while preserving original words"""
        original_words = [word for word in script.split() if word.strip()]
        cleaned_text = self.clean_text(script)
        clean_words = [word for word in cleaned_text.split() if word.strip()]
        
        if not clean_words:
            clean_words = ["a"]
            
        return clean_words, original_words

    def predict(
        self,
        audio: Path,
        script: str,
    ) -> List[Dict[str, float | str]]:
        """Run a single prediction on the model"""
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            
            # Process script
            clean_words, original_words = self.preprocess_script(script)
            
            # Convert words to token indices
            token_indices = []
            for word in clean_words:
                for char in word:
                    if char in self.dictionary:
                        token_indices.append(self.dictionary[char])
            
            # Handle empty token_indices
            if not token_indices:
                duration = waveform.size(1) / 16000
                return [{"word": word, "start": 0.0, "end": duration} 
                       for word in original_words]
            
            # Generate emissions
            with torch.inference_mode():
                emissions, _ = self.model(waveform.to(self.device))
            
            # Get alignments
            targets = torch.tensor([token_indices], dtype=torch.int32, device=self.device)
            alignments, scores = torchaudio.functional.forced_align(emissions, targets, blank=0)
            alignments, scores = alignments[0], scores[0]
            
            # Merge tokens
            token_spans = torchaudio.functional.merge_tokens(alignments, scores)
            
            # Calculate timings
            frame_duration = waveform.size(1) / emissions.size(1) / 16000
            total_duration = waveform.size(1) / 16000
            
            if not token_spans:
                word_count = len(original_words)
                segment_duration = total_duration / word_count
                return [
                    {
                        "word": word,
                        "start": i * segment_duration,
                        "end": (i + 1) * segment_duration,
                    }
                    for i, word in enumerate(original_words)
                ]
            
            # Generate word timings
            word_timings = []
            spans_per_word = max(1, len(token_spans) // len(original_words))
            
            for i, word in enumerate(original_words):
                start_idx = i * spans_per_word
                end_idx = min((i + 1) * spans_per_word, len(token_spans))
                
                if start_idx >= len(token_spans):
                    remaining_duration = total_duration - word_timings[-1]["end"]
                    remaining_words = len(original_words) - i
                    segment_duration = remaining_duration / remaining_words
                    
                    word_timings.append({
                        "word": word,
                        "start": word_timings[-1]["end"],
                        "end": word_timings[-1]["end"] + segment_duration,
                    })
                else:
                    start_time = token_spans[start_idx].start * frame_duration
                    end_time = token_spans[min(end_idx - 1, len(token_spans) - 1)].end * frame_duration
                    
                    word_timings.append({
                        "word": word,
                        "start": start_time,
                        "end": end_time,
                    })
            
            return word_timings
            
        except Exception as e:
            # Fallback: even distribution
            try:
                duration = waveform.size(1) / 16000
                word_count = len(script.split())
                segment_duration = duration / word_count
                
                return [
                    {
                        "word": word,
                        "start": i * segment_duration,
                        "end": (i + 1) * segment_duration,
                    }
                    for i, word in enumerate(script.split())
                ]
            except:
                raise Exception(f"Failed to get word timings: {str(e)}")