import soundfile as sf
from kokoro import KPipeline
import spacy
import numpy as np
from typing import Tuple
import torch

# Load spacy model once
nlp = spacy.load('en_core_web_sm')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextToSpeech:
    """Text to Speech converter using Kokoro"""

    LANGUAGES = {
        "English": "a",
        "British English": "b",
        "Hindi": "h",
    }

    VOICES = {
        "English Male": "am_adam",
        "English Female": "af_heart",
        "British English Male": "bm_daniel",
        "British English Female": "bf_alice",
        "Hindi Female": "hf_alpha",
        "Hindi Male": "hm_omega",
    }

    def __init__(self):
        """Initialize TTS pipeline"""
        self.sample_rate = 24000

    def get_voice(self, lang: str, gender: str) -> Tuple[str, str]:
        """Get voice code based on language and gender"""
        lang_code = self.LANGUAGES.get(lang)
        voice_key = f"{lang} {gender}"
        voice_code = self.VOICES.get(voice_key)

        if not lang_code or not voice_code:
            raise ValueError(
                f"Unsupported language/gender combination: {lang}/{gender}")

        return lang_code, voice_code

    def generate_audio(self, text: str, language: str, gender: str) -> str:
        """Generate audio file from text"""
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            lang_code, voice = self.get_voice(language, gender)
            pipeline = KPipeline(lang_code=lang_code, device=device)

            # Generate audio in chunks
            generator = pipeline(
                text, voice=voice, speed=1, split_pattern=r'\n+')

            # Collect all audio chunks
            audio_chunks = []
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(audio)

            if not audio_chunks:
                raise ValueError("No audio generated")

            # Proper handling of audio concatenation
            if len(audio_chunks) == 1:
                # If only one chunk, no need to concatenate
                audio_data = audio_chunks[0]
            else:
                # Make sure we're only concatenating arrays of the same size
                # or resample if needed
                audio_data = np.concatenate(audio_chunks)

            output_file = 'tts_output.wav'
            sf.write(output_file, audio_data, self.sample_rate)
            return output_file

        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {str(e)}")
