# transcription_worker.py
import numpy as np
import librosa
from faster_whisper import WhisperModel
from PyQt6.QtCore import QThread, pyqtSignal

class TranscriptionWorker(QThread):
    transcription_done = pyqtSignal(str)

    def __init__(self, model, audio_data, sample_rate):
        super().__init__()
        self.model = model
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def preprocess_audio(self, audio_data, sample_rate):
        if audio_data.ndim > 1:  # Convert stereo to mono
            audio_data = librosa.to_mono(audio_data.T)
        if sample_rate != 16000:  # Resample if necessary
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        return audio_data.astype(np.float32)

    def run(self):

        self.audio_data = self.preprocess_audio(self.audio_data, self.sample_rate).flatten()
        self.sample_rate = 16000  # Ensure it's 16kHz (for correct shape)

        try:
            segments, info = self.model.transcribe(self.audio_data, language="en", beam_size=5, vad_filter=False)
            segments = list(segments)  # Ensure it's a list

        except Exception as e:
            self.transcription_done.emit("Error during transcription.")
            return

        if not segments:
            return

        transcription = " ".join(segment.text.strip() for segment in segments if getattr(segment, "text", "").strip())

        if not transcription:
            self.transcription_done.emit("No speech detected.")
        else:
            self.transcription_done.emit(transcription)