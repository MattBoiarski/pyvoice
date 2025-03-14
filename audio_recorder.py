# audio_recorder.py
import numpy as np
import sounddevice as sd
import queue
from PyQt6.QtCore import QThread, pyqtSignal

class AudioRecorder(QThread):
    new_data = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, sample_rate=16000, buffer_size=1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_running = False
        self.audio_queue = queue.Queue()

    def run(self):
        self.is_running = True
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback, blocksize=self.buffer_size):
                while self.is_running:
                    sd.sleep(100)
        except Exception as e:
            return
        finally:
            self.finished.emit()

    def audio_callback(self, indata, frames, time, status):
        if indata.size > 0:
            self.new_data.emit(indata[:, 0])
            self.audio_queue.put(indata.copy())

    def stop(self):
        self.is_running = False

    def get_audio_data(self):
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get_nowait())
        return np.concatenate(audio_data) if audio_data else np.array([])

