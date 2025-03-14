# voice.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import numpy as np

from .audio_recorder import AudioRecorder
from .transcription_worker import TranscriptionWorker
from . import import_or_install

### VOICE APP CLASS DEF ###
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech-to-Text ML App")
        self.setGeometry(100, 100, 600, 500)

        layout = QVBoxLayout()
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-1, 1)
        self.curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.model = WhisperModel("small", device="cpu", compute_type="float32")
        self.audio_recorder = None
        self.audio_data = np.zeros(1000)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(30)
        self.is_processing = False

    def start_recording(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.text_edit.clear()
        self.text_edit.append("Recording... Speak now!")

        self.audio_recorder = AudioRecorder()
        self.audio_recorder.new_data.connect(self.process_audio_data)
        self.audio_recorder.finished.connect(self.on_recording_finished)
        self.audio_recorder.start()

    def stop_recording(self):
        if self.audio_recorder and self.audio_recorder.is_running:
            self.audio_recorder.stop()
            self.audio_recorder.wait()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.is_processing = True

    def process_audio_data(self, data):
        if not self.is_processing:
            self.audio_data = np.concatenate((self.audio_data[-900:], data))[-1000:]

    def update_plot(self):
        self.curve.setData(self.audio_data if not self.is_processing else np.zeros(1000))

    def on_recording_finished(self):
        self.text_edit.append("Processing audio...")
        audio_data = self.audio_recorder.get_audio_data()

        if audio_data.size == 0:
            self.text_edit.append("No audio recorded.")
            self.is_processing = False
            return

        self.transcription_worker = TranscriptionWorker(self.model, audio_data.astype(np.float32), self.audio_recorder.sample_rate)
        self.transcription_worker.transcription_done.connect(self.display_transcription)
        self.transcription_worker.start()

    def display_transcription(self, text):
        self.text_edit.append("\nTranscription:\n" + text)
        self.is_processing = False

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
