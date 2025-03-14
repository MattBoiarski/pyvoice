import importlib
import subprocess
import sys


# Pip handling
def get_pip_version():
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
        return result.stdout.split()[1]
    except Exception:
        return None
        
import time     
SLEEP_DURATION = 0.25  # Adjust this value as needed    
def get_latest_pip_version():
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--dry-run"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "Would install" in line:
                return line.split()[-1]
        return None
    except Exception:
        return None
        
SLEEP_DURATION = 0.25  
def import_or_install(packages):
    if isinstance(packages, str):
        packages = [{'import_name': packages}]
    elif isinstance(packages, dict):
        packages = [packages]
    elif not isinstance(packages, list):
        raise ValueError("packages must be a string, a dictionary, or a list of dictionaries")

    yes_to_all = False
    packages_to_install = []

    # First, check and update pip if necessary
    print("\033[93mChecking pip version...\033[0m")
    time.sleep(SLEEP_DURATION)
    
    current_version = get_pip_version()
    latest_version = get_latest_pip_version()
    
    if current_version and latest_version and current_version != latest_version:
        print(f"\033[93mUpgrading pip from {current_version} to {latest_version}\033[0m")
        time.sleep(SLEEP_DURATION)
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("\033[92mpip has been successfully updated\033[0m")
        except subprocess.CalledProcessError:
            print("\033[91mFailed to update pip. Continuing with package installation...\033[0m")
    else:
        print("\033[92mpip is already up-to-date\033[0m")

    for package in packages:
        if isinstance(package, str):
            package = {'import_name': package}
        
        import_name = package['import_name']
        install_name = package.get('install_name', import_name)
        version = package.get('version', '')

        if import_name.lower() == 'tkinter':
            print(f"\033[92mtkinter is installed by default with Python\033[0m")
            time.sleep(SLEEP_DURATION)
            continue

        try:
            importlib.import_module(import_name)
            print(f"\033[92m{import_name} is installed\033[0m")
            time.sleep(SLEEP_DURATION)
        except ImportError:
            print(f"\033[91m{import_name} not found.\033[0m")
            time.sleep(SLEEP_DURATION)
            packages_to_install.append((import_name, install_name, version))

    if packages_to_install:
        if not yes_to_all:
            user_input = input(f"\033[33mDo you want to install all missing packages? ([\033[32mY\033[0m\033[33m]/n/all):\033[0m")
            if user_input.strip().lower() == 'all' or user_input.strip().lower() == 'y' or user_input.strip().lower() == "":
                yes_to_all = True
            elif user_input.strip().lower() != 'y':
                print("The program requires these packages to run. Exiting...")
                time.sleep(SLEEP_DURATION)
                sys.exit(1)

        for import_name, install_name, version in packages_to_install:
            if not yes_to_all:
                user_input = input(f"Do you want to install {install_name}? (y/n): ")
                if user_input.strip().lower() != 'y':
                    print(f"The program requires {import_name} to run. Exiting...")
                    time.sleep(SLEEP_DURATION)
                    sys.exit(1)

            try:
                install_command = [sys.executable, "-m", "pip", "install"]
                if version:
                    install_command.append(f"{install_name}{version}")
                else:
                    install_command.append(install_name)

                subprocess.run(install_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                importlib.import_module(import_name)
                print(f"\033[92m{install_name} has been successfully installed\033[0m")
                time.sleep(SLEEP_DURATION)
            except subprocess.CalledProcessError:
                print(f"\033[93m{install_name} installation failed\033[0m")
                time.sleep(SLEEP_DURATION)
                sys.exit(1)

# List of required packages
required_packages = [
    'numpy',
    'sounddevice',
    'librosa',
    'faster_whisper',
    'PyQt6',
    'pyqtgraph',
]
import_or_install(required_packages)

import numpy as np
import sounddevice as sd
import librosa
import queue
from faster_whisper import WhisperModel
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import pyqtgraph as pg


### RECORD AUDIO ###
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

### TRANSCRIBE RECORDED AUDIO ###
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


