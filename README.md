# PyVoice
[![PyPI version](https://badge.fury.io/py/pyvoice.svg)](https://badge.fury.io/py/pyvoice)

PyVoice is a real-time speech-to-text transcription tool using machine learning. It provides a graphical user interface for recording audio and transcribing it into text.

## Features

- Real-time audio recording  
- Live audio waveform visualization  
- Speech-to-text transcription using the Whisper model  
- User-friendly GUI built with PyQt6  

## Requirements

The script automatically checks for and installs the following required packages:

- `numpy`  
- `sounddevice`  
- `librosa`  
- `faster_whisper`  
- `PyQt6`  
- `pyqtgraph`  

## Installation

1. Ensure you have Python installed on your system.  
2. Download the PyVoice script file.  
3. Run the script. It will automatically check for and install required packages.  

## Usage

1. Run the PyVoice script to launch the application.  
2. Click **"Start Recording"** to begin audio capture.  
3. Speak into your microphone.  
4. Click **"Stop Recording"** when you're done speaking.  
5. PyVoice will process the audio and display the transcription.  

## Key Components

- **AudioRecorder**: Handles real-time audio recording.  
- **TranscriptionWorker**: Processes recorded audio and generates transcriptions.  
- **MainWindow**: Manages the graphical user interface and coordinates recording and transcription processes.  

## Notes

- PyVoice uses the **"small" Whisper model** for transcription.  
- Audio is processed at a **16kHz sample rate**.  
- The waveform display updates in real-time during recording.  
