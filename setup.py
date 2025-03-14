from setuptools import setup

setup(
    name="pyvoice",
    version="0.1.0",
    py_modules=["voice"],
    install_requires=[
        "numpy",
        "sounddevice",
        "librosa",
        "faster-whisper",
        "PyQt6",
        "pyqtgraph"
    ],
    author="Matthew Boiarski",
    author_email="21boiarskim@gmail.com",
    description="A real-time speech-to-text transcription tool using machine learning (NumPy), PyQt6, and faster-whisper.",
    long_description=open("README.md").read() if open("README.md").readable() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/MattBoiarski/pyvoice",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "pyvoice=voice:run_app",
        ],
    },
)
