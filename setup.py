from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyvoice",
    version="0.1.1",
    author="Matthew Boiarski",
    author_email="21boiarskim@gmail.com",
    description="A real-time speech-to-text transcription tool using machine learning (NumPy), PyQt6, and faster-whisper.",  # Short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MattBoiarski/pyvoice",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],

    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "sounddevice",
        "librosa",
        "faster-whisper",
        "PyQt6",
        "pyqtgraph",
    ],
    entry_points={
        'console_scripts': [
            'pyvoice_run = pyvoice:main',
        ],
    },
)
