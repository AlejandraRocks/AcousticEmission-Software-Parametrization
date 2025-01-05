# AcousticEmission-Software-Parametrization



This project processes waveform data for Acoustic Emission analysis. It provides a GUI for selecting input/output folders, setting a threshold, HDT waveform threshold, and processing waveforms.

## Instructions

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the application:
    ```
    python src/gui.py
    ```

3. Provide input waveform files in the `input` folder, set a threshold, set a HDT hreshold, and process.

Results will be saved in the `output` folder.

## Requirements
- Python 3.8+
- PyQt5
- NumPy
- pandas
- tqdm
- matplotlib
