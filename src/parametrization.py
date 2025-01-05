import math
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def db_to_volts(db):
    """Convert decibel values to volts."""
    return 10 ** ((db / 20) - 6)

def volts_to_db(volts):
    """Convert volts to decibels."""
    uV = 10 ** -6
    return 20 * math.log10(volts / uV)

def calculate_rise_time_from_fixed_threshold(waveform, time_microsec, fixed_threshold):
    """Calculate rise time using a fixed threshold."""
    start_index = np.argmax(waveform > fixed_threshold)
    start_time = time_microsec[start_index]
    waveform_from_start = waveform[start_index:]
    time_microsec_from_start = time_microsec[start_index:]
    max_index = np.argmax(waveform_from_start)
    max_time = time_microsec_from_start[max_index]
    rise_time = max_time - start_time
    max_amplitude = waveform_from_start[max_index]
    max_amplitude_db = volts_to_db(max_amplitude)
    return rise_time, start_time, max_time, max_amplitude_db, max_amplitude

def calculate_max_duration(waveform, time_microsec, fixed_threshold):
    start_index = np.argmax(waveform > fixed_threshold)
    start_time = time_microsec[start_index]
    threshold_cross_indices = np.where(np.diff((waveform > fixed_threshold).astype(int)) == 1)[0]
    if len(threshold_cross_indices) < 2:
        return None, None, None
    end_time = time_microsec[threshold_cross_indices[-1]]
    max_duration = end_time - start_time
    return max_duration, start_time, end_time

def calculate_threshold_cross_counts(waveform, fixed_threshold):
    """Count the number of threshold crossings in the waveform."""
    threshold_cross_indices = np.where(np.diff((waveform > fixed_threshold).astype(int)) == 1)[0]
    return len(threshold_cross_indices) // 2

def calculate_absolute_energy(waveform, fixed_threshold, sample_rate):
    """Calculate absolute energy in attojoules."""
    threshold_cross_indices = np.where(np.diff((waveform > fixed_threshold).astype(int)) == 1)[0]
    if len(threshold_cross_indices) < 2:
        return None
    start_time = threshold_cross_indices[0]
    end_time = threshold_cross_indices[-1]
    squared_waveform = waveform[start_time:end_time] ** 2
    dx = 1 / sample_rate
    energy = np.trapz(squared_waveform, dx=dx) / 10000
    return energy * 1e18

def calculate_pac_energy(waveform, fixed_threshold, sample_rate):
    """Calculate PAC energy in uV^2/100."""
    threshold_cross_indices = np.where(np.diff((waveform > fixed_threshold).astype(int)) == 1)[0]
    if len(threshold_cross_indices) < 2:
        return None
    start_time = threshold_cross_indices[0]
    end_time = threshold_cross_indices[-1]
    abs_squared_waveform = np.abs(waveform[start_time:end_time])
    dx = 1 / sample_rate
    energy = np.trapz(abs_squared_waveform, dx=dx) / 1e-6
    return energy * 100

def peak_frequency(waveform, sample_rate):
    """Find the peak frequency in the waveform."""
    N = len(waveform)
    T = 1 / sample_rate
    xf = fftfreq(N, T)[:N // 2]
    yf = fft(waveform)[:N // 2]
    peak_freq = xf[np.argmax(np.abs(yf))]
    return peak_freq / 1000

def process_waveforms(input_folder, output_folder, threshold):
    """
    Process waveform files, calculate parameters, generate plots,
    and export results to .txt and Excel.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    results = []
    sample_rate = 5000000

    for file in tqdm(files, desc="Processing waveform files"):
        file_path = os.path.join(input_folder, file)

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Read numerical data starting from line 9
            waveform = np.loadtxt(lines[9:], skiprows=0) / 100

            # Configure time and scale
            time_microsec = np.arange(len(waveform)) * 0.2  # Time interval in microseconds
            time_in_seconds = lines[7].split(': ')[1].strip()

            # Extract information from the file name
            file_name = os.path.basename(file)
            file_name_no_ext = os.path.splitext(file_name)[0]
            split_name = file_name_no_ext.split('_')
            event_number = split_name[2]
            sensor_number = split_name[1]

            if sensor_number != '1':
                continue

            # Calculate parameters
            rise_time, start_time, max_time, max_amplitude_db, max_amplitude = calculate_rise_time_from_fixed_threshold(
                waveform, time_microsec, threshold)
            max_duration, _, _ = calculate_max_duration(waveform, time_microsec, threshold)
            counts = calculate_threshold_cross_counts(waveform, threshold)
            abs_energy = calculate_absolute_energy(waveform, threshold, sample_rate)
            pac_energy = calculate_pac_energy(waveform, threshold, sample_rate)
            peak_freq = peak_frequency(waveform, sample_rate)

            results.append({
                "Event": event_number,
                "Time in Seconds": time_in_seconds,
                "Channel": sensor_number,
                "Rise Time (us)": rise_time,
                "Counts": counts,
                "Max Duration (us)": max_duration,
                "Max Amplitude (dB)": max_amplitude_db,
                "Absolute Energy (aJ)": abs_energy,
                "PAC Energy (uV^2/100)": pac_energy,
                "Peak Frequency (kHz)": peak_freq
            })

            # Plot the waveform
            plt.figure()
            plt.plot(time_microsec, waveform, label='Waveform EA')
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold in dB: {round(20 * np.log10(threshold / (10 ** -6)), 2)}')
            plt.axvline(x=start_time, color='r')
            plt.text(max_time, max_amplitude, f'RT: {rise_time:.2f} us', ha='left', va='bottom', fontsize=10)
            plt.xlabel('Time [us]')
            plt.ylabel('Amplitude [V]')
            plt.title(f"Channel {sensor_number}, Time Test: {time_in_seconds} µs")
            plt.legend()

            # Save the image
            output_plot_path = os.path.join(output_folder, f"{file_name_no_ext}_plot.png")
            plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error processing the file {file}: {e}")

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_folder, "waveform_parameters.txt"), index=False, sep='\t')
        results_df.to_excel(os.path.join(output_folder, "waveform_parameters.xlsx"), index=False)
        print(f"Processing completed. Results saved in: {output_folder}")
    else:
        print("No results were generated. Check the input files.")
