import numpy as np
from scipy import signal
import PyEMD
import pywt
import csv
from featureFunctions import *
import os


def process_emg_csv(input_csv):
    # Load CSV file with error handling
    try:
        data = np.loadtxt(input_csv, delimiter=',')
        electrode_1 = data[:, 0]
        # Corrected this line to use the second column
        electrode_2 = data[:, 1]
    except IOError:
        print("Error: Could not load the CSV file.")

    # Design Bandpass filter
    fs = 4000        # Sampling frequency (Hz)
    lowcut = 20      # Lower cutoff frequency (Hz)
    highcut = 450    # Upper cutoff frequency (Hz)
    order = 4        # Filter order
    nyquist = 0.5 * fs
    desired_bandwidth = highcut - lowcut  # 430 Hz
    scaling_factor = 4.0
    numtaps = int(scaling_factor * fs / desired_bandwidth)
    if numtaps % 2 == 0:
        numtaps += 1
    b_bandpass = signal.firwin(
        numtaps, [lowcut, highcut],  window='hamming', pass_zero=False, fs=fs)

    # Design the FIR notch filter
    f0 = 50.0  # Frequency to be removed in Hz
    Q = 30.0    # Quality factor
    b_notch = signal.firwin(
        numtaps, [f0 - 1.0, f0 + 1.0], fs=fs, pass_zero=False)

    # Filter Data using the bandpass and notch filters on Electrode 1
    bandpass_data_1 = signal.filtfilt(b_bandpass, 1, electrode_1, padlen=0)
    notch_and_bandpass_1 = signal.filtfilt(
        b_notch, [1.0], bandpass_data_1, padlen=0)

    # Filter Data using the bandpass and notch filters on Electrode 2
    bandpass_data_2 = signal.filtfilt(b_bandpass, 1, electrode_2, padlen=0)
    notch_and_bandpass_2 = signal.filtfilt(
        b_notch, [1.0], bandpass_data_2, padlen=0)

    # Parameters for overlapped windowing
    frame_length = 400  # Length of each frame
    overlap = 200  # Overlap between consecutive frames

    # Perform overlapped windowing and save each window to a nested array
    windows_electrode_1 = [notch_and_bandpass_1[i:i+frame_length]
                           for i in range(0, len(notch_and_bandpass_1) - frame_length + 1, overlap)]
    windows_electrode_2 = [notch_and_bandpass_2[i:i+frame_length]
                           for i in range(0, len(notch_and_bandpass_2) - frame_length + 1, overlap)]

    # Get EMD of each window and save it to array
    imfs_per_window = []
    for window in windows_electrode_1:
        # Perform EMD
        emd = PyEMD.EMD()
        imfs = emd(window)
        imfs_per_window.append(imfs)

    # Show All IMFs of first wndow
    imfs_for_window = imfs_per_window[0]
    for i, imf in enumerate(imfs_for_window):
        print(f'IMF {i+1}: {imf}')

    # Save first imf of each window and save it to array
    first_imf_per_window = []
    for window_imfs in imfs_per_window:
        first_imf = window_imfs[0]
        first_imf_per_window.append(first_imf)

    IEMG_c2_coeff = []
    MAV_c2_coeff = []
    SSI_c2_coeff = []
    RMS_c2_coeff = []
    VAR_c2_coeff = []
    MYOP_c2_coeff = []
    WL_c2_coeff = []
    DAMV_c2_coeff = []
    M2_c2_coeff = []
    DVARV_c2_coeff = []
    DASDV_c2_coeff = []
    IASD_c2_coeff = []
    IATD_c2_coeff = []
    IEAV_c2_coeff = []
    IALV_c2_coeff = []
    IE_c2_coeff = []
    MIN_c2_coeff = []
    MAX_c2_coeff = []

    IEMG_d1_coeff = []
    MAV_d1_coeff = []
    SSI_d1_coeff = []
    RMS_d1_coeff = []
    VAR_d1_coeff = []
    MYOP_d1_coeff = []
    WL_d1_coeff = []
    DAMV_d1_coeff = []
    M2_d1_coeff = []
    DVARV_d1_coeff = []
    DASDV_d1_coeff = []
    IASD_d1_coeff = []
    IATD_d1_coeff = []
    IEAV_d1_coeff = []
    IALV_d1_coeff = []
    IE_d1_coeff = []
    MIN_d1_coeff = []
    MAX_d1_coeff = []

    IEMG_d2_coeff = []
    MAV_d2_coeff = []
    SSI_d2_coeff = []
    RMS_d2_coeff = []
    VAR_d2_coeff = []
    MYOP_d2_coeff = []
    WL_d2_coeff = []
    DAMV_d2_coeff = []
    M2_d2_coeff = []
    DVARV_d2_coeff = []
    DASDV_d2_coeff = []
    IASD_d2_coeff = []
    IATD_d2_coeff = []
    IEAV_d2_coeff = []
    IALV_d2_coeff = []
    IE_d2_coeff = []
    MIN_d2_coeff = []
    MAX_d2_coeff = []

    wavelet = 'db1'  # You can choose a different wavelet as needed
    # mode = 'symmetric'

    for i, first_imf in enumerate(first_imf_per_window):
        # Perform wavelet transform on each first IMF (2 Level)
        coeffs = pywt.wavedec(first_imf, wavelet, level=2)
        # c2: approximation coefficients at level 2, d2: detail coefficients at level 2, d1: detail coefficients at level 1
        c2, d2, d1 = coeffs[0], coeffs[1], coeffs[2]
        # C4, D4, D3, D2, D1 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]

        IEMG_c2_coeff.append(IEMG(c2))
        MAV_c2_coeff.append(MAV(c2))
        SSI_c2_coeff.append(SSI(c2))
        RMS_c2_coeff.append(RMS(c2))
        VAR_c2_coeff.append(VAR(c2))
        MYOP_c2_coeff.append(MYOP(c2))
        WL_c2_coeff.append(WL(c2))
        DAMV_c2_coeff.append(DAMV(c2))
        M2_c2_coeff.append(M2(c2))
        DVARV_c2_coeff.append(DVARV(c2))
        DASDV_c2_coeff.append(DASDV(c2))
        IASD_c2_coeff.append(IASD(c2))
        IATD_c2_coeff.append(IATD(c2))
        IEAV_c2_coeff.append(IEAV(c2))
        IALV_c2_coeff.append(IALV(c2))
        IE_c2_coeff.append(IE(c2))
        MIN_c2_coeff.append(MIN(c2))
        MAX_c2_coeff.append(MAX(c2))

        IEMG_d1_coeff.append(IEMG(d1))
        MAV_d1_coeff.append(MAV(d1))
        SSI_d1_coeff.append(SSI(d1))
        RMS_d1_coeff.append(RMS(d1))
        VAR_d1_coeff.append(VAR(d1))
        MYOP_d1_coeff.append(MYOP(d1))
        WL_d1_coeff.append(WL(d1))
        DAMV_d1_coeff.append(DAMV(d1))
        M2_d1_coeff.append(M2(d1))
        DVARV_d1_coeff.append(DVARV(d1))
        DASDV_d1_coeff.append(DASDV(d1))
        IASD_d1_coeff.append(IASD(d1))
        IATD_d1_coeff.append(IATD(d1))
        IEAV_d1_coeff.append(IEAV(d1))
        IALV_d1_coeff.append(IALV(d1))
        IE_d1_coeff.append(IE(d1))
        MIN_d1_coeff.append(MIN(d1))
        MAX_d1_coeff.append(MAX(d1))

        IEMG_d2_coeff.append(IEMG(d2))
        MAV_d2_coeff.append(MAV(d2))
        SSI_d2_coeff.append(SSI(d2))
        RMS_d2_coeff.append(RMS(d2))
        VAR_d2_coeff.append(VAR(d2))
        MYOP_d2_coeff.append(MYOP(d2))
        WL_d2_coeff.append(WL(d2))
        DAMV_d2_coeff.append(DAMV(d2))
        M2_d2_coeff.append(M2(d2))
        DVARV_d2_coeff.append(DVARV(d2))
        DASDV_d2_coeff.append(DASDV(d2))
        IASD_d2_coeff.append(IASD(d2))
        IATD_d2_coeff.append(IATD(d2))
        IEAV_d2_coeff.append(IEAV(d2))
        IALV_d2_coeff.append(IALV(d2))
        IE_d2_coeff.append(IE(d2))
        MIN_d2_coeff.append(MIN(d2))
        MAX_d2_coeff.append(MAX(d2))

    return (IEMG_c2_coeff, MAV_c2_coeff, SSI_c2_coeff, RMS_c2_coeff, VAR_c2_coeff, MYOP_c2_coeff, WL_c2_coeff, DAMV_c2_coeff, M2_c2_coeff,
            DVARV_c2_coeff, DASDV_c2_coeff, IASD_c2_coeff, IATD_c2_coeff, IEAV_c2_coeff, IALV_c2_coeff, IE_c2_coeff, MIN_c2_coeff, MAX_c2_coeff,
            IEMG_d1_coeff, MAV_d1_coeff, SSI_d1_coeff, RMS_d1_coeff, VAR_d1_coeff, MYOP_d1_coeff, WL_d1_coeff, DAMV_d1_coeff, M2_d1_coeff,
            DVARV_d1_coeff, DASDV_d1_coeff, IASD_d1_coeff, IATD_d1_coeff, IEAV_d1_coeff, IALV_d1_coeff, IE_d1_coeff, MIN_d1_coeff, MAX_d1_coeff,
            IEMG_d2_coeff, MAV_d2_coeff, SSI_d2_coeff, RMS_d2_coeff, VAR_d2_coeff, MYOP_d2_coeff, WL_d2_coeff, DAMV_d2_coeff, M2_d2_coeff,
            DVARV_d2_coeff, DASDV_d2_coeff, IASD_d2_coeff, IATD_d2_coeff, IEAV_d2_coeff, IALV_d2_coeff, IE_d2_coeff, MIN_d2_coeff, MAX_d2_coeff)


TT_Files = ['T-T1.csv', 'T-T2.csv', 'T-T3.csv',
            'T-T4.csv', 'T-T5.csv', 'T-T6.csv',]

TR_Files = ['T-R1.csv', 'T-R2.csv', 'T-R3.csv',
            'T-R4.csv', 'T-R5.csv', 'T-R6.csv',]

TM_Files = ['T-M1.csv', 'T-M2.csv', 'T-M3.csv',
            'T-M4.csv', 'T-M5.csv', 'T-M6.csv',]

TL_Files = ['T-L1.csv', 'T-L2.csv', 'T-L3.csv',
            'T-L4.csv', 'T-L5.csv', 'T-L6.csv',]

TI_Files = ['T-I1.csv', 'T-I2.csv', 'T-I3.csv',
            'T-I4.csv', 'T-I5.csv', 'T-I6.csv',]

RR_Files = ['R-R1.csv', 'R-R2.csv', 'R-R3.csv',
            'R-R4.csv', 'R-R5.csv', 'R-R6.csv',]

MM_Files = ['M-M1.csv', 'M-M2.csv', 'M-M3.csv',
            'M-M4.csv', 'M-M5.csv', 'M-M6.csv',]

LL_Files = ['L-L1.csv', 'L-L2.csv', 'L-L3.csv',
            'L-L4.csv', 'L-L5.csv', 'L-L6.csv',]

II_Files = ['I-I1.csv', 'I-I2.csv', 'I-I3.csv',
            'I-I4.csv', 'I-I5.csv', 'I-I6.csv',]

HC_Files = ['HC-1.csv', 'HC-2.csv', 'HC-3.csv',
            'HC-4.csv', 'HC-5.csv', 'HC-6.csv']

Movement_Types = ['HC', 'II', 'LL', 'MM', 'RR', 'TI', 'TL', 'TM', 'TR', 'TT']

Person_Files = [HC_Files, II_Files, LL_Files, MM_Files,
                RR_Files, TI_Files, TL_Files, TM_Files, TR_Files, TT_Files]

Persons = ['EMG-S1', 'EMG-S2', 'EMG-S4', 'EMG-S5',
           'EMG-S6', 'EMG-S8', 'EMG-S9', 'EMG-S10']

# Output root folder
output_root_folder = 'output'

for person in Persons:
    # Create a folder for each person
    person_output_folder = os.path.join(output_root_folder, person)
    os.makedirs(person_output_folder, exist_ok=True)

    for movement_type in Movement_Types:
        # Create a folder for each movement type inside the person's folder
        movement_output_folder = os.path.join(
            person_output_folder, movement_type)
        os.makedirs(movement_output_folder, exist_ok=True)

        all_c2 = []
        all_d1 = []
        all_d2 = []

        for movement_file in Person_Files[Movement_Types.index(movement_type)]:
            input_csv_path = f'input/{person}/{movement_file}'

            extracted_features = process_emg_csv(input_csv_path)

            # Check if the function returned valid features
            if extracted_features is not None:
                (IEMG_c2_coeff, MAV_c2_coeff, SSI_c2_coeff, RMS_c2_coeff, VAR_c2_coeff, MYOP_c2_coeff, WL_c2_coeff, DAMV_c2_coeff, M2_c2_coeff,
                 DVARV_c2_coeff, DASDV_c2_coeff, IASD_c2_coeff, IATD_c2_coeff, IEAV_c2_coeff, IALV_c2_coeff, IE_c2_coeff, MIN_c2_coeff, MAX_c2_coeff,
                 IEMG_d1_coeff, MAV_d1_coeff, SSI_d1_coeff, RMS_d1_coeff, VAR_d1_coeff, MYOP_d1_coeff, WL_d1_coeff, DAMV_d1_coeff, M2_d1_coeff,
                 DVARV_d1_coeff, DASDV_d1_coeff, IASD_d1_coeff, IATD_d1_coeff, IEAV_d1_coeff, IALV_d1_coeff, IE_d1_coeff, MIN_d1_coeff, MAX_d1_coeff,
                 IEMG_d2_coeff, MAV_d2_coeff, SSI_d2_coeff, RMS_d2_coeff, VAR_d2_coeff, MYOP_d2_coeff, WL_d2_coeff, DAMV_d2_coeff, M2_d2_coeff,
                 DVARV_d2_coeff, DASDV_d2_coeff, IASD_d2_coeff, IATD_d2_coeff, IEAV_d2_coeff, IALV_d2_coeff, IE_d2_coeff, MIN_d2_coeff, MAX_d2_coeff) = extracted_features

                # Append the features to the respective lists
                all_c2.extend(zip(IEMG_c2_coeff, MAV_c2_coeff, SSI_c2_coeff, RMS_c2_coeff, VAR_c2_coeff, MYOP_c2_coeff, WL_c2_coeff, DAMV_c2_coeff, M2_c2_coeff,
                                  DVARV_c2_coeff, DASDV_c2_coeff, IASD_c2_coeff, IATD_c2_coeff, IEAV_c2_coeff, IALV_c2_coeff, IE_c2_coeff, MIN_c2_coeff, MAX_c2_coeff))

                all_d1.extend(zip(IEMG_d1_coeff, MAV_d1_coeff, SSI_d1_coeff, RMS_d1_coeff, VAR_d1_coeff, MYOP_d1_coeff, WL_d1_coeff, DAMV_d1_coeff, M2_d1_coeff,
                                  DVARV_d1_coeff, DASDV_d1_coeff, IASD_d1_coeff, IATD_d1_coeff, IEAV_d1_coeff, IALV_d1_coeff, IE_d1_coeff, MIN_d1_coeff, MAX_d1_coeff))

                all_d2.extend(zip(IEMG_d2_coeff, MAV_d2_coeff, SSI_d2_coeff, RMS_d2_coeff, VAR_d2_coeff, MYOP_d2_coeff, WL_d2_coeff, DAMV_d2_coeff, M2_d2_coeff,
                                  DVARV_d2_coeff, DASDV_d2_coeff, IASD_d2_coeff, IATD_d2_coeff, IEAV_d2_coeff, IALV_d2_coeff, IE_d2_coeff, MIN_d2_coeff, MAX_d2_coeff))

        # Save features for c2 to a CSV file
        c2_csv_file_path = os.path.join(
            movement_output_folder, f'all_{movement_type}_c2_features.csv')
        with open(c2_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            features_header = ['IEMG', 'MAV', 'SSI', 'RMS', 'VAR', 'MYOP', 'WL', 'DAMV',
                               'M2', 'DVARV', 'DASDV', 'IASD', 'IATD', 'IEAV', 'IALV', 'IE', 'MIN', 'MAX']
            writer.writerow(features_header)
            writer.writerows(all_c2)

        # Save features for d1 to a CSV file
        d1_csv_file_path = os.path.join(
            movement_output_folder, f'all_{movement_type}_d1_features.csv')
        with open(d1_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(features_header)
            writer.writerows(all_d1)

        # Save features for d2 to a CSV file
        d2_csv_file_path = os.path.join(
            movement_output_folder, f'all_{movement_type}_d2_features.csv')
        with open(d2_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(features_header)
            writer.writerows(all_d2)
