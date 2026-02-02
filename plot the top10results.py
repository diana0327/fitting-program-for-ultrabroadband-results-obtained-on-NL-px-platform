import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

# Constants for molecular vibrations
wr1, wr2, wr3, wr4, wr5, wr6 = 995, 1003.15, 1011, 1038.61, 1080.9, 1188
gamma1, gamma2, gamma3, gamma4, gamma5, gamma6 = 5, 5, 4, 5, 5, 11
Plasmon_Center = 1081.15

# Model function with plasmonic resonance
def model_function_with_plasmon(x, Ampl1, Ampl2, Ampl3, Ampl4, Ampl5, Ampl6, Phase1, Phase2, Phase3, Phase4, Phase5, Phase6,
                                 Plasmon_Amp, Plasmon_Width, Plasmon_Phase):
    epsilon = 1e-8  # Small value to prevent division by zero

    # Molecular vibrational contributions
    Xr_SFG = (Ampl1 * np.exp(1j * Phase1) / (wr1 - x - 1j * gamma1 / 2) +
              Ampl2 * np.exp(1j * Phase2) / (wr2 - x - 1j * gamma2 / 2) +
              Ampl3 * np.exp(1j * Phase3) / (wr3 - x - 1j * gamma3 / 2) +
              Ampl4 * np.exp(1j * Phase4) / (wr4 - x - 1j * gamma4 / 2) +
              Ampl5 * np.exp(1j * Phase5) / (wr5 - x - 1j * gamma5 / 2) +
              Ampl6 * np.exp(1j * Phase6) / (wr6 - x - 1j * gamma6 / 2))

    Xr_DFG = np.conj(Xr_SFG)  # DFG contribution as conjugate

    # Frequency-dependent non-resonant contribution with plasmonic resonance
    Xnr = (Plasmon_Amp * np.exp(1j * Plasmon_Phase) / 
           (Plasmon_Center - x + 1j * Plasmon_Width / 2)) + 0.9

    # Total susceptibilities
    X_SFG = Xr_SFG + Xnr
    X_DFG = Xr_DFG + Xnr

    modulus_SFG = np.abs(X_SFG) ** 2
    modulus_DFG = np.abs(X_DFG) ** 2

    # Ratios assuming plasmonic effects cancel out
    ratio = modulus_SFG / (modulus_DFG + epsilon)

    return modulus_SFG, modulus_DFG, ratio

# Function to load and process two files
def process_files_and_plot():
    # Hide the root Tkinter window
    Tk().withdraw()

    # Prompt user to select the raw data file
    print("Select the raw data file:")
    raw_data_filename = askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if not raw_data_filename:
        print("No raw data file selected.")
        return

    # Prompt user to select the fitting parameters file
    print("Select the fitting parameters file:")
    params_filename = askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if not params_filename:
        print("No fitting parameters file selected.")
        return

    # Load raw data
    raw_data = pd.read_excel(raw_data_filename)
    if not all(col in raw_data.columns for col in ['x', 'SFG', 'DFG']):
        print("The raw data file does not contain required columns: ['x', 'SFG', 'DFG']")
        return

    x_data = raw_data['x'].values
    SFG_raw = raw_data['SFG'].values
    DFG_raw = raw_data['DFG'].values
    intensity_raw = raw_data['intensity'].values
    # Load fitting parameters
    params_data = pd.read_excel(params_filename)
    required_columns = ['Ampl1', 'Ampl2', 'Ampl3', 'Ampl4', 'Ampl5', 'Ampl6',
                        'Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5', 'Phase6',
                        'Plasmon_Amp', 'Plasmon_Width', 'Plasmon_Phase']
    if not all(col in params_data.columns for col in required_columns):
        print(f"The fitting parameters file does not contain required columns: {required_columns}")
        return

    # Extract the top 10 parameter sets
    top_10_params = params_data.head(1:5:10)

    # Plot all sets of parameters in the same figure for SFG
    plt.figure(figsize=(10, 7))
    plt.plot(x_data, SFG_raw, label='Raw SFG', linestyle='-', marker='o', linewidth=2)
    for idx, params in top_10_params.iterrows():
        modulus_SFG, _, _ = model_function_with_plasmon(
            x_data,
            params['Ampl1'], params['Ampl2'], params['Ampl3'], params['Ampl4'], params['Ampl5'], params['Ampl6'],
            params['Phase1'], params['Phase2'], params['Phase3'], params['Phase4'], params['Phase5'], params['Phase6'],
            params['Plasmon_Amp'], params['Plasmon_Width'], params['Plasmon_Phase']
        )
        plt.plot(x_data, modulus_SFG, label=f'Modeled SFG (Set {idx+1})', linestyle='--')
    plt.title('SFG Signal: Raw vs Modeled (Top 10 Sets)')
    plt.xlabel('Frequency (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot all sets of parameters in the same figure for DFG
    plt.figure(figsize=(10, 7))
    plt.plot(x_data, DFG_raw, label='Raw DFG', linestyle='-', marker='o', linewidth=2)
    for idx, params in top_10_params.iterrows():
        _, modulus_DFG, _ = model_function_with_plasmon(
            x_data,
            params['Ampl1'], params['Ampl2'], params['Ampl3'], params['Ampl4'], params['Ampl5'], params['Ampl6'],
            params['Phase1'], params['Phase2'], params['Phase3'], params['Phase4'], params['Phase5'], params['Phase6'],
            params['Plasmon_Amp'], params['Plasmon_Width'], params['Plasmon_Phase']
        )
        plt.plot(x_data, modulus_DFG, label=f'Modeled DFG (Set {idx+1})', linestyle='--')
    plt.title('DFG Signal: Raw vs Modeled (Top 10 Sets)')
    plt.xlabel('Frequency (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure(figsize=(10, 7))
    plt.plot(x_data, intensity_raw, label='Raw ratio', linestyle='-', marker='o', linewidth=2)
    for idx, params in top_10_params.iterrows():
        _, ratio , _ = model_function_with_plasmon(
            x_data,
            params['Ampl1'], params['Ampl2'], params['Ampl3'], params['Ampl4'], params['Ampl5'], params['Ampl6'],
            params['Phase1'], params['Phase2'], params['Phase3'], params['Phase4'], params['Phase5'], params['Phase6'],
            params['Plasmon_Amp'], params['Plasmon_Width'], params['Plasmon_Phase']
        )
        plt.plot(x_data, modulus_DFG, label=f'Modeled DFG (Set {idx+1})', linestyle='--')
    plt.title('sfg/dfg: Raw vs Modeled (Top 10 Sets)')
    plt.xlabel('Frequency (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid()
    plt.show()    
# Run the function
process_files_and_plot()
