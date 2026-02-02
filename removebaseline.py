# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Baseline Calculation and Normalization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Load data
#path = r'\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\allpeaks_NP12.xlsx'
path = r'\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\forthefirst3peaks\New folder\testwithdifferentphases\1600samplefitting\1080baselinec.xlsx'

df = pd.read_excel(path)

# Extract data
x_data = df['x'].values
intensity_data = df['intensity'].values
SFG_data = df['SFG'].values
DFG_data = df['DFG'].values

# Define a function for baseline calculation using polynomial fitting
def calculate_baseline(x, y, degree=3):
    coeffs = np.polyfit(x, y, degree)  # Fit a polynomial
    baseline = np.polyval(coeffs, x)  # Calculate baseline values
    return baseline

# Calculate baselines for SFG and DFG
SFG_baseline = calculate_baseline(x_data, SFG_data, degree=2)
DFG_baseline = calculate_baseline(x_data, DFG_data, degree=2)
ratio_baseline=calculate_baseline(x_data, intensity_data, degree=2)


# Normalize the data by dividing by their respective baselines
epsilon = 1e-8 # Prevent division by zero
SFG_normalized = SFG_data / (SFG_baseline + epsilon)
DFG_normalized = DFG_data / (DFG_baseline + epsilon)
SFG_DFG_ratio = intensity_data/(ratio_baseline + epsilon)

# Calculate the SFG/DFG ratio
# = SFG_normalized / (DFG_normalized + epsilon)

# Plot the original data with baselines
plt.figure(figsize=(10, 6))

# Plot SFG data with its baseline
plt.subplot(3, 1, 1)
plt.plot(x_data, SFG_data, label='Original SFG Data', color='blue')
plt.plot(x_data, SFG_baseline, label='SFG Baseline', color='red', linestyle='--')

plt.title('SFG Data with Baseline')
plt.xlabel('X Data')
plt.ylabel('Intensity')
plt.legend()

# Plot DFG data with its baseline
plt.subplot(3, 1, 2)
plt.plot(x_data, DFG_data, label='Original DFG Data', color='blue')
plt.plot(x_data, DFG_baseline, label='DFG Baseline', color='red', linestyle='--')

plt.title('DFG Data with Baseline')
plt.xlabel('X Data')
plt.ylabel('Intensity')
plt.legend()

# Plot SFG/DFG ratio
plt.subplot(3, 1, 3)
plt.plot(x_data, SFG_DFG_ratio, label='SFG/DFG Ratio', color='green')
plt.title('Normalized SFG/DFG Ratio')
plt.xlabel('X Data')
plt.ylabel('Ratio')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the normalized SFG, DFG, and their ratio for comparison
plt.figure(figsize=(10, 6))
plt.plot(x_data, SFG_normalized, label='Normalized SFG', color='blue')
plt.plot(x_data, DFG_normalized+1, label='Normalized DFG', color='red')
plt.plot(x_data, SFG_DFG_ratio+2, label='SFG/DFG Ratio', color='green')
plt.title('Normalized Data and SFG/DFG Ratio')
plt.xlabel('X Data')
plt.ylabel('Normalized Values')
plt.legend()
plt.grid()
plt.show()

# Calculate similarity metrics between SFG and DFG normalized data
correlation, p_value = pearsonr(SFG_normalized, DFG_normalized)
mse = mean_squared_error(SFG_normalized, DFG_normalized)

print(f"Correlation coefficient between normalized SFG and DFG data: {correlation:.3f}")
print(f"Mean Squared Error (MSE) between normalized SFG and DFG data: {mse:.3f}")

x = np.array(x_data).flatten()
sfg = np.array(SFG_normalized).flatten()
dfg = np.array(DFG_normalized).flatten()
ratio = np.array(SFG_DFG_ratio).flatten()

# 合并为 DataFrame
df = pd.DataFrame({
    'x': x,
    'SFG': sfg,
    'DFG': dfg,
    'intensity': ratio
})

# 保存为 Excel 文件
df.to_excel(r'\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\forthefirst3peaks\New folder\testwithdifferentphases\1600samplefitting\1080doubleclean.xlsx', index=False)


"""
Created on Tue Oct 22 15:21:12 2024

@author: zhxie


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Load data
path = r'\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\all5peaksforfitting.xlsx'
df = pd.read_excel(path)

# Extract data
x_data = df['x'].values
intensity_data = df['intensity'].values
SFG_data = df['SFG'].values
DFG_data = df['DFG'].values

# Define a function for baseline removal using polynomial fitting
def baseline_removal(x, y, degree=2):
    # Fit a polynomial to the data (baseline approximation)
    coeffs = np.polyfit(x, y, degree)
    baseline = np.polyval(coeffs, x)
    
    # Subtract the baseline from the original data
    y_baseline_removed = y - baseline
    return y_baseline_removed, baseline

# Remove the baseline from the intensity data
intensity_baseline_removed, intensity_baseline = baseline_removal(x_data, intensity_data)
SFG_baseline_removed, SFG_baseline = baseline_removal(x_data, SFG_data)
DFG_baseline_removed, DFG_baseline = baseline_removal(x_data, DFG_data)

# Small epsilon value to prevent division by zero
epsilon = 1e-5

# Exclude the last data point from DFG data and baseline for normalization
DFG_data_excluded = DFG_data[0:198]
DFG_baseline_excluded = DFG_baseline[0:198]
x_data_excluded = x_data[0:198]

# Calculate normalized data by dividing each data point by its baseline
SFG_normalized = SFG_data / (SFG_baseline + epsilon)
DFG_normalized = DFG_data_excluded / (DFG_baseline_excluded + epsilon)

# Plot the normalized SFG and DFG data
plt.figure(figsize=(10, 6))

# Plot SFG normalized data
plt.subplot(2, 1, 1)
plt.plot(x_data, SFG_normalized, label='Normalized SFG Data', color='blue')
plt.title('Normalized SFG Data (SFG / SFG_baseline)')
plt.xlabel('X Data')
plt.ylabel('SFG / Baseline')
plt.legend()

# Plot DFG normalized data (excluding last point)
plt.subplot(2, 1, 2)
plt.plot(x_data_excluded, DFG_normalized, label='Normalized DFG Data (excluding last point)', color='red')
plt.title('Normalized DFG Data (DFG / DFG_baseline, excluding last point)')
plt.xlabel('X Data')
plt.ylabel('DFG / Baseline')
plt.legend()

plt.tight_layout()
plt.show()

# Additional comparison of the similarity between SFG and DFG normalized data

# 1. Plot SFG and DFG normalized data together
plt.figure(figsize=(10, 4))
plt.plot(x_data, SFG_normalized, label='Normalized SFG', color='blue')
plt.plot(x_data_excluded, DFG_normalized, label='Normalized DFG (excluding last point)', color='red')
plt.title('Comparison of Normalized SFG and DFG Data')
plt.xlabel('X Data')
plt.ylabel('Normalized Values (Data / Baseline)')
plt.legend()
plt.show()

# 2. Calculate the correlation between the normalized SFG and DFG data (excluding last point)
correlation, p_value = pearsonr(SFG_normalized, DFG_normalized)
print(f"Correlation coefficient between normalized SFG and DFG data (excluding last point): {correlation:.3f}")

# 3. Calculate the mean squared error between the normalized SFG and DFG data (excluding last point)
mse = mean_squared_error(SFG_normalized, DFG_normalized)
print(f"Mean Squared Error (MSE) between normalized SFG and DFG data (excluding last point): {mse:.3f}")


# -*- coding: utf-8 -*-

Created on Tue Oct 22 15:21:12 2024

@author: zhxie


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Load data
path = r'\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\all5peaksforfitting.xlsx'
df = pd.read_excel(path)

# Extract data
x_data = df['x'].values
intensity_data = df['intensity'].values
SFG_data = df['SFG'].values
DFG_data = df['DFG'].values

# Define a function for baseline removal while maintaining original average
def baseline_removal_with_mean_adjustment(x, y, degree=2):
    # Fit a polynomial to the data (baseline approximation)
    coeffs = np.polyfit(x, y, degree)
    baseline = np.polyval(coeffs, x)
    
    # Subtract the baseline from the original data
    y_baseline_removed = y - baseline
    
    # Calculate the original mean and the new mean after baseline removal
    original_mean = np.mean(y)
    corrected_mean = np.mean(y_baseline_removed)
    
    # Adjust the baseline-removed data to retain the original mean
    y_baseline_adjusted = y_baseline_removed + (original_mean - corrected_mean)
    
    return y_baseline_adjusted, baseline

# Apply baseline removal with mean adjustment
intensity_baseline_removed, intensity_baseline = baseline_removal_with_mean_adjustment(x_data, intensity_data)
SFG_baseline_removed, SFG_baseline = baseline_removal_with_mean_adjustment(x_data, SFG_data)
DFG_baseline_removed, DFG_baseline = baseline_removal_with_mean_adjustment(x_data, DFG_data)

# Small epsilon value to prevent division by zero
epsilon = 1e-10

# Exclude the last data point from DFG data and baseline for normalization
DFG_data_excluded = DFG_data[:-1]
DFG_baseline_excluded = DFG_baseline[:-1]
x_data_excluded = x_data[:-1]

# Calculate normalized data by dividing each data point by its baseline
SFG_normalized = SFG_data / (SFG_baseline + epsilon)
DFG_normalized = DFG_data_excluded / (DFG_baseline_excluded + epsilon)

# Plot the adjusted baseline-corrected data
plt.figure(figsize=(10, 6))

# Plot intensity data with adjusted baseline removed
plt.subplot(3, 1, 1)
plt.plot(x_data, intensity_data, label='Original SFG/DFG Data', color='blue')
plt.plot(x_data, intensity_baseline, label='Estimated Baseline', color='red', linestyle='--')
plt.plot(x_data, intensity_baseline_removed, label='Baseline Removed (Adjusted)', color='green')
plt.title('SFG/DFG Data: Baseline Removal with Mean Adjustment')
plt.legend()

# Plot SFG data with adjusted baseline removed
plt.subplot(3, 1, 2)
plt.plot(x_data, SFG_data, label='Original SFG Data', color='blue')
plt.plot(x_data, SFG_baseline, label='Estimated Baseline', color='red', linestyle='--')
plt.plot(x_data, SFG_baseline_removed, label='Baseline Removed (Adjusted)', color='green')
plt.title('SFG Data: Baseline Removal with Mean Adjustment')
plt.legend()

# Plot DFG data with adjusted baseline removed
plt.subplot(3, 1, 3)
plt.plot(x_data, DFG_data, label='Original DFG Data', color='blue')
plt.plot(x_data, DFG_baseline, label='Estimated Baseline', color='red', linestyle='--')
plt.plot(x_data, DFG_baseline_removed, label='Baseline Removed (Adjusted)', color='green')
plt.title('DFG Data: Baseline Removal with Mean Adjustment')
plt.legend()

plt.tight_layout()
plt.show()

# Additional comparison of the similarity between SFG and DFG normalized data
plt.figure(figsize=(10, 4))
plt.plot(x_data, SFG_normalized, label='Normalized SFG', color='blue')
plt.plot(x_data_excluded, DFG_normalized, label='Normalized DFG (excluding last point)', color='red')
plt.title('Comparison of Normalized SFG and DFG Data')
plt.xlabel('X Data')
plt.ylabel('Normalized Values (Data / Baseline)')
plt.legend()
plt.show()

# Calculate the correlation between the normalized SFG and DFG data (excluding last point)
correlation, p_value = pearsonr(SFG_normalized[:-1], DFG_normalized)
print(f"Correlation coefficient between normalized SFG and DFG data (excluding last point): {correlation:.3f}")

# Calculate the mean squared error between the normalized SFG and DFG data (excluding last point)
mse = mean_squared_error(SFG_normalized[:-1], DFG_normalized)
print(f"Mean Squared Error (MSE) between normalized SFG and DFG data (excluding last point): {mse:.3f}")

"""
