import json
import numpy as np
from scipy.stats import kurtosis, skew, iqr, scoreatpercentile
from sklearn.preprocessing import MinMaxScaler
import csv

# Load your JSON data from the file (replace 'your_gyroscope_data_file.json' with the actual file name)
with open('/content/gyroscope.json', 'r') as file:
    data = [json.loads(line) for line in file]

# Define a function to extract features for the gyroscope
def extract_gyroscope_features(window_data):
    gyroscope_features = {}

    # Check if there is data for the gyroscope
    if any("Gyroscope" in entry["SensorName"] for entry in window_data):
        # Extract gyroscope data
        gyroscope_x_values = [float(entry["x"]) for entry in window_data if "Gyroscope" in entry["SensorName"]]
        gyroscope_y_values = [float(entry["y"]) for entry in window_data if "Gyroscope" in entry["SensorName"]]
        gyroscope_z_values = [float(entry["z"]) for entry in window_data if "Gyroscope" in entry["SensorName"]]

        # Normalize gyroscope data
        scaler = MinMaxScaler()
        gyroscope_x_values = scaler.fit_transform(np.array(gyroscope_x_values).reshape(-1, 1)).flatten()
        gyroscope_y_values = scaler.fit_transform(np.array(gyroscope_y_values).reshape(-1, 1)).flatten()
        gyroscope_z_values = scaler.fit_transform(np.array(gyroscope_z_values).reshape(-1, 1)).flatten()

        # Calculate features for each axis
        gyroscope_features["mean_x"] = np.mean(gyroscope_x_values)
        gyroscope_features["mean_y"] = np.mean(gyroscope_y_values)
        gyroscope_features["mean_z"] = np.mean(gyroscope_z_values)
        gyroscope_features["std_x"] = np.std(gyroscope_x_values)
        gyroscope_features["std_y"] = np.std(gyroscope_y_values)
        gyroscope_features["std_z"] = np.std(gyroscope_z_values)
        gyroscope_features["min_x"] = np.min(gyroscope_x_values)
        gyroscope_features["min_y"] = np.min(gyroscope_y_values)
        gyroscope_features["min_z"] = np.min(gyroscope_z_values)
        gyroscope_features["max_x"] = np.max(gyroscope_x_values)
        gyroscope_features["max_y"] = np.max(gyroscope_y_values)
        gyroscope_features["max_z"] = np.max(gyroscope_z_values)
        gyroscope_features["kurtosis_x"] = kurtosis(gyroscope_x_values)
        gyroscope_features["kurtosis_y"] = kurtosis(gyroscope_y_values)
        gyroscope_features["kurtosis_z"] = kurtosis(gyroscope_z_values)
        gyroscope_features["skewness_x"] = skew(gyroscope_x_values)
        gyroscope_features["skewness_y"] = skew(gyroscope_y_values)
        gyroscope_features["skewness_z"] = skew(gyroscope_z_values)
        gyroscope_features["iqr_x"] = iqr(gyroscope_x_values)
        gyroscope_features["iqr_y"] = iqr(gyroscope_y_values)
        gyroscope_features["iqr_z"] = iqr(gyroscope_z_values)
        gyroscope_features["abs_energy_x"] = np.sum(np.abs(gyroscope_x_values) ** 2)
        gyroscope_features["abs_energy_y"] = np.sum(np.abs(gyroscope_y_values) ** 2)
        gyroscope_features["abs_energy_z"] = np.sum(np.abs(gyroscope_z_values) ** 2)
        gyroscope_features["mean_abs_diff_x"] = np.mean(np.abs(np.diff(gyroscope_x_values)))
        gyroscope_features["mean_abs_diff_y"] = np.mean(np.abs(np.diff(gyroscope_y_values)))
        gyroscope_features["mean_abs_diff_z"] = np.mean(np.abs(np.diff(gyroscope_z_values)))
        gyroscope_features["median_x"] = np.median(gyroscope_x_values)
        gyroscope_features["median_y"] = np.median(gyroscope_y_values)
        gyroscope_features["median_z"] = np.median(gyroscope_z_values)
        gyroscope_features["variance_x"] = np.var(gyroscope_x_values)
        gyroscope_features["variance_y"] = np.var(gyroscope_y_values)
        gyroscope_features["variance_z"] = np.var(gyroscope_z_values)
        gyroscope_features["percentile_25_x"] = scoreatpercentile(gyroscope_x_values, 25)
        gyroscope_features["percentile_25_y"] = scoreatpercentile(gyroscope_y_values, 25)
        gyroscope_features["percentile_25_z"] = scoreatpercentile(gyroscope_z_values, 25)
        gyroscope_features["percentile_75_x"] = scoreatpercentile(gyroscope_x_values, 75)
        gyroscope_features["percentile_75_y"] = scoreatpercentile(gyroscope_y_values, 75)
        gyroscope_features["percentile_75_z"] = scoreatpercentile(gyroscope_z_values, 75)
        gyroscope_features["rms_x"] = np.sqrt(np.mean(gyroscope_x_values**2))
        gyroscope_features["rms_y"] = np.sqrt(np.mean(gyroscope_y_values**2))
        gyroscope_features["rms_z"] = np.sqrt(np.mean(gyroscope_z_values**2))

    return gyroscope_features

# Set a constant label value for all windows
constant_label = 5  # Replace with your desired label value

# Process the data in windows (you can change the window size and step size)
window_size = 50
step_size = 25
windows_with_labels = []

for i in range(0, len(data), step_size):
    window = data[i:i + window_size]
    gyroscope_features = extract_gyroscope_features(window)
    windows_with_labels.append((gyroscope_features, constant_label))

# Specify the existing output CSV file name (replace with your existing CSV file path)
output_file = '/content/extended_gyroscope_features_with_labels.csv'

# Append the data to the CSV file
with open(output_file, mode='a', newline='') as csvfile:
    fieldnames = list(gyroscope_features.keys()) + ['label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Append the features and labels
    for features, label in windows_with_labels:
        features['label'] = label
        writer.writerow(features)

print(f"Normalized gyroscope features with a constant label have been appended to '{output_file}'.")