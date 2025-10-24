import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, linregress
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Load selected folders
def load_training_folders(setup_file):
	setup_df = pd.read_csv(setup_file)

	if "select" in setup_df.columns:
		setup_df = setup_df[setup_df["select"].str.lower().isin(["yes", "1"])]
	
	return setup_df["folder"].tolist()

def get_label(filename):
	if "cpu" in filename.lower() and "memory" in filename.lower():
		return "CPU-Memory-mixed"
	elif "cpu" in filename.lower():
		return "CPU intensive"
	elif "memory" in filename.lower():
		return "Memory intensive"
	elif "disk" in filename.lower():
		return "Disk intensive"
	elif "network" in filename.lower():
		return "Network intensive"
	else:
		return "N/A"  # Skip files that don't match expected labels

def extract_features(df, selected_columns=None):
	"""Extracts statistical, trend, frequency, and shape-based features from time-series data."""
	features = {}
	if selected_columns is None:
		selected_columns = df.columns
	# selected_columns = load_selected_columns(config_path)

	for col in selected_columns:
		# data = df[col].dropna()  # Remove NaN values
		data = df[col].fillna(df[col].mean())  # Replace NaNs with column mean

		if len(data) > 1:  # Ensure enough data points
			# Basic Statistics
			features[f"{col}_mean"] = data.mean()
			features[f"{col}_std"] = data.std()
			features[f"{col}_min"] = data.min()
			features[f"{col}_max"] = data.max()
			features[f"{col}_median"] = data.median()
			features[f"{col}_iqr"] = data.quantile(0.75) - data.quantile(0.25)
			features[f"{col}_skew"] = skew(data)
			features[f"{col}_kurtosis"] = kurtosis(data)

			# Trend Features (Linear Regression)
			x = np.arange(len(data))
			slope, _, _, _, _ = linregress(x, data)
			features[f"{col}_trend"] = slope

			# Rolling Window Features
			features[f"{col}_rolling_mean"] = data.rolling(window=5).mean().mean()
			features[f"{col}_rolling_var"] = data.rolling(window=5).var().mean()

			# Frequency-Domain Features (FFT Peak & Spectral Entropy)
			fft_result = np.abs(fft(data.values))
			features[f"{col}_fft_peak"] = np.max(fft_result)
			features[f"{col}_spectral_entropy"] = entropy(fft_result)

			# Shape-Based Features (Peaks, Valleys)
			peaks, _ = find_peaks(data, height=np.mean(data))
			valleys, _ = find_peaks(-data, height=-np.mean(data))
			features[f"{col}_num_peaks"] = len(peaks)
			features[f"{col}_num_valleys"] = len(valleys)

			# Peak Characteristics
			if len(peaks) > 0:
				features[f"{col}_peak_max_height"] = data.iloc[peaks].max()
				features[f"{col}_peak_width"] = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
			else:
				# Ensure features exist even if no peaks are detected
				features[f"{col}_peak_max_height"] = 0  # Default value if no peaks
				features[f"{col}_peak_width"] = 0  # Default width if no peaks

	return features