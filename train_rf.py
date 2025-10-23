import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
from sklearn.model_selection import train_test_split
import warnings
import argparse
from my_config import *
from common_function import *

warnings.simplefilter("ignore", category=RuntimeWarning)

# Argument Parser
parser = argparse.ArgumentParser(description="Train RF model with hyperparameters")
parser.add_argument("--trees", type=int, default=100, help="Number of decision trees in the Random Forest")
args = parser.parse_args()
n_trees = args.trees

# Constants
model_type = "rf"

# Load folders
folders = load_training_folders(TRAINING_DATA_CONFIG)

for group_name, selected_columns in FEATURE_GROUPS.items():
	X, y = [], []
	for folder in folders:
		for file in os.listdir(folder):
			if file.endswith(".csv"):
				full_path = os.path.join(folder, file)
				df = pd.read_csv(full_path)
				# Ensure all required features exist, fill missing with 0
				missing_cols = [col for col in selected_columns if col not in df.columns]
				# Add dummy columns for missing features
				for col in missing_cols:
					df[col] = 0  # or use np.nan if needed

				feats = extract_features(df, selected_columns)
				label = get_label(file)
				if feats:
					X.append(feats)
					y.append(label)
	if X:
		df_X = pd.DataFrame(X)
		# Handle missing values with mean values
		df_X = df_X.fillna(df_X.mean()) 
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(df_X)

		# Split into training & testing sets
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

		# Train the model
		model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
		model.fit(X_train, y_train)

		joblib.dump(model, f"{MODEL_DIR}/model_{model_type}_{group}_{RF_NO_DECISION_TREES}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
		joblib.dump(scaler, f"{MODEL_DIR}/scaler_{model_type}_{group}_{RF_NO_DECISION_TREES}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
		joblib.dump(list(df_X.columns), f"{MODEL_DIR}/metrics_{model_type}_{group}_{RF_NO_DECISION_TREES}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")

		# Evaluate model
		accuracy = model.score(X_test, y_test)
		print(f"Model {group_name} trained successfully! Accuracy: {accuracy:.2f}")
