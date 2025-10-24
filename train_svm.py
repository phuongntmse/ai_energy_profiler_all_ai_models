import os
import warnings
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import argparse
from config.my_config import *
from common_function import *

warnings.simplefilter("ignore", category=RuntimeWarning)

# Argument Parser
parser = argparse.ArgumentParser(description="Train SVM models for workload classification")
parser.add_argument("--kernel", default='rbf', help="SVM kernel types")
parser.add_argument("--C", type=int, default=10, help="SVM regularization parameters")
parser.add_argument("--gamma", default='scale', help="SVM gamma parameter")
args = parser.parse_args()
l_kernel = args.kernel
l_c = args.C
l_gamma = args.gamma

# Constants
model_type = "svm"

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

		# Split the dataset
		X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42)
		
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		# Train the SVM classifier
		svm_model = SVC(kernel=l_kernel, C=l_c, gamma=l_gamma, random_state=42,probability=True)
		svm_model.fit(X_train_scaled, y_train)

		# Make predictions
		y_pred = svm_model.predict(X_test_scaled)

		# Evaluate the model
		print(group_name, " - Accuracy:", accuracy_score(y_test, y_pred))
		print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
		
		os.makedirs(MODEL_DIR, exist_ok=True)
		joblib.dump(svm_model, f"{MODEL_DIR}/model_{model_type}_{group_name}_{l_kernel}_{l_c}_{l_gamma}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
		joblib.dump(scaler, f"{MODEL_DIR}/scaler_{model_type}_{group_name}_{l_kernel}_{l_c}_{l_gamma}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
		joblib.dump(list(df_X.columns), f"{MODEL_DIR}/metrics_{model_type}_{group_name}_{l_kernel}_{l_c}_{l_gamma}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
