import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import argparse
from config.my_config import *
from common_function import *

warnings.simplefilter("ignore", category=RuntimeWarning)

# Argument Parser
parser = argparse.ArgumentParser(description="Train NN models for workload classification")
parser.add_argument("--layers", type=int, default=2, help="Number of layers")
parser.add_argument("--nodes", type=int, default=32, help="Number of nodes per layer")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
parser.add_argument("--activation", default='relu', help="Hidden layer activation function")
parser.add_argument("--output_activation", default='softmax', help="Output layer activation function")
args = parser.parse_args()

l_layers = args.layers
l_nodes = args.nodes
l_dropout = args.dropout
l_activation = args.activation
l_output_activation = args.output_activation

# Constants
model_type = "nn"

# Load folders
folders = load_training_folders(TRAINING_DATA_CONFIG)

for group_name, selected_columns in FEATURE_GROUPS.items():
	X, y = [], []

	for folder in folders:
		for file in os.listdir(folder):
			if file.endswith(".csv"):
				full_path = os.path.join(folder, file)
				df = pd.read_csv(full_path)

				# Ensure all required features exist
				missing_cols = [col for col in selected_columns if col not in df.columns]
				for col in missing_cols:
					df[col] = 0

				feats = extract_features(df, selected_columns)
				label = get_label(file)

				if feats and label != "N/A":
					X.append(feats)
					y.append(label)

	if X:
		df_X = pd.DataFrame(X)
		# Handle missing values with mean values
		X = df_X.fillna(df_X.mean())
		
		# Encode labels
		label_encoder = LabelEncoder()
		y_encoded = label_encoder.fit_transform(y)
		y_categorical = to_categorical(y_encoded)

		# Train/test split
		X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

		# Scale features
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		# Build Neural Network
		model = Sequential()

		#first hidden layer
		model.add(Dense(l_nodes*l_layers, activation=l_activation, input_shape=(X_train_scaled.shape[1],)))
		model.add(Dropout(l_dropout))

		#remain hidden layers
		for i in range(l_layers - 1,1):
			model.add(Dense(l_nodes*i, activation=l_activation))
			model.add(Dropout(l_dropout))

		#output layer
		model.add(Dense(y_categorical.shape[1], activation=l_output_activation))

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		# Early stopping
		early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

		# Train model
		model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

		# Evaluate
		y_pred_probs = model.predict(X_test_scaled)
		y_pred_labels = np.argmax(y_pred_probs, axis=1)
		y_true_labels = np.argmax(y_test, axis=1)

		print(f"{group_name} - Accuracy:", accuracy_score(y_true_labels, y_pred_labels))
		print("Classification Report:\n", classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_, zero_division=0))

		# Save model and scaler
		os.makedirs(MODEL_DIR, exist_ok=True)
		model.save(f"{MODEL_DIR}/model_{model_type}_{group_name}_{l_layers}_{l_nodes}_{l_dropout}_{l_activation}_{l_output_activation}_{HW_INFRAS_FOR_TRAINING_DATA}.keras")
		joblib.dump(scaler, f"{MODEL_DIR}/scaler_{model_type}_{group_name}_{l_layers}_{l_nodes}_{l_dropout}_{l_activation}_{l_output_activation}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
		joblib.dump(list(X.columns), f"{MODEL_DIR}/metrics_{model_type}_{group_name}_{l_layers}_{l_nodes}_{l_dropout}_{l_activation}_{l_output_activation}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
		joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder_{model_type}_{group_name}_{l_layers}_{l_nodes}_{l_dropout}_{l_activation}_{l_output_activation}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")