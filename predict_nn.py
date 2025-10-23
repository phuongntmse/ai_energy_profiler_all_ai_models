import sys
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from extract_features import extract_features
from scipy.special import softmax
from scipy.stats import mode
import os
import warnings
import numpy as np
from my_config import *

warnings.simplefilter("ignore", category=RuntimeWarning)

if not os.path.isdir(TESTING_DATA_PATH):
	print(f"Error: '{TESTING_DATA_PATH}' is not a directory.")
	sys.exit(1)

# For storing results
results = []
results2= []
# 
# Process each CSV file
for file_name in os.listdir(TESTING_DATA_PATH):
	if file_name.endswith(".csv"):
		file_path = os.path.join(TESTING_DATA_PATH, file_name)
		df = pd.read_csv(file_path)
		if df.empty:
			print(f"Skipping {file_name}: empty file.")
			continue

		all_probs = []
		class_labels=None

		for group, features in FEATURE_GROUPS.items():
			# Ensure all required features exist, fill missing with 0
			missing_cols = [col for col in features if col not in df.columns]
			# Add dummy columns for missing features
			for col in missing_cols:
				df[col] = 0  # or use np.nan if needed
			
			feats = extract_features(df, features)
			X = pd.DataFrame([feats])

			try:
				# Load NN model, scaler, encoder, and expected cols
				model = load_model(f"{MODEL_DIR}/model_nn_{group}_{NN_NO_OF_LAYERS}_{NN_NODES_LAYER_BASIC}_{NN_DROPOUT}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.keras")
				scaler = joblib.load(f"{MODEL_DIR}/scaler_nn_{group}_{NN_NO_OF_LAYERS}_{NN_NODES_LAYER_BASIC}_{NN_DROPOUT}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
				expected_cols = joblib.load(f"{MODEL_DIR}/metrics_{group}_{NN_NO_OF_LAYERS}_{NN_NODES_LAYER_BASIC}_{NN_DROPOUT}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")
				label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder_{group}_{NN_NO_OF_LAYERS}_{NN_NODES_LAYER_BASIC}_{NN_DROPOUT}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl")

				# Ensure the order of features matches
				X = X[expected_cols]
				# Handle missing values
				X = X.fillna(X.mean())

				X_scaled = scaler.transform(X)

				# NN prediction (probabilities)
				probs = model.predict(X_scaled, verbose=0)[0]
				all_probs.append(probs)

				if class_labels is None:
					class_labels = label_encoder.classes_

				# Predicted class
				pred_idx = np.argmax(probs)
				pred_label = class_labels[pred_idx]
				confidence = probs[pred_idx]

				results.append({
					"file_name": file_name,
					"model": group,
					"prediction": pred_label,
					"confidence": round(float(confidence), 4)
				})
			except Exception as e:
				results.append({
					"file_name": file_name,
					"model": group,
					"prediction": str(e),
					"confidence": 0.0
				})

		if not all_probs:
			continue

		# Average predictions (soft voting)
		avg_probs = np.mean(all_probs, axis=0)

		# Top-2 predictions
		top2_idx = np.argsort(avg_probs)[-2:][::-1]
		top1_idx, top2_idx = top2_idx

		top1_conf = avg_probs[top1_idx]
		top2_conf = avg_probs[top2_idx]

		top1_label = class_labels[top1_idx]
		top2_label = class_labels[top2_idx]

		# Detect possible mix
		confidence_gap = top1_conf - top2_conf
		is_mixed = confidence_gap < 0.2

		results2.append({
			"file_name": file_name,
			"prediction": top1_label,
			"confidence": round(float(top1_conf), 4),
			"second_prediction": top2_label,
			"second_confidence": round(float(top2_conf), 4),
			"possible_mix_guess": f"{top1_label} + {top2_label}" if is_mixed else "None"
		})


		# Final prediction = class with highest average probability
		# final_preds = np.argmax(avg_probs, axis=1)
		# predicted_labels = class_labels[final_preds]
		# results2.append({
		# 	"file_name": file_name,
		# 	"prediction": predicted_labels[0],
		# 	"confidence": round(confidence_scores[0], 4) if confidence_scores[0] is not None else "N/A"											
		# })
									
# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULT_DIR}/{RESULT_PATH}", index=False)

results2_df = pd.DataFrame(results2)
results2_df.to_csv(f"{RESULT_DIR}/full_{RESULT_PATH}", index=False)

print(f"\n Prediction complete. Results saved to {RESULT_PATH} and full_{RESULT_PATH}.")