import sys
import os
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from config.my_config import *
from common_function import *
import warnings
import itertools

warnings.simplefilter("ignore", category=RuntimeWarning)

# Initialize list to collect results
evaluation_results = []

for model_type in MODEL_TYPES:
	print(f"\n===== Running predictions for model type: {model_type.upper()} =====")

	if model_type == "svm":
		hyperparam_combos = list(itertools.product(SVM_KERNEL, SVM_C))
	elif model_type == "rf":
		hyperparam_combos = [(n,) for n in RF_NO_DECISION_TREES]
	elif model_type == "nn":
		hyperparam_combos = list(itertools.product(NN_NO_OF_LAYERS, NN_NODES_LAYER_BASIC, NN_DROPOUT))
	else:
		continue

	for combo in hyperparam_combos:
		print(f"\nTesting hyperparameters: {combo}")

		for test_name, test_info in TESTING_OPTIONS.items():			
			print(f"\nTesting with {test_name}")

			correct_count = 0
			wrong_count = 0
			total_files = 0

			TESTING_DATA_PATH = test_info["path"]
			prefix = test_info["prefix"]
			# For storing results
			results = []
			results_summary= []

			for file_name in os.listdir(TESTING_DATA_PATH):
				if not file_name.endswith(".csv"):
					continue

				total_files += 1

				file_path = os.path.join(TESTING_DATA_PATH, file_name)
				df = pd.read_csv(file_path)

				if df.empty:
					print(f"Skipping {file_name}: empty file.")
					continue

				all_probs = []
				class_labels = None

				for group, features in FEATURE_GROUPS.items():
					missing_cols = [col for col in features if col not in df.columns]
					for col in missing_cols:
						df[col] = 0

					feats = extract_features(df, features)
					X = pd.DataFrame([feats])
	
					try:
						# Model loading
						if model_type == "svm":							
							l_kernel, l_C = combo
							model_path = f"{MODEL_DIR}/model_{model_type}_{group}_{l_kernel}_{l_C}_{SVM_GAMMA}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							scaler_path = f"{MODEL_DIR}/scaler_{model_type}_{group}_{l_kernel}_{l_C}_{SVM_GAMMA}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							metrics_path = f"{MODEL_DIR}/metrics_{model_type}_{group}_{l_kernel}_{l_C}_{SVM_GAMMA}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							model = joblib.load(model_path)
							scaler = joblib.load(scaler_path)
							expected_cols = joblib.load(metrics_path)
							is_nn = False

						elif model_type == "nn":							
							l_layers, l_nodes, l_dropout = combo
							model_path = f"{MODEL_DIR}/model_{model_type}_{group}_{l_layers}_{l_nodes}_{l_dropout}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.keras"
							scaler_path = f"{MODEL_DIR}/scaler_{model_type}_{group}_{l_layers}_{l_nodes}_{l_dropout}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							metrics_path = f"{MODEL_DIR}/metrics_{model_type}_{group}_{l_layers}_{l_nodes}_{l_dropout}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							encoder_path = f"{MODEL_DIR}/label_encoder_{model_type}_{group}_{l_layers}_{l_nodes}_{l_dropout}_{NN_ACTIVATION}_{NN_OUTPUT_ACTIVATION}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							model = load_model(model_path)
							scaler = joblib.load(scaler_path)
							expected_cols = joblib.load(metrics_path)
							label_encoder = joblib.load(encoder_path)
							is_nn = True

						elif model_type == "rf":							
							n_trees, = combo
							model_path = f"{MODEL_DIR}/model_{model_type}_{group}_{n_trees}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							scaler_path = f"{MODEL_DIR}/scaler_{model_type}_{group}_{n_trees}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							metrics_path = f"{MODEL_DIR}/metrics_{model_type}_{group}_{n_trees}_{HW_INFRAS_FOR_TRAINING_DATA}.pkl"
							model = joblib.load(model_path)
							scaler = joblib.load(scaler_path)
							expected_cols = joblib.load(metrics_path)
							is_nn = False

						# Preprocess and predict
						X = X[expected_cols]
						X = X.fillna(X.mean()).fillna(0)
						X_scaled = scaler.transform(X)

						if is_nn:
							probs = model.predict(X_scaled, verbose=0)[0]
							class_labels = label_encoder.classes_
							pred_idx = np.argmax(probs)
							pred_label = class_labels[pred_idx]
							confidence = probs[pred_idx]
						else:
							probs = model.predict_proba(X_scaled)[0]
							pred_label = model.predict(X_scaled)[0]
							confidence = probs[model.classes_.tolist().index(pred_label)]
							class_labels = model.classes_

						all_probs.append(probs)
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

				# Ensemble averaging (soft voting)
				if not all_probs:
					continue

				avg_probs = np.mean(all_probs, axis=0)
				top2_idx = np.argsort(avg_probs)[-2:][::-1]
				top1_idx, top2_idx = top2_idx
				top1_conf = avg_probs[top1_idx]
				top2_conf = avg_probs[top2_idx]
				top1_label = class_labels[top1_idx]
				top2_label = class_labels[top2_idx]
				is_mixed = (top1_conf - top2_conf) < 0.2

				results_summary.append({
					"file_name": file_name,
					"prediction": top1_label,
					"confidence": round(float(top1_conf), 4),
					"second_prediction": top2_label,
					"second_confidence": round(float(top2_conf), 4)
				})

				# Get ground-truth from filename
				true_label = get_label(file_name)
				if true_label == "N/A":
					continue

				if top1_label == true_label:
					correct_count += 1
				else:
					wrong_count += 1

			# Save outputs
			results_df = pd.DataFrame(results)
			results_df.to_csv(f"{RESULT_DIR}/{prefix}_{model_type}_{'_'.join(map(str, combo))}", index=False)

			results2_df = pd.DataFrame(results_summary)
			results2_df.to_csv(f"{RESULT_DIR}/full_{prefix}_{model_type}_{'_'.join(map(str, combo))}", index=False)

			accuracy = (correct_count / total_files) * 100 if total_files > 0 else 0
			# Record the results
			evaluation_results.append({
				"model_type_hyperparam": f"{model_type}_{'_'.join(map(str, combo))}",
				"test_option": test_name,
				"correct_count": correct_count,
				"wrong_count": wrong_count,
				"accuracy": round(accuracy, 4)
			})

# Save all results to CSV
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv(os.path.join(RESULT_DIR, "evaluation_summary.csv"), index=False)
print("\nEvaluation complete. Results saved to evaluation_summary.csv")
