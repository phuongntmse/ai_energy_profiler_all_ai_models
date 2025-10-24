import os
import sys
import itertools
from config.my_config import *

py = sys.executable
os.makedirs(LOG_DIR, exist_ok=True)
for model_type in MODEL_TYPES:
	if model_type == "svm":
		for kernel, C in itertools.product(SVM_KERNEL, SVM_C):			
			log_file = os.path.join(LOG_DIR, f"svm_{kernel}_{C}.txt")
			print(log_file)
			os.system(f'"{py}" train_svm.py --kernel {kernel} --C {C} --gamma {SVM_GAMMA} > {log_file} 2>&1')
	elif model_type == "nn":
		for layers, nodes, dropout in itertools.product(NN_NO_OF_LAYERS, NN_NODES_LAYER_BASIC, NN_DROPOUT):
			log_file = os.path.join(LOG_DIR, f"nn_{layers}_{nodes}_{dropout}.txt")	
			print(log_file)
			os.system(f'"{py}" train_nn.py --layers {layers} --nodes {nodes} --dropout {dropout} --activation {NN_ACTIVATION} --output_activation {NN_OUTPUT_ACTIVATION} > {log_file} 2>&1')
	elif model_type == "rf":
		for n_trees in RF_NO_DECISION_TREES:
			log_file = os.path.join(LOG_DIR, f"rf_{n_trees}.txt")
			print(log_file)
			os.system(f'"{py}" train_rf.py --trees {n_trees} > {log_file} 2>&1')
	else:
		print(f"Unknown model type: {model_type}")
		continue

print(f"\nAll training runs complete. Logs saved in {LOG_DIR}")