import os
import itertools
from config.my_config import *

py = sys.executable
for model_type in MODEL_TYPES:
	if model_type == "svm":
		for kernel, C in itertools.product(SVM_KERNEL, SVM_C):
			os.system(f'"{py}" train_svm.py --kernel {kernel} --C {C} --gamma {SVM_GAMMA}')
	elif model_type == "nn":
		for layers, nodes, dropout in itertools.product(NN_NO_OF_LAYERS, NN_NODES_LAYER_BASIC, NN_DROPOUT):
			os.system(f'"{py}" train_nn.py --layers {layers} --nodes {nodes} --dropout {dropout} --activation {NN_ACTIVATION} --output_activation {NN_OUTPUT_ACTIVATION}')
	elif model_type == "rf":
		for n_trees in RF_NO_DECISION_TREES:
			os.system(f'"{py}" train_rf.py --n_trees {n_trees}')
	else:
		print(f"Unknown model type: {model_type}")
		continue

print(f"\nAll training runs complete. Logs saved in {log_file}")