# Model types
MODEL_TYPES = ["svm", "nn", "rf"]  # choose which models to train

# NN parameters
NN_ACTIVATION = 'relu'
NN_OUTPUT_ACTIVATION = 'softmax'  
NN_NO_OF_LAYERS = [2, 3, 4]
NN_NODES_LAYER_BASIC = [32, 64]
NN_DROPOUT = [0.1, 0.2, 0.3]

# SVM parameters
SVM_C = [10, 100, 1000]
SVM_GAMMA = 'scale'
SVM_KERNEL = ['poly', 'sigmoid', 'rbf']

# RF parameters
RF_NO_DECISION_TREES = [10, 50, 100]

# Hardware / training infra
HW_INFRAS_FOR_TRAINING_DATA = "grid5000_gros"  # local-server or Grid5000

# Training and feature configuration
TRAINING_DATA_CONFIG = "./config/training_data_setup.csv"
FEATURE_CONFIG = "./config/feature_config.csv"

FEATURE_GROUPS = {
	"CPU": ["cpu_all_util_percent", "cpu_frequency_ghz"],
	"MEMORY": ["memory_util_gb"],	
	"DISK_IO": ["sda_write_count_during_interval", "sda_read_count_during_interval"],
	"ENERGY": ["energy_util_during_interval_j"],
	"ALL": ["cpu_all_util_percent", "cpu_frequency_ghz",
			"energy_util_during_interval_j", "memory_util_gb",
			"sda_write_count_during_interval", "sda_read_count_during_interval"]
}

# Paths for models and results
MODEL_DIR = "./ai_models"
RESULT_DIR = "./all-results"

# Define available testing datasets and their corresponding output filename patterns
TESTING_OPTIONS = {
	"localsever_seen": {
		"path": "./test-localsever",
		"prefix": "prediction_results_seenbm_localsever"
	},
	"localsever_unseen": {
		"path": "./test-unseen-localsever",
		"prefix": "prediction_results_unseenbm_localsever"
	},
	"grid5000_seen": {
		"path": "./test-grid5000-gros",
		"prefix": "prediction_results_seenbm"
	},
	"grid5000_unseen": {
		"path": "./test-unseen-grid5000-gros",
		"prefix": "prediction_results_unseenbm"
	}
}