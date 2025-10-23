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