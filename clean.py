import os
from my_config import *

def clean_generated_files():
	for filename in os.listdir(MODEL_DIR):
		file_path = os.path.join(MODEL_DIR, filename)
		# Only delete files (not subdirectories)
		if os.path.isfile(file_path):
			try:
				os.remove(file_path)
				print(f"Deleted: {file_path}")
			except Exception as e:
				print(f"Failed to delete {file_path}: {e}")
		else:
			print(f"Skipping non-file entry: {file_path}")

print("Cleaning generated files...")
clean_generated_files()
