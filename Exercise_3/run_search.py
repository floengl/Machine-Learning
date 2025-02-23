import subprocess
import os
import sys

# Path to the Python interpreter in your virtual environment
venv_python = sys.executable

# List of directories containing the search scripts
directories = [
    r"C:\Users\englf\OneDrive\Dokumente\Flo\Studium\Mathe\Machine_Learning\Machine-Learning\Exercise_3\concrete",
    r"C:\Users\englf\OneDrive\Dokumente\Flo\Studium\Mathe\Machine_Learning\Machine-Learning\Exercise_3\superconductivity",
    r"C:\Users\englf\OneDrive\Dokumente\Flo\Studium\Mathe\Machine_Learning\Machine-Learning\Exercise_3\waves",
    r"C:\Users\englf\OneDrive\Dokumente\Flo\Studium\Mathe\Machine_Learning\Machine-Learning\Exercise_3\wine"
]

# Name of the search script to run in each directory
script_name = "search.py"

for directory in directories:
    script_path = os.path.join(directory, script_name)
    if os.path.exists(script_path):
        try:
            print(f"Running {script_path}...")
            result = subprocess.run([venv_python, script_path], check=True)
            print(f"{script_path} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script_path}: {e}")
            break
    else:
        print(f"{script_path} does not exist.")