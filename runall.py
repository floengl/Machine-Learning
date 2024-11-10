import os
import subprocess

def execute_all_scripts(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Executing {file_path}")
                result = subprocess.run(["python3", file_path], capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print(f"Error in {file_path}:\n{result.stderr}")

if __name__ == "__main__":
    execute_all_scripts("Exercise_1")