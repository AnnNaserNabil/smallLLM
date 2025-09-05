import subprocess
import datetime
import os

# List of model scripts to run
model_scripts = [
    "MiniLM-L12-H384-uncased.py",
    "TinyBERT_General_6L_768D.py",
    "albert-base-v2.py",
    "distilbert-base-multilingual-cased.py",
    "flaubert_small_cased.py",
    "mobilebert-uncased.py",
    "xlm-roberta-comet-small.py"
]

# Directory to save logs
log_dir = "error_logs"
os.makedirs(log_dir, exist_ok=True)

for script in model_scripts:
    print(f"\n===== Running {script} =====")
    try:
        # Run the script
        result = subprocess.run(
            ["python3", script],
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if exit code != 0
        )
        print(f"{script} completed successfully.")
    except subprocess.CalledProcessError as e:
        # Create a log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{script}_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(f"===== STDOUT =====\n{e.stdout}\n\n")
            f.write(f"===== STDERR =====\n{e.stderr}\n")
        print(f"❌ {script} failed. Log saved to {log_file}")
    except Exception as e:
        # Catch any other exceptions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{script}_unexpected_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(str(e))
        print(f"❌ {script} encountered an unexpected error. Log saved to {log_file}")

print("\nAll scripts executed.")
import subprocess
import datetime
import os

# List of model scripts to run
model_scripts = [
    "MiniLM-L12-H384-uncased.py",
    "TinyBERT_General_6L_768D.py",
    "albert-base-v2.py",
    "distilbert-base-multilingual-cased.py",
    "flaubert_small_cased.py",
    "mobilebert-uncased.py",
    "xlm-roberta-comet-small.py"
]

# Directory to save logs
log_dir = "error_logs"
os.makedirs(log_dir, exist_ok=True)

for script in model_scripts:
    print(f"\n===== Running {script} =====")
    try:
        # Run the script
        result = subprocess.run(
            ["python3", script],
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if exit code != 0
        )
        print(f"{script} completed successfully.")
    except subprocess.CalledProcessError as e:
        # Create a log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{script}_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(f"===== STDOUT =====\n{e.stdout}\n\n")
            f.write(f"===== STDERR =====\n{e.stderr}\n")
        print(f"❌ {script} failed. Log saved to {log_file}")
    except Exception as e:
        # Catch any other exceptions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{script}_unexpected_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(str(e))
        print(f"❌ {script} encountered an unexpected error. Log saved to {log_file}")

print("\nAll scripts executed.")
