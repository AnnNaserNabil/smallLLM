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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script}_{timestamp}.txt")

    try:
        with open(log_file, "w") as f:
            # Start the subprocess
            process = subprocess.Popen(
                ["python3", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Stream output line by line
            for line in process.stdout:
                print(line, end="")   # show in terminal
                f.write(line)         # save to log file

            process.wait()
            if process.returncode == 0:
                print(f"\n✅ {script} completed successfully.")
            else:
                print(f"\n❌ {script} failed. See log: {log_file}")

    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\nUnexpected error:\n{str(e)}")
        print(f"\n❌ {script} encountered an unexpected error. See log: {log_file}")

print("\nAll scripts executed.")
