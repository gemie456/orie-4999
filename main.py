import subprocess
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# -------- create results/batch_X folder --------
RESULTS_ROOT = ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)

existing = [p for p in RESULTS_ROOT.glob("batch_*") if p.is_dir()]
batch_num = len(existing) + 1

BATCH_DIR = RESULTS_ROOT / f"batch_{batch_num}"
BATCH_DIR.mkdir(exist_ok=True)

print(f"Running batch {batch_num}")
print(f"Results will be saved in: {BATCH_DIR}")

subprocess.run([sys.executable, "src/batch_transitions.py", str(BATCH_DIR)])
subprocess.run([sys.executable, "src/cleaned_data.py", str(BATCH_DIR)])
subprocess.run([sys.executable, "src/v2_similarity_scores.py", str(BATCH_DIR)])
subprocess.run([sys.executable, "src/v2compatibility_scores.py", str(BATCH_DIR)])
subprocess.run([sys.executable, "src/matching_algorithms.py", str(BATCH_DIR)])