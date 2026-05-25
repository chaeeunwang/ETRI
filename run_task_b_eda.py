from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
scripts = [
    PROJECT_ROOT / "code" / "eda_codes" / "01_label_distribution" / "run.py",
    PROJECT_ROOT / "code" / "eda_codes" / "02_subject_prior" / "run.py",
    PROJECT_ROOT / "code" / "eda_codes" / "03_naive_baseline" / "run.py",
    PROJECT_ROOT / "code" / "eda_codes" / "04_target_correlation" / "run.py",
]

for script in scripts:
    print(f"\n[RUN] {script}")
    subprocess.run([sys.executable, str(script)], check=True)

print("\n[DONE] All Task B EDA scripts completed.")
