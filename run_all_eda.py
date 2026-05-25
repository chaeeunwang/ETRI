import subprocess
import sys

modules = [
    "eda.01_label_distribution.code.run_label_distribution",
    "eda.02_subject_prior.code.run_subject_prior",
    "eda.03_naive_baseline.code.run_naive_baseline",
    "eda.04_target_correlation.code.run_target_correlation",
]

for m in modules:
    print(f"\n[RUN] {m}")
    subprocess.run([sys.executable, "-m", m], check=True)

print("\n[DONE] All EDA scripts completed.")
