from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\deoha\Downloads\etri_2026_eda_template_visual_full\etri_2026_eda_template")
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "ch2026_metrics_train.csv"
SUBMISSION_SAMPLE = DATA_DIR / "ch2026_submission_sample.csv"
DATA_ITEMS_DIR = DATA_DIR / "ch2025_data_items"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COL = "subject_id"
SLEEP_DATE_COL = "sleep_date"
LIFELOG_DATE_COL = "lifelog_date"
RANDOM_STATE = 42
