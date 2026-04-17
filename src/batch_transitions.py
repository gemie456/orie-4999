# -*- coding: utf-8 -*-
"""Batch Transitions"""

import pandas as pd
import sys
from pathlib import Path

# -------------------------------
# Paths
# -------------------------------
BATCH_DIR = Path(sys.argv[1])
BATCH_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"

batch_num = int(BATCH_DIR.name.split("_")[1])

form_df = pd.read_csv(ROOT / "data/real_data/PCCW_Batch_5.csv")
output_file = BATCH_DIR / "pccw_form_responses.csv"
print(form_df.shape)
print(form_df.head(3))

# -------------------------------
# Batch 1: just save raw input
# -------------------------------
if batch_num == 1:
    form_df.to_csv(output_file, index=False)
    print(f"Saved batch 1 input to {output_file}")
    sys.exit()

# -------------------------------
# Find needed columns in form file
# -------------------------------
email_col = [c for c in form_df.columns if "email address" in c.lower()][0]
role_col = [c for c in form_df.columns if "are you a/an" in c.lower()][0]
capacity_col = [c for c in form_df.columns if "how many mentees" in c.lower()][0]

# -------------------------------
# Read previous matched pairs only
# -------------------------------
matched_files = [
    RESULTS_ROOT / f"batch_{i}" / "matched_pairs.csv"
    for i in range(1, batch_num)
    if (RESULTS_ROOT / f"batch_{i}" / "matched_pairs.csv").exists()
]

matched_frames = []

for f in matched_files:
    df = pd.read_csv(f)

    # skip truly empty files
    if df.empty and len(df.columns) == 0:
        continue

    # normalize possible column names
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if c == "mentee_email" or c == "mentee email":
            rename_map[col] = "Mentee_Email"
        elif c == "mentor_email" or c == "mentor email":
            rename_map[col] = "Mentor_Email"

    df = df.rename(columns=rename_map)

    # only keep files that actually contain these columns
    if "Mentee_Email" in df.columns and "Mentor_Email" in df.columns:
        matched_frames.append(df[["Mentee_Email", "Mentor_Email"]])

if matched_frames:
    matched_all = pd.concat(matched_frames, ignore_index=True)
    matched_all["Mentee_Email"] = matched_all["Mentee_Email"].astype(str).str.strip().str.lower()
    matched_all["Mentor_Email"] = matched_all["Mentor_Email"].astype(str).str.strip().str.lower()

    matched_mentees = set(matched_all["Mentee_Email"])
    used_mentors = set(matched_all["Mentor_Email"])
else:
    matched_mentees = set()
    used_mentors = set()

# -------------------------------
# Read previous batch mentor capacity
# -------------------------------
cap_file = RESULTS_ROOT / f"batch_{batch_num - 1}" / "mentors_remaining_capacity.csv"
if not cap_file.exists():
    raise FileNotFoundError(f"Missing file: {cap_file}")

cap_df = pd.read_csv(cap_file)
cap_df["Email"] = cap_df["Email"].astype(str).str.strip().str.lower()
cap_map = dict(zip(cap_df["Email"], cap_df["Remaining_Capacity"]))

# -------------------------------
# Build cleaned dataframe
# -------------------------------
rows = []

for _, row in form_df.iterrows():
    email = str(row[email_col]).strip().lower()
    role = str(row[role_col]).strip().lower()

    if "undergraduate" in role or "graduate" in role:
        if email not in matched_mentees:
            rows.append(row)

    elif "pccw" in role:
        remaining = cap_map.get(email)

        if remaining is None:
            if email not in used_mentors:
                rows.append(row)
        elif remaining > 0:
            new_row = row.copy()
            new_row[capacity_col] = remaining
            rows.append(new_row)

    else:
        rows.append(row)

cleaned_df = pd.DataFrame(rows)

# -------------------------------
# Save into current batch folder
# -------------------------------
cleaned_df.to_csv(output_file, index=False)
print(f"Saved batch {batch_num} input to {output_file}")