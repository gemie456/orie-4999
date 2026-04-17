# ===============================
# PCCW Minimum Threshold Matching (τ = 0.4)
# Full Updated Version (ZIP output + clean schemas)
# ===============================

import pandas as pd
import numpy as np

import zipfile
import sys
from pathlib import Path
import pulp

if len(sys.argv) > 1:
    BATCH_DIR = Path(sys.argv[1])
else:
    BATCH_DIR = Path(".")

print("Using batch directory:", BATCH_DIR)

# ---------- Load data ----------
mentors = pd.read_csv(BATCH_DIR / "mentors_clean.csv")
mentees = pd.read_csv(BATCH_DIR / "mentees_clean.csv")
C = pd.read_csv(BATCH_DIR / "compatibility_matrix.csv", index_col=0)

# Align C with actually present mentors/mentees
mentees_list = [m for m in C.index if m in set(mentees["Name"])]
mentors_list = [m for m in C.columns if m in set(mentors["Name"])]
C = C.loc[mentees_list, mentors_list].copy()

# ---------- Parameters ----------
TAU = 0.40      # minimum compatibility threshold
SOLVER_TIME_LIMIT = None  # you can set a time limit in seconds if needed
USE_FIELD_PROTECTION = True
FIELD_POPULARITY_THRESHOLD = 0.3 #0.1
FLEXIBILITY_WEIGHT = 0.3 #0.3

# ---------- Mentor capacity + prefs ----------
def safe_capacity(v):
    try:
        return max(int(v), 0)
    except:
        return 1

mentor_capacity = (
    mentors.set_index("Name")["Mentor_Capacity"]
    .apply(safe_capacity)
    .reindex(mentors_list)
    .fillna(1)
    .astype(int)
)

def parse_pref(s):
    s = str(s).lower()
    if "under" in s and "grad" in s:
        return "both"
    if "under" in s:
        return "undergraduate"
    if "grad" in s:
        return "graduate"
    return "both"

mentor_pref = (
    mentors.set_index("Name")["Mentor_Level_Preferences"]
    .apply(parse_pref)
    .reindex(mentors_list)
    .fillna("both")
)

mentee_level = (
    mentees.set_index("Name")["Student_Level"]
    .reindex(mentees_list)
    .fillna("undergraduate")
    .str.lower()
)

# ---------- Organization field ----------
mentee_org = mentees.set_index("Name")["Organization"].reindex(mentees_list)


# ---------- Info dictionaries ----------
mentee_info = mentees.set_index("Name")[
    ["Email", "Phone_Number", "Bio", "Additional_Notes"]
].to_dict(orient="index")

mentor_info = mentors.set_index("Name")[
    ["Email", "Phone_Number", "Bio", "Additional_Notes"]
].to_dict(orient="index")

def _get(d, name, field):
    v = d.get(name, {}).get(field, "None")
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return "None"
    return str(v)

# ---------- Using Field Protection ----------
if USE_FIELD_PROTECTION:
    field_demand = {}
    for i in mentees_list:
        mentee_row = mentees.set_index("Name").loc[i]
        primary_field = str(mentee_row["Field_1"]).strip()
        if primary_field not in ["None", "", "nan"]:
            field_demand[primary_field] = field_demand.get(primary_field, 0) + 1

    sorted_fields = sorted(field_demand.items(), key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(np.ceil(len(sorted_fields) * FIELD_POPULARITY_THRESHOLD)))
    popular_fields = dict(sorted_fields[:cutoff])

    max_demand = max(popular_fields.values())

    field_reserved_capacity = {}
    for j in mentors_list:
        mentor_row = mentors.set_index("Name").loc[j]
        primary_field = str(mentor_row["Field_1"]).strip()

        if primary_field in popular_fields:
            field_pop = popular_fields[primary_field]
            popularity_ratio = 0.25 + 0.25 * (field_pop / max_demand)

            # mentor_fields = [
            #     str(mentor_row.get(f"Field_{k}", "")).strip()
            #     for k in range(1, 6)
            #     if str(mentor_row.get(f"Field_{k}", "")).strip()
            #     not in ["None", "", "nan"]
            # ]

            mentor_fields = [
                str(mentor_row.get(f"Field_{k}", "")).strip()
                for k in range(1, 6)
                if str(mentor_row.get(f"Field_{k}", "")).strip()
                in popular_fields
            ]
            num_fields = len(mentor_fields)
            flexibility_score = num_fields / 5

            reserve_ratio = (
                (1 - FLEXIBILITY_WEIGHT) * popularity_ratio +
                FLEXIBILITY_WEIGHT * flexibility_score
            )

            original_cap = mentor_capacity.loc[j]
            reserved = max(1, int(np.ceil(original_cap * reserve_ratio)))
            effective_cap = max(1, original_cap - reserved)
            field_reserved_capacity[j] = effective_cap
        else:
            field_reserved_capacity[j] = int(mentor_capacity.loc[j])

# ---------- Feasible pairs ----------
allowed = []
for i in mentees_list:
    for j in mentors_list:
        score = C.loc[i, j]
        if np.isnan(score) or score < TAU:
            continue

        lvl = mentee_level.loc[i]
        pref = mentor_pref.loc[j]

        # respect mentor level preferences
        if not (
            pref == "both"
            or (pref == "undergraduate" and "under" in lvl)
            or (pref == "graduate" and "grad" in lvl)
        ):
            continue

        allowed.append((i, j))

# ---------- Optimization ----------
prob = pulp.LpProblem("MinThresholdMatch", pulp.LpMaximize)

# binary decision variables x_{i,j}
x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for (i, j) in allowed}

# objective: maximize sum of compatibility scores
prob += pulp.lpSum(C.loc[i, j] * x[(i, j)] for (i, j) in allowed)

# each mentee matched to at most one mentor
for i in mentees_list:
    prob += pulp.lpSum(x[(i, j)] for j in mentors_list if (i, j) in x) <= 1

# respect mentor capacities
for j in mentors_list:
    effective_cap = (
        field_reserved_capacity[j]
        if USE_FIELD_PROTECTION and j in field_reserved_capacity
        else int(mentor_capacity.loc[j])
    )
    prob += (
        pulp.lpSum(x[(i, j)] for i in mentees_list if (i, j) in x)
        <= effective_cap
    )

# solve
prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT))

# ---------- Matched pairs ----------
rows = [
    (i, j, float(C.loc[i, j]))
    for (i, j) in allowed
    if pulp.value(x[(i, j)]) > 0.5
]

matched_df = pd.DataFrame(rows, columns=["Mentee", "Mentor", "Compatibility_Score"])

# ---------- Attach detailed info ----------
if not matched_df.empty:
    matched_df["Mentee_Email"] = matched_df["Mentee"].apply(
        lambda n: _get(mentee_info, n, "Email")
    )
    matched_df["Mentee_Phone"] = matched_df["Mentee"].apply(
        lambda n: _get(mentee_info, n, "Phone_Number")
    )
    matched_df["Mentee_Organization"] = (
        matched_df["Mentee"].map(mentee_org).fillna("None")
    )
    matched_df["Mentee_Bio"] = matched_df["Mentee"].apply(
        lambda n: _get(mentee_info, n, "Bio")
    )
    matched_df["Mentee_Additional_Notes"] = matched_df["Mentee"].apply(
        lambda n: _get(mentee_info, n, "Additional_Notes")
    )

    matched_df["Mentor_Email"] = matched_df["Mentor"].apply(
        lambda n: _get(mentor_info, n, "Email")
    )
    matched_df["Mentor_Phone"] = matched_df["Mentor"].apply(
        lambda n: _get(mentor_info, n, "Phone_Number")
    )
    matched_df["Mentor_Bio"] = matched_df["Mentor"].apply(
        lambda n: _get(mentor_info, n, "Bio")
    )
    matched_df["Mentor_Additional_Notes"] = matched_df["Mentor"].apply(
        lambda n: _get(mentor_info, n, "Additional_Notes")
    )

    matched_df = matched_df[
        [
            "Mentee",
            "Mentee_Email",
            "Mentee_Phone",
            "Mentee_Organization",
            "Mentee_Bio",
            "Mentee_Additional_Notes",
            "Mentor",
            "Mentor_Email",
            "Mentor_Phone",
            "Mentor_Bio",
            "Mentor_Additional_Notes",
            "Compatibility_Score",
        ]
    ]

    matched_df = matched_df.sort_values(
        by=["Mentee_Organization", "Mentee"]
    )

# Save matched pairs
matched_df.to_csv(BATCH_DIR / "matched_pairs.csv", index=False)

# ---------- Remaining mentors (FULL INFO, explicit Name column) ----------
used_counts = matched_df["Mentor"].value_counts().to_dict()

remaining_capacity = {
    j: max(mentor_capacity.loc[j] - used_counts.get(j, 0), 0)
    for j in mentors_list
}
remaining_mentors = [j for j, cap in remaining_capacity.items() if cap > 0]

mentor_cols_ordered = [
    "Name",
    "Role",
    "Email",
    "Phone_Number",
    "Organization",
    "Location",
    "College",
    "Career_Interests",
    "Field_1",
    "Field_2",
    "Field_3",
    "Field_4",
    "Field_5",
    "Bio",
    "Additional_Notes",
    "Mentor_Level_Preferences",
    "Remaining_Capacity",
]

if remaining_mentors:
    full_remaining_df = mentors.set_index("Name").loc[remaining_mentors].copy()
    full_remaining_df["Remaining_Capacity"] = [
        remaining_capacity[j] for j in remaining_mentors
    ]

    # drop original capacity column; we only keep Remaining_Capacity
    if "Mentor_Capacity" in full_remaining_df.columns:
        full_remaining_df = full_remaining_df.drop(columns=["Mentor_Capacity"])

    # bring Name back as a real column
    full_remaining_df = full_remaining_df.reset_index()  # index -> Name

    # re-order columns
    full_remaining_df = full_remaining_df[mentor_cols_ordered]
else:
    full_remaining_df = pd.DataFrame(columns=mentor_cols_ordered)

full_remaining_df.to_csv(BATCH_DIR / "mentors_remaining_capacity.csv", index=False)

# ---------- Unmatched mentees (full profile, explicit Name column) ----------
matched_mentees = set(matched_df["Mentee"])
unmatched = [i for i in mentees_list if i not in matched_mentees]

if unmatched:
    unmatched_df = mentees.set_index("Name").loc[unmatched].copy()
    unmatched_df = unmatched_df.reset_index()  # index -> Name

    if "Organization" in unmatched_df.columns:
        unmatched_df = unmatched_df.sort_values(by=["Organization", "Name"])
    else:
        unmatched_df = unmatched_df.sort_values(by=["Name"])
else:
    # empty with same columns as mentees_clean.csv
    unmatched_df = pd.DataFrame(columns=list(mentees.columns))

unmatched_df.to_csv(BATCH_DIR /"unmatched_mentees.csv", index=False)

# ---------- Report + ZIP download ----------
num_pairs = len(matched_df)
num_unmatched = len(unmatched_df)
print(f"✅ Matching complete — {num_pairs} pairs, {num_unmatched} unmatched mentees.")

zip_filename = "matching_algorithms_output.zip"
zip_path = BATCH_DIR / zip_filename

with zipfile.ZipFile(zip_path, "w") as z:
    z.write(BATCH_DIR / "matched_pairs.csv", arcname="matched_pairs.csv")
    z.write(BATCH_DIR / "mentors_remaining_capacity.csv", arcname="mentors_remaining_capacity.csv")
    z.write(BATCH_DIR / "unmatched_mentees.csv", arcname="unmatched_mentees.csv")

print(f"📦 ZIP saved to {zip_path}")

