# ===============================
# PCCW Greedy Matching - Modified Algorithm
# Solution 1: Popular Mentor Capacity Reservation
# Solution 2: Constrained Mentee Priority
# ===============================
# HOW THIS DIFFERS FROM THE ORIGINAL Greedy Matching Algorithms.ipynb:
#
# ORIGINAL: Mentees are processed in whatever order they appear in the data.
#   This means early mentees consume the most popular/flexible mentors first,
#   leaving later batches with fewer and worse options.
#
# THIS VERSION applies two improvements (both are active; set flags to toggle):
#
# [Solution 2 - Constrained Mentee Priority]
#   Before matching begins, mentees are SORTED by how many compatible mentors
#   they have (fewest first). Ties are broken by their best available score.
#   Effect: "hardest to place" mentees are matched first, preserving flexible
#   mentors for mentees who actually need them.
#
# [Solution 1 - Popular Mentor Reservation]
#   Mentors are ranked by popularity (# of compatible mentees x avg score).
#   The top POPULARITY_THRESHOLD fraction (default 20%) are flagged "popular".
#   Popular mentors may only fill up to POPULAR_BATCH_RESERVE_FRAC (default 50%)
#   of their capacity per batch, keeping the rest for later batches.
#   Effect: high-demand mentors remain available across multiple rounds instead
#   of being fully consumed in the first batch.
#
# All other logic (TAU threshold, level preferences, output format, ZIP) is
# identical to the original so outputs are directly comparable.
# ===============================

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import zipfile

if len(sys.argv) > 1:
    BATCH_DIR = Path(sys.argv[1])
else:
    BATCH_DIR = Path(".")   # default to current folder

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
TAU = 0.40  # minimum compatibility threshold (same as original)

# --- Modification toggles ---
USE_CONSTRAINED_PRIORITY = True   # Solution 2: sort mentees hardest-to-place first
USE_POPULARITY_RESERVE   = True   # Solution 1: reserve capacity for popular mentors

# Solution 1 parameters
POPULARITY_THRESHOLD      = 0.30  # top 20% of mentors classified as "popular"
POPULAR_BATCH_RESERVE_FRAC = 0.25 # popular mentors fill at most 50% capacity per batch

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

# ===============================
# Helper: check if a mentee-mentor pair is feasible
# ===============================
def is_feasible(mentee, mentor):
    score = C.loc[mentee, mentor]
    if np.isnan(score) or score < TAU:
        return False
    pref = mentor_pref.loc[mentor]
    lvl  = mentee_level.loc[mentee]
    return (
        pref == "both"
        or (pref == "undergraduate" and "under" in lvl)
        or (pref == "graduate"      and "grad"  in lvl)
    )

# ===============================
# Solution 1: Compute mentor popularity scores & flag popular mentors
# ===============================
mentor_popularity = {}
for mentor in mentors_list:
    compatible_scores = [
        C.loc[mentee, mentor]
        for mentee in mentees_list
        if is_feasible(mentee, mentor)
    ]
    if compatible_scores:
        # popularity = count * average score
        mentor_popularity[mentor] = len(compatible_scores) * np.mean(compatible_scores)
    else:
        mentor_popularity[mentor] = 0.0

# Rank and flag top POPULARITY_THRESHOLD as "popular"
sorted_by_pop = sorted(mentors_list, key=lambda m: mentor_popularity[m], reverse=True)
n_popular = max(1, int(np.ceil(len(mentors_list) * POPULARITY_THRESHOLD)))
popular_mentors = set(sorted_by_pop[:n_popular])

print(f"[Solution 1] {n_popular} mentors flagged as popular out of {len(mentors_list)} total.")
if USE_POPULARITY_RESERVE:
    print(f"  -> Popular mentors capped at {int(POPULAR_BATCH_RESERVE_FRAC*100)}% of capacity per batch.")
else:
    print("  -> Popularity reservation DISABLED (USE_POPULARITY_RESERVE=False).")

# Per-batch cap for popular mentors (rounds up so cap >= 1)
def popular_batch_cap(mentor):
    full_cap = int(mentor_capacity.loc[mentor])
    return max(1, int(np.ceil(full_cap * POPULAR_BATCH_RESERVE_FRAC)))

# ===============================
# Solution 2: Build sorted mentee order (constrained first)
# ===============================
def build_mentee_order():
    mentee_options = {}
    for mentee in mentees_list:
        feasible_scores = [
            C.loc[mentee, mentor]
            for mentor in mentors_list
            if is_feasible(mentee, mentor)
        ]
        count = len(feasible_scores)
        best  = max(feasible_scores) if feasible_scores else 0.0
        mentee_options[mentee] = (count, best)
    # sort: fewest options first; break ties by lower best score first
    return sorted(mentees_list, key=lambda m: (mentee_options[m][0], mentee_options[m][1]))

if USE_CONSTRAINED_PRIORITY:
    ordered_mentees = build_mentee_order()
    print(f"[Solution 2] Mentees reordered: hardest-to-place first.")
else:
    ordered_mentees = mentees_list
    print("[Solution 2] Constrained priority DISABLED (USE_CONSTRAINED_PRIORITY=False).")

# ===============================
# Greedy matching (modified)
# ===============================
pairs = []
used = {j: 0 for j in mentors_list}  # how many mentees each mentor has so far

for mentee in ordered_mentees:
    scores = []
    for mentor in mentors_list:
        if not is_feasible(mentee, mentor):
            continue
        # respect total capacity
        if used[mentor] >= mentor_capacity.loc[mentor]:
            continue
        # Solution 1: respect per-batch cap for popular mentors
        if USE_POPULARITY_RESERVE and mentor in popular_mentors:
            if used[mentor] >= popular_batch_cap(mentor):
                continue
        scores.append((C.loc[mentee, mentor], mentor))

    if scores:
        best_score, best_mentor = max(scores, key=lambda t: t[0])
        pairs.append((mentee, best_mentor, float(best_score)))
        used[best_mentor] += 1

# ===============================
# Build output DataFrame (identical structure to original)
# ===============================
matched_df = pd.DataFrame(pairs, columns=["Mentee", "Mentor", "Compatibility_Score"])

if not matched_df.empty:
    matched_df["Mentee_Email"]            = matched_df["Mentee"].apply(lambda n: _get(mentee_info, n, "Email"))
    matched_df["Mentee_Phone"]            = matched_df["Mentee"].apply(lambda n: _get(mentee_info, n, "Phone_Number"))
    matched_df["Mentee_Organization"]     = matched_df["Mentee"].map(mentee_org).fillna("None")
    matched_df["Mentee_Bio"]              = matched_df["Mentee"].apply(lambda n: _get(mentee_info, n, "Bio"))
    matched_df["Mentee_Additional_Notes"] = matched_df["Mentee"].apply(lambda n: _get(mentee_info, n, "Additional_Notes"))
    matched_df["Mentor_Email"]            = matched_df["Mentor"].apply(lambda n: _get(mentor_info, n, "Email"))
    matched_df["Mentor_Phone"]            = matched_df["Mentor"].apply(lambda n: _get(mentor_info, n, "Phone_Number"))
    matched_df["Mentor_Bio"]              = matched_df["Mentor"].apply(lambda n: _get(mentor_info, n, "Bio"))
    matched_df["Mentor_Additional_Notes"] = matched_df["Mentor"].apply(lambda n: _get(mentor_info, n, "Additional_Notes"))

    matched_df = matched_df[[
        "Mentee", "Mentee_Email", "Mentee_Phone", "Mentee_Organization",
        "Mentee_Bio", "Mentee_Additional_Notes",
        "Mentor", "Mentor_Email", "Mentor_Phone",
        "Mentor_Bio", "Mentor_Additional_Notes",
        "Compatibility_Score",
    ]]
    matched_df = matched_df.sort_values(by=["Mentee_Organization", "Mentee"])

matched_df.to_csv(BATCH_DIR / "matched_pairs.csv", index=False)

# ---------- Remaining mentors ----------
used_counts = matched_df["Mentor"].value_counts().to_dict()
remaining_capacity = {
    j: max(mentor_capacity.loc[j] - used_counts.get(j, 0), 0)
    for j in mentors_list
}
remaining_mentors = [j for j, cap in remaining_capacity.items() if cap > 0]

mentor_cols_ordered = [
    "Name", "Role", "Email", "Phone_Number", "Organization", "Location",
    "College", "Career_Interests", "Field_1", "Field_2", "Field_3",
    "Field_4", "Field_5", "Bio", "Additional_Notes",
    "Mentor_Level_Preferences", "Remaining_Capacity",
]

if remaining_mentors:
    full_remaining_df = mentors.set_index("Name").loc[remaining_mentors].copy()
    full_remaining_df["Remaining_Capacity"] = [remaining_capacity[j] for j in remaining_mentors]
    if "Mentor_Capacity" in full_remaining_df.columns:
        full_remaining_df = full_remaining_df.drop(columns=["Mentor_Capacity"])
    full_remaining_df = full_remaining_df.reset_index()
    full_remaining_df = full_remaining_df[mentor_cols_ordered]
else:
    full_remaining_df = pd.DataFrame(columns=mentor_cols_ordered)

full_remaining_df.to_csv(BATCH_DIR / "mentors_remaining_capacity.csv", index=False)

# ---------- Unmatched mentees ----------
matched_mentees = set(matched_df["Mentee"])
unmatched = [i for i in mentees_list if i not in matched_mentees]

if unmatched:
    unmatched_df = mentees.set_index("Name").loc[unmatched].copy()
    unmatched_df = unmatched_df.reset_index()
    if "Organization" in unmatched_df.columns:
        unmatched_df = unmatched_df.sort_values(by=["Organization", "Name"])
    else:
        unmatched_df = unmatched_df.sort_values(by=["Name"])
else:
    unmatched_df = pd.DataFrame(columns=list(mentees.columns))

unmatched_df.to_csv(BATCH_DIR / "unmatched_mentees.csv", index=False)

# ---------- Report + ZIP ----------
num_pairs     = len(matched_df)
num_unmatched = len(unmatched_df)
avg_score     = matched_df["Compatibility_Score"].mean() if num_pairs > 0 else 0.0

print(f"\n✅ Modified greedy matching complete.")
print(f"   Pairs matched : {num_pairs}")
print(f"   Unmatched     : {num_unmatched}")
print(f"   Avg comp score: {avg_score:.4f}")

zip_filename = "matching_algorithms_output.zip"
zip_path = BATCH_DIR / zip_filename
with zipfile.ZipFile(zip_path, "w") as z:
    z.write(BATCH_DIR / "matched_pairs.csv", arcname="matched_pairs.csv")
    z.write(BATCH_DIR / "mentors_remaining_capacity.csv", arcname="mentors_remaining_capacity.csv")
    z.write(BATCH_DIR / "unmatched_mentees.csv", arcname="unmatched_mentees.csv")

print(f"📦 ZIP saved to {zip_path}")

# ===============================
# CELL 2: Side-by-Side Comparison
# Original Greedy vs. Modified Greedy
# Run AFTER Cell 1 has executed
# ===============================
# This cell re-runs the ORIGINAL greedy logic internally (no file I/O)
# and compares it against the results already produced by Cell 1.
# Nothing is written to disk here — this is purely diagnostic.

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------
# Re-run original greedy (same data/params already loaded from Cell 1)
# -----------------------------------------------------------------------
def run_original_greedy():
    orig_pairs = []
    orig_used = {j: 0 for j in mentors_list}

    for mentee in mentees_list:          # original order — no sorting
        scores = []
        for mentor in mentors_list:
            if not is_feasible(mentee, mentor):
                continue
            if orig_used[mentor] >= mentor_capacity.loc[mentor]:
                continue
            scores.append((C.loc[mentee, mentor], mentor))
        if scores:
            best_score, best_mentor = max(scores, key=lambda t: t[0])
            orig_pairs.append((mentee, best_mentor, float(best_score)))
            orig_used[best_mentor] += 1

    return pd.DataFrame(orig_pairs, columns=["Mentee", "Mentor", "Compatibility_Score"])

orig_df = run_original_greedy()

# -----------------------------------------------------------------------
# matched_df is already available from Cell 1 (modified version)
# -----------------------------------------------------------------------
mod_df = matched_df.copy()

# -----------------------------------------------------------------------
# Helper: compute stats for a results dataframe
# -----------------------------------------------------------------------
def compute_stats(df, mentees_list):
    n_matched   = len(df)
    n_unmatched = len(mentees_list) - n_matched
    if n_matched == 0:
        return dict(matched=0, unmatched=n_unmatched, avg=0, med=0,
                    std=0, min_=0, max_=0, pct_above_08=0,
                    q1=0, q2=0, q3=0, q4=0)
    scores = df["Compatibility_Score"]
    # Split mentees into 4 quartile groups by their position in the
    # ORIGINAL ordering (earlier = more advantaged in original greedy)
    n = len(mentees_list)
    quartile_labels = {}
    for idx, name in enumerate(mentees_list):
        q = min(idx * 4 // n, 3)   # 0,1,2,3
        quartile_labels[name] = q

    q_scores = {0: [], 1: [], 2: [], 3: []}
    for _, row in df.iterrows():
        q = quartile_labels.get(row["Mentee"], 3)
        q_scores[q].append(row["Compatibility_Score"])

    def qavg(q):
        return np.mean(q_scores[q]) if q_scores[q] else float("nan")

    return dict(
        matched    = n_matched,
        unmatched  = n_unmatched,
        avg        = scores.mean(),
        med        = scores.median(),
        std        = scores.std(),
        min_       = scores.min(),
        max_       = scores.max(),
        pct_above_08 = (scores >= 0.80).sum() / n_matched * 100,
        q1         = qavg(0),
        q2         = qavg(1),
        q3         = qavg(2),
        q4         = qavg(3),
    )

orig_stats = compute_stats(orig_df, mentees_list)
mod_stats  = compute_stats(mod_df,  mentees_list)

# -----------------------------------------------------------------------
# Print comparison table
# -----------------------------------------------------------------------
SEP  = "=" * 62
SEP2 = "-" * 62

def fmt(val, is_pct=False, is_int=False):
    if np.isnan(val):
        return "  N/A "
    if is_int:
        return f"{int(val):>6}"
    if is_pct:
        return f"{val:>5.1f}%"
    return f"{val:>6.4f}"

def delta(o, m, higher_is_better=True, is_pct=False, is_int=False):
    if np.isnan(o) or np.isnan(m):
        return "     —"
    diff = m - o
    sign = "+" if diff >= 0 else ""
    if is_int:
        marker = "✅" if (diff > 0) == higher_is_better else ("🔴" if diff != 0 else "  ")
        return f"{marker} {sign}{int(diff)}"
    if is_pct:
        marker = "✅" if (diff > 0) == higher_is_better else ("🔴" if abs(diff) > 0.01 else "  ")
        return f"{marker} {sign}{diff:.1f}%"
    marker = "✅" if (diff > 0) == higher_is_better else ("🔴" if abs(diff) > 0.0001 else "  ")
    return f"{marker} {sign}{diff:.4f}"

print()
print(SEP)
print("  COMPARISON: Original Greedy vs. Modified Greedy")
print(SEP)
print(f"  {'Metric':<30} {'Original':>9}  {'Modified':>9}  {'Δ Change'}")
print(SEP2)

rows = [
    ("Pairs matched",         orig_stats["matched"],     mod_stats["matched"],     True,  False, True),
    ("Unmatched mentees",     orig_stats["unmatched"],   mod_stats["unmatched"],   False, False, True),
    ("Avg compatibility",     orig_stats["avg"],          mod_stats["avg"],          True,  False, False),
    ("Median compatibility",  orig_stats["med"],          mod_stats["med"],          True,  False, False),
    ("Std deviation",         orig_stats["std"],          mod_stats["std"],          False, False, False),
    ("Min score",             orig_stats["min_"],         mod_stats["min_"],         True,  False, False),
    ("Max score",             orig_stats["max_"],         mod_stats["max_"],         True,  False, False),
    ("% scores >= 0.80",      orig_stats["pct_above_08"],mod_stats["pct_above_08"], True,  True,  False),
]
for label, ov, mv, hib, is_pct, is_int in rows:
    print(f"  {label:<30} {fmt(ov, is_pct, is_int):>9}  {fmt(mv, is_pct, is_int):>9}  {delta(ov, mv, hib, is_pct, is_int)}")

print(SEP2)
print("  Score by mentee batch quartile (Q1=earliest, Q4=latest)")
print(SEP2)
for qi, label in enumerate(["Q1 (1st 25% of mentees)", "Q2 (2nd 25%)",
                              "Q3 (3rd 25%)", "Q4 (last 25% — latest batch)"]):
    ov = orig_stats[f"q{qi+1}"]
    mv = mod_stats[f"q{qi+1}"]
    print(f"  {label:<30} {fmt(ov):>9}  {fmt(mv):>9}  {delta(ov, mv, True, False, False)}")

print(SEP)
print()

# -----------------------------------------------------------------------
# Key insight summary
# -----------------------------------------------------------------------
avg_delta = mod_stats["avg"] - orig_stats["avg"]
q4_delta  = mod_stats["q4"]  - orig_stats["q4"]
match_delta = mod_stats["matched"] - orig_stats["matched"]

print("📋 KEY INSIGHTS")
print(SEP2)
if match_delta == 0:
    print("  • Same number of total matches — no mentees lost.")
elif match_delta > 0:
    print(f"  • {match_delta} MORE mentees matched vs. original.")
else:
    print(f"  • {abs(match_delta)} fewer mentees matched (trade-off of reserving popular mentors).")

if avg_delta >= 0:
    print(f"  • Overall avg score {'improved' if avg_delta > 0.0001 else 'unchanged'} by {avg_delta:+.4f}.")
else:
    print(f"  • Overall avg score decreased by {avg_delta:.4f} (expected trade-off).")

if not np.isnan(q4_delta):
    if q4_delta > 0.001:
        print(f"  • Later-batch mentees (Q4) improved by {q4_delta:+.4f} — core goal achieved ✅")
    elif q4_delta < -0.001:
        print(f"  • Later-batch mentees (Q4) decreased by {q4_delta:.4f} — tune parameters.")
    else:
        print(f"  • Later-batch mentees (Q4) score roughly unchanged.")

std_delta = mod_stats["std"] - orig_stats["std"]
if std_delta < -0.001:
    print(f"  • Score std dev decreased by {std_delta:.4f} — matches are more equitable ✅")
elif std_delta > 0.001:
    print(f"  • Score std dev increased by {std_delta:+.4f} — more variance in outcomes.")

print(SEP)
print()
print("Parameters used in modified version:")
print(f"  USE_CONSTRAINED_PRIORITY  = {USE_CONSTRAINED_PRIORITY}")
print(f"  USE_POPULARITY_RESERVE    = {USE_POPULARITY_RESERVE}")
print(f"  POPULARITY_THRESHOLD      = {POPULARITY_THRESHOLD}  (top {int(POPULARITY_THRESHOLD*100)}% flagged popular)")
print(f"  POPULAR_BATCH_RESERVE_FRAC= {POPULAR_BATCH_RESERVE_FRAC}  ({int(POPULAR_BATCH_RESERVE_FRAC*100)}% cap per batch)")
