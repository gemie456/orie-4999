import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

BATCH_DIR   = Path(".")
RESULTS_DIR = BATCH_DIR / "results"


def evaluate_batch(batch_path):
    matched           = pd.read_csv(batch_path / "matched_pairs.csv")
    unmatched         = pd.read_csv(batch_path / "unmatched_mentees.csv")
    mentors_remaining = pd.read_csv(batch_path / "mentors_remaining_capacity.csv")

    mentors = pd.read_csv(batch_path / "mentors_clean.csv")
    mentees = pd.read_csv(batch_path / "mentees_clean.csv")  # FIXED: load mentees

    mentor_field_map = (
        mentors.set_index("Name")["Field_1"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .to_dict()
    )

    # FIXED: build a separate map for mentees
    mentee_field_map = (
        mentees.set_index("Name")["Field_1"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .to_dict()
    )

# ===============================
# Popular field evaluation (FIXED)
# ===============================

    # Build full mentee pool safely
    all_mentees = []

    if len(matched) > 0 and "Mentee" in matched.columns:
        all_mentees.append(matched[["Mentee"]])

    if len(unmatched) > 0 and "Mentee" in unmatched.columns:
        all_mentees.append(unmatched[["Mentee"]])

    if len(all_mentees) > 0:
        all_mentees = pd.concat(all_mentees)
        all_mentees["Field"] = all_mentees["Mentee"].map(mentee_field_map)  # FIXED
        field_demand = all_mentees["Field"].value_counts()
    else:
        field_demand = pd.Series(dtype=int)

    top_fields = set(field_demand.head(max(1, int(0.2 * len(field_demand)))).index)

    # Coverage
    n_matched   = len(matched)
    n_unmatched = len(unmatched)
    coverage    = n_matched / (n_matched + n_unmatched) if (n_matched + n_unmatched) > 0 else 0

    # Compatibility score stats
    mean_score   = matched["Compatibility_Score"].mean()
    median_score = matched["Compatibility_Score"].median()
    std_score    = matched["Compatibility_Score"].std()
    score_min   = matched["Compatibility_Score"].min() if len(matched) > 0 else 0
    score_max   = matched["Compatibility_Score"].max() if len(matched) > 0 else 0
    score_range = score_max - score_min

    if len(matched) > 0 and "Mentee" in matched.columns:
        matched["Mentee_Field"] = matched["Mentee"].map(mentee_field_map)  # FIXED

        popular_mask = matched["Mentee_Field"].isin(top_fields)

        popular_mean_score = (
            matched.loc[popular_mask, "Compatibility_Score"].mean()
            if popular_mask.any()
            else 0
        )
    else:
        popular_mean_score = 0

    if len(unmatched) > 0 and "Mentee" in unmatched.columns:
        unmatched["Field"] = unmatched["Mentee"].map(mentee_field_map)  # FIXED

        popular_unmatched_rate = (
            unmatched["Field"].isin(top_fields).mean()
            if len(unmatched) > 0
            else 0
        )
    else:
        popular_unmatched_rate = 0

    # Mentor depletion
    remaining_capacity = mentors_remaining["Remaining_Capacity"].sum()

    # Opportunity inequality
    comp_matrix_loaded = pd.read_csv(batch_path / "compatibility_matrix.csv", index_col=0)
    comp_long          = comp_matrix_loaded.stack().reset_index()
    comp_long.columns  = ["mentor_id", "mentee_id", "Compatibility_Score"]
    compatible         = comp_long[comp_long["Compatibility_Score"] > 0]
    opp_counts         = compatible.groupby("mentee_id")["mentor_id"].nunique()
    avg_opportunities  = opp_counts.mean()

    # Mentor load balance
    mentor_loads = matched["Mentor"].value_counts()
    load_mean    = mentor_loads.mean()
    load_std     = mentor_loads.std()
    load_cv      = load_std / load_mean if load_mean > 0 else 0

    loads_sorted = sorted(mentor_loads.values)
    n = len(loads_sorted)
    if sum(loads_sorted) == 0:
        gini = 0
    elif n == 0:
        gini = -1
    else:
        gini = (
        2 * sum((i + 1) * v for i, v in enumerate(loads_sorted))
    ) / (n * sum(loads_sorted)) - (n + 1) / n

    # Score percentile distribution
    pct_above_05 = (matched["Compatibility_Score"] >= 0.50).mean()
    pct_above_06 = (matched["Compatibility_Score"] >= 0.60).mean()
    pct_above_07 = (matched["Compatibility_Score"] >= 0.70).mean()
    pct_above_08 = (matched["Compatibility_Score"] >= 0.80).mean()
    pct_above_09 = (matched["Compatibility_Score"] >= 0.90).mean()

    # ===============================
    # Remaining capacity by mentor field (ROBUST VERSION)
    # ===============================

    # Use mentor table as source of truth (NOT matched)
    mentors_remaining["Field"] = mentors_remaining["Name"].map(mentor_field_map)

    remaining_by_field = {
        f"remaining_capacity_field_{k}": v
        for k, v in (
            mentors_remaining
            .groupby("Field")["Remaining_Capacity"]
            .sum()
            .to_dict()
            .items()
        )
    }

    return {
        "batch": batch_path.name,
        # Coverage
        "n_matched":   n_matched,
        "n_unmatched": n_unmatched,
        "coverage":    coverage,
        # Score stats
        "popular_mean_score": popular_mean_score,
        "popular_unmatched_rate": popular_unmatched_rate,
        "mean_score":   mean_score,
        "median_score": median_score,
        "std_score":    std_score,
        "score_range": score_range,
        # Mentor depletion
        "remaining_capacity":     remaining_capacity,
        # Opportunity inequality
        "avg_compatible_mentors": avg_opportunities,
        # Mentor load balance
        "load_mean": load_mean,
        "load_std":  load_std,
        "load_cv":   load_cv,
        "load_gini": gini,
        # Score percentile distribution
        "pct_above_0.50": pct_above_05,
        "pct_above_0.60": pct_above_06,
        "pct_above_0.70": pct_above_07,
        "pct_above_0.80": pct_above_08,
        "pct_above_0.90": pct_above_09,
        # Remaining capacity by top field
        **remaining_by_field,
    }


def evaluate_all_batches():
    batch_folders = sorted(RESULTS_DIR.glob("batch*"))
    if not batch_folders:
        print(f"No batch folders found in '{RESULTS_DIR}'. Make sure results are saved there.")
        return pd.DataFrame()
    results = [evaluate_batch(b) for b in batch_folders]
    df = pd.DataFrame(results)
    df = df.sort_values("batch")
    return df


def print_metrics(df: pd.DataFrame) -> None:
    console = Console()

    for _, row in df.iterrows():
        console.rule(f"[bold]{row['batch']}[/bold]", style="dim")

        # --- Coverage ---
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  title="Coverage", title_style="bold")
        t.add_column("Matched",                justify="right")
        t.add_column("Unmatched",              justify="right")
        t.add_column("Coverage",               justify="right")
        t.add_column("Avg compatible mentors", justify="right")
        t.add_row(
            str(int(row["n_matched"])),
            str(int(row["n_unmatched"])),
            f"{row['coverage']:.1%}",
            f"{row['avg_compatible_mentors']:.1f}",
        )
        console.print(t)

        # --- Popular field dynamics ---
        p = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            title="Popular Field Dynamics",
            title_style="bold"
        )
        p.add_column("Popular mean score", justify="right")
        p.add_column("Popular unmatched rate", justify="right")
        p.add_row(
            f"{row['popular_mean_score']:.3f}",
            f"{row['popular_unmatched_rate']:.2%}",
        )
        console.print(p)

        # --- Scores ---
        s = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            title="Compatibility scores",
            title_style="bold"
        )
        s.add_column("Mean",   justify="right")
        s.add_column("Median", justify="right")
        s.add_column("Std",    justify="right")
        s.add_column("≥ 0.50", justify="right")
        s.add_column("≥ 0.60", justify="right")
        s.add_column("≥ 0.70", justify="right")
        s.add_column("≥ 0.80", justify="right")
        s.add_column("≥ 0.90", justify="right")
        s.add_column("Range",  justify="right")
        s.add_row(
            f"{row['mean_score']:.3f}",
            f"{row['median_score']:.3f}",
            f"{row['std_score']:.3f}",
            f"{row['pct_above_0.50']:.0%}",
            f"{row['pct_above_0.60']:.0%}",
            f"{row['pct_above_0.70']:.0%}",
            f"{row['pct_above_0.80']:.0%}",
            f"{row['pct_above_0.90']:.0%}",
            f"{row['score_range']:.3f}",
        )
        console.print(s)

        # --- Load balance ---
        l = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  title="Mentor load", title_style="bold")
        l.add_column("Remaining capacity", justify="right")
        l.add_column("Load mean",          justify="right")
        l.add_column("Load CV",            justify="right")
        l.add_column("Gini",               justify="right")
        gini_color = (
            "green"  if row["load_gini"] < 0.2 else
            "yellow" if row["load_gini"] < 0.4 else
            "red"
        )
        l.add_row(
            str(int(row["remaining_capacity"])),
            f"{row['load_mean']:.1f}",
            f"{row['load_cv']:.3f}",
            f"[{gini_color}]{row['load_gini']:.3f}[/{gini_color}]",
        )
        console.print(l)

        # --- Remaining capacity by field (if present) ---
        field_cols = [c for c in row.index if c.startswith("remaining_capacity_field_")]
        if field_cols:
            f_table = Table(
                box=box.SIMPLE,
                show_header=True,
                header_style="bold dim",
                title="Remaining capacity by field (Mentor_Field_1)",
                title_style="bold",
                expand=True
            )
            for col in field_cols:
                f_table.add_column(col.replace("remaining_capacity_field_", ""), justify="right")
            f_table.add_row(*[str(int(row[c])) if pd.notna(row[c]) else "0" for c in field_cols])
            console.print(f_table)

        console.print()


if __name__ == "__main__":
    console = Console()
    summary = evaluate_all_batches()

    if summary.empty:
        console.print("[red]No data to display.[/red]")
    else:
        console.print("\n[bold underline]Batch Evaluation Summary[/bold underline]\n")
        print_metrics(summary)
        summary.to_csv(BATCH_DIR / "algorithm_evaluation_summary.csv", index=False)
        console.print("[green]✅ Summary saved to algorithm_evaluation_summary.csv[/green]")
