"""
Student Job Board Application Funnel Analysis (July–September 2026)
==================================================================
Reproducible script that cleans `job-board-applications.csv`, answers all three
planning questions, prints statistical verification checks, exports cleaned data,
and generates the Q3 summary chart (`chart_q3_paid_vs_unpaid.png`).

Usage:
    python analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load raw CSV, apply documented cleaning fixes, and return clean DataFrame + audit stats."""
    raw = pd.read_csv(csv_path)
    audit = {"raw_rows": len(raw)}

    # 1. Remove exact duplicate rows (all 6 duplicate application_ids are exact row duplicates)
    exact_dups = int(raw.duplicated().sum())
    dup_ids = sorted(raw.loc[raw.duplicated(subset=["application_id"], keep=False), "application_id"].unique().tolist())
    df = raw.drop_duplicates().copy()
    assert df["application_id"].is_unique, "Unexpected non-exact duplicate application_id found!"
    audit["exact_duplicates_removed"] = exact_dups
    audit["duplicate_ids"] = dup_ids
    audit["clean_rows"] = len(df)

    # 2. Standardise inconsistent work_mode labels (e.g., 'On-site' -> 'on_site', 'Hybrid' -> 'hybrid')
    cleaned_wm = df["work_mode"].str.strip().str.lower().str.replace("-", "_", regex=False)
    wm_fixes = int((df["work_mode"] != cleaned_wm).sum())
    df["work_mode"] = cleaned_wm
    assert set(df["work_mode"].unique()) == {"remote", "hybrid", "on_site"}
    audit["work_mode_fixes"] = wm_fixes

    # 3. Document missing sources (retained for Q2 & Q3; excluded from named source rankings in Q1)
    audit["missing_source_rows"] = int(df["source"].isna().sum())

    # 4. Validate dates, ranges, and funnel integrity
    df["applied_dt"] = pd.to_datetime(df["applied_on"], format="%Y-%m-%d", errors="raise")
    assert df["applied_dt"].min() >= pd.Timestamp("2026-07-01")
    assert df["applied_dt"].max() <= pd.Timestamp("2026-09-30")
    assert df["steps_done"].between(0, 4).all()
    assert (df["stipend_inr_month"] >= 0).all()

    # Funnel logic checks: interviewed implies shortlisted; hired implies interviewed
    assert ((df["interviewed"] == "yes") & (df["shortlisted"] == "no")).sum() == 0
    assert ((df["hired"] == "yes") & (df["interviewed"] == "no")).sum() == 0
    assert (df["days_to_first_reply"].isna() & (df["shortlisted"] == "yes")).sum() == 0

    # 5. Derived helper flags
    df["is_shortlisted"] = (df["shortlisted"] == "yes").astype(int)
    df["is_interviewed"] = (df["interviewed"] == "yes").astype(int)
    df["is_hired"] = (df["hired"] == "yes").astype(int)
    df["all_4_steps"] = df["steps_done"] == 4
    df["is_paid"] = df["stipend_inr_month"] > 0
    df["no_reply"] = df["days_to_first_reply"].isna()

    # Flag data anomaly: 0 steps completed yet shortlisted/hired
    audit["zero_step_shortlisted"] = int(((df["steps_done"] == 0) & (df["is_shortlisted"] == 1)).sum())
    audit["zero_step_hired"] = int(((df["steps_done"] == 0) & (df["is_hired"] == 1)).sum())

    return df, audit


def analyze_q1(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q1: Hires per 100 applications by source, overall and by role category."""
    known = df[df["source"].notna()].copy()

    overall = (
        known.groupby("source")
        .agg(
            applications=("application_id", "count"),
            shortlisted=("is_shortlisted", "sum"),
            interviewed=("is_interviewed", "sum"),
            hired=("is_hired", "sum"),
        )
        .assign(hires_per_100=lambda d: (d["hired"] / d["applications"] * 100).round(2))
        .sort_values("hires_per_100", ascending=False)
    )

    apps_ct = pd.crosstab(known["role_category"], known["source"])
    hires_ct = pd.crosstab(known["role_category"], known["source"], values=known["is_hired"], aggfunc="sum").fillna(0).astype(int)
    rates_ct = (hires_ct / apps_ct * 100).round(2)

    role_summary_rows = []
    for role in rates_ct.index:
        top_rate = rates_ct.loc[role].max()
        top_src = rates_ct.loc[role].idxmax()
        role_summary_rows.append(
            {
                "role_category": role,
                "top_source": top_src,
                "top_hires": int(hires_ct.loc[role, top_src]),
                "top_apps": int(apps_ct.loc[role, top_src]),
                "top_hires_per_100": top_rate,
                "referral_hires": int(hires_ct.loc[role, "referral"]),
                "referral_apps": int(apps_ct.loc[role, "referral"]),
                "referral_hires_per_100": rates_ct.loc[role, "referral"],
            }
        )
    role_summary = pd.DataFrame(role_summary_rows).set_index("role_category")
    return overall, role_summary


def analyze_q2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Q2: Shortlist rates for 4 steps completed vs 0-3 steps."""
    grouped = (
        df.groupby("all_4_steps")
        .agg(
            applications=("application_id", "count"),
            shortlisted=("is_shortlisted", "sum"),
            hired=("is_hired", "sum"),
        )
        .assign(
            shortlist_rate_pct=lambda d: (d["shortlisted"] / d["applications"] * 100).round(2),
            hire_rate_pct=lambda d: (d["hired"] / d["applications"] * 100).round(2),
        )
    )
    grouped.index = grouped.index.map({False: "0-3 steps", True: "All 4 steps"})

    by_step = (
        df.groupby("steps_done")
        .agg(
            applications=("application_id", "count"),
            shortlisted=("is_shortlisted", "sum"),
            hired=("is_hired", "sum"),
        )
        .assign(shortlist_rate_pct=lambda d: (d["shortlisted"] / d["applications"] * 100).round(2))
    )

    ct = pd.crosstab(df["all_4_steps"], df["is_shortlisted"])
    chi2, p_val, _, _ = stats.chi2_contingency(ct)
    r4 = df.loc[df["all_4_steps"], "is_shortlisted"].mean() * 100
    r03 = df.loc[~df["all_4_steps"], "is_shortlisted"].mean() * 100
    metrics = {
        "rate_4_steps": r4,
        "rate_0_3_steps": r03,
        "diff_pp": r4 - r03,
        "ratio": r4 / r03,
        "chi2": chi2,
        "p_value": p_val,
    }
    return grouped, by_step, metrics


def analyze_q3(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Q3: Paid vs unpaid reply speed and no-reply share, plus statistical & censoring checks."""
    summary = (
        df.groupby("is_paid")
        .agg(
            applications=("application_id", "count"),
            replied=("days_to_first_reply", lambda s: int(s.notna().sum())),
            no_reply=("no_reply", "sum"),
            no_reply_pct=("no_reply", lambda s: round(s.mean() * 100, 2)),
            median_days=("days_to_first_reply", "median"),
            mean_days=("days_to_first_reply", lambda s: round(s.mean(), 2)),
            p25_days=("days_to_first_reply", lambda s: s.quantile(0.25)),
            p75_days=("days_to_first_reply", lambda s: s.quantile(0.75)),
            max_days=("days_to_first_reply", "max"),
        )
    )
    summary.index = summary.index.map({False: "Unpaid (Rs 0)", True: "Paid (> Rs 0)"})

    paid_days = df.loc[df["is_paid"], "days_to_first_reply"].dropna()
    unpaid_days = df.loc[~df["is_paid"], "days_to_first_reply"].dropna()
    _, p_mwu = stats.mannwhitneyu(paid_days, unpaid_days, alternative="two-sided")

    ct_nr = pd.crosstab(df["is_paid"], df["no_reply"])
    chi2_nr, p_nr, _, _ = stats.chi2_contingency(ct_nr)

    # Right-censoring sensitivity check: exclude final 14 days of September (applied_on <= 2026-09-16)
    sub = df[df["applied_dt"] <= "2026-09-16"]
    sens_paid_nr = sub.loc[sub["is_paid"], "no_reply"].mean() * 100
    sens_unpaid_nr = sub.loc[~sub["is_paid"], "no_reply"].mean() * 100

    metrics = {
        "median_diff_days": float(unpaid_days.median() - paid_days.median()),
        "mean_diff_days": float(unpaid_days.mean() - paid_days.mean()),
        "no_reply_diff_pp": float((df.loc[~df["is_paid"], "no_reply"].mean() - df.loc[df["is_paid"], "no_reply"].mean()) * 100),
        "no_reply_ratio": float(df.loc[~df["is_paid"], "no_reply"].mean() / df.loc[df["is_paid"], "no_reply"].mean()),
        "unpaid_share_pct": float((~df["is_paid"]).mean() * 100),
        "p_mwu": float(p_mwu),
        "chi2_nr": float(chi2_nr),
        "p_nr": float(p_nr),
        "sens_paid_nr_pct": float(sens_paid_nr),
        "sens_unpaid_nr_pct": float(sens_unpaid_nr),
    }
    return summary, metrics


def make_q3_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Generate clean two-panel executive chart answering Question 3 (Paid vs. Unpaid responsiveness)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.8,
        }
    )

    paid_color = "#1B6B7A"     # Muted deep teal
    unpaid_color = "#C85A3A"   # Muted terracotta
    text_dark = "#1F2933"
    text_muted = "#52606D"
    grid_color = "#E4E7EB"

    paid_days = df.loc[df["is_paid"], "days_to_first_reply"].dropna().values
    unpaid_days = df.loc[~df["is_paid"], "days_to_first_reply"].dropna().values

    paid_nr_pct = df.loc[df["is_paid"], "no_reply"].mean() * 100
    unpaid_nr_pct = df.loc[~df["is_paid"], "no_reply"].mean() * 100
    paid_nr_n = int(df.loc[df["is_paid"], "no_reply"].sum())
    unpaid_nr_n = int(df.loc[~df["is_paid"], "no_reply"].sum())
    paid_tot = int(df["is_paid"].sum())
    unpaid_tot = int((~df["is_paid"]).sum())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.2, 5.4), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.28}
    )
    fig.patch.set_facecolor("white")

    # ---- Panel A: Reply time distribution (Horizontal boxplot + subtle jittered points) ----
    rng = np.random.default_rng(42)
    y_paid, y_unpaid = 1, 0

    for y_pos, vals, color in [(y_paid, paid_days, paid_color), (y_unpaid, unpaid_days, unpaid_color)]:
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax1.scatter(vals, np.full_like(vals, y_pos, dtype=float) + jitter, color=color, alpha=0.14, s=16, edgecolors="none", zorder=2)

    bp = ax1.boxplot(
        [unpaid_days, paid_days],
        positions=[y_unpaid, y_paid],
        vert=False,
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2.4},
        whiskerprops={"color": text_muted, "linewidth": 1.3},
        capprops={"color": text_muted, "linewidth": 1.3},
        boxprops={"linewidth": 1.2},
        zorder=3,
    )
    bp["boxes"][0].set(facecolor=unpaid_color, alpha=0.88, edgecolor=unpaid_color)
    bp["boxes"][1].set(facecolor=paid_color, alpha=0.88, edgecolor=paid_color)

    # Mean markers (diamond)
    ax1.scatter([unpaid_days.mean(), paid_days.mean()], [y_unpaid, y_paid], marker="D", s=42, color="#1F2933", edgecolor="white", linewidth=0.9, zorder=5, label="Mean")

    # Direct stat labels
    ax1.text(
        np.median(paid_days), y_paid + 0.28,
        f"Median: {np.median(paid_days):.0f} days  (IQR: 2–4  |  Mean: {paid_days.mean():.1f})",
        color=paid_color, fontsize=9.5, fontweight="bold", ha="left", va="bottom"
    )
    ax1.text(
        np.median(unpaid_days), y_unpaid + 0.28,
        f"Median: {np.median(unpaid_days):.0f} days  (IQR: 4–10  |  Mean: {unpaid_days.mean():.1f})",
        color=unpaid_color, fontsize=9.5, fontweight="bold", ha="left", va="bottom"
    )

    # Gap callout between medians
    ax1.annotate(
        "", xy=(7.0, 0.48), xytext=(3.0, 0.48),
        arrowprops=dict(arrowstyle="<->", color=text_dark, lw=1.3)
    )
    ax1.text(5.0, 0.51, "+4 days slower\n(2.3× median)", color=text_dark, fontsize=8.8, fontweight="bold", ha="center", va="bottom")

    ax1.set_yticks([y_unpaid, y_paid])
    ax1.set_yticklabels([f"Unpaid roles\n(n = {len(unpaid_days)} replied)", f"Paid roles\n(n = {len(paid_days)} replied)"], fontsize=9.8, color=text_dark, fontweight="bold")
    ax1.set_xlim(0, 22)
    ax1.set_xticks(range(0, 23, 2))
    ax1.set_ylim(-0.55, 1.62)
    ax1.set_xlabel("Days to first employer reply (axis capped at 22; 3 unpaid replies took 25–32 days)", fontsize=8.6, color=text_muted, labelpad=8)
    ax1.set_title("A. Speed of First Reply (Among Replied Applications)", fontsize=10.8, fontweight="bold", color=text_dark, loc="left", pad=12)
    ax1.grid(axis="x", color=grid_color, linestyle="-", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E4E7EB", fontsize=8.5)

    # ---- Panel B: Share of applicants with no reply ----
    bars = ax2.bar(
        [0, 1],
        [paid_nr_pct, unpaid_nr_pct],
        width=0.50,
        color=[paid_color, unpaid_color],
        alpha=0.92,
        zorder=3,
    )

    for bar, pct, nr_n, tot_n, col in zip(
        bars, [paid_nr_pct, unpaid_nr_pct], [paid_nr_n, unpaid_nr_n], [paid_tot, unpaid_tot], [paid_color, unpaid_color]
    ):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 1.0,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=13.5, fontweight="bold", color=col
        )
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            pct / 2,
            f"{nr_n} of {tot_n}\napplications",
            ha="center", va="center", fontsize=9.0, color="white", fontweight="bold"
        )

    # Difference callout pointing cleanly to the left edge of the unpaid bar
    ax2.annotate(
        "+16.5 pp gap\n(2.3× as likely\nto get no reply)",
        xy=(0.74, unpaid_nr_pct - 2.0),
        xytext=(0.14, 29.8),
        ha="center", va="center",
        fontsize=8.8, fontweight="bold", color=text_dark,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#FDF2EE", edgecolor=unpaid_color, linewidth=1.0),
        arrowprops=dict(arrowstyle="->", color=unpaid_color, lw=1.4, connectionstyle="arc3,rad=-0.10"),
    )

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Paid roles\n(stipend > ₹0)", "Unpaid roles\n(stipend = ₹0)"], fontsize=9.8, color=text_dark, fontweight="bold")
    ax2.set_ylim(0, 38)
    ax2.set_ylabel("Share of applications with no reply (%)", fontsize=9.4, color=text_muted)
    ax2.set_title("B. Share of Applicants Never Hearing Back", fontsize=10.8, fontweight="bold", color=text_dark, loc="left", pad=12)
    ax2.grid(axis="y", color=grid_color, linestyle="-", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Overall title, subtitle & footnote
    fig.suptitle(
        "Unpaid Roles Reply Over Twice as Slowly and Leave 2.3× More Students Without a Reply",
        fontsize=13.2, fontweight="bold", color=text_dark, x=0.04, y=0.968, ha="left"
    )
    fig.text(
        0.04, 0.908,
        "Student job board applications, July–September 2026  |  n = 1,200 unique applications (734 paid, 466 unpaid; 6 exact duplicates removed)",
        fontsize=9.3, color=text_muted, ha="left"
    )
    fig.text(
        0.04, 0.025,
        "Note: Unpaid listings represent 38.8% of all applications. Both gaps are statistically significant (Mann-Whitney U & Chi-square, p < 0.001) and persist when excluding late-September applications.",
        fontsize=8.0, color=text_muted, ha="left"
    )

    fig.subplots_adjust(left=0.145, right=0.97, top=0.80, bottom=0.16)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "job-board-applications.csv"

    df, audit = load_and_clean_data(csv_path)

    print("=== DATA CLEANING AUDIT ===")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    print("\n=== Q1: HIRES PER 100 APPLICATIONS BY SOURCE ===")
    q1_overall, q1_by_role = analyze_q1(df)
    print(q1_overall.to_string())
    print("\nQ1 by Role Category (Top Source vs Referral):")
    print(q1_by_role.to_string())

    print("\n=== Q2: SHORTLIST RATE — ALL 4 STEPS VS 0-3 STEPS ===")
    q2_grouped, q2_by_step, q2_metrics = analyze_q2(df)
    print(q2_grouped.to_string())
    print("\nBreakdown by step count:")
    print(q2_by_step.to_string())
    print(
        f"\nLift: +{q2_metrics['diff_pp']:.2f} pp ({q2_metrics['ratio']:.2f}x), "
        f"Chi2={q2_metrics['chi2']:.2f}, p={q2_metrics['p_value']:.2e}"
    )

    print("\n=== Q3: PAID VS UNPAID RESPONSIVENESS ===")
    q3_summary, q3_metrics = analyze_q3(df)
    print(q3_summary.to_string())
    print(
        f"\nMedian diff: +{q3_metrics['median_diff_days']:.1f} days | Mean diff: +{q3_metrics['mean_diff_days']:.2f} days "
        f"(MWU p={q3_metrics['p_mwu']:.2e})"
    )
    print(
        f"No-reply gap: +{q3_metrics['no_reply_diff_pp']:.2f} pp ({q3_metrics['no_reply_ratio']:.2f}x, "
        f"Chi2={q3_metrics['chi2_nr']:.2f}, p={q3_metrics['p_nr']:.2e})"
    )
    print(
        f"Censoring sensitivity (<= 2026-09-16): Paid no-reply={q3_metrics['sens_paid_nr_pct']:.2f}%, "
        f"Unpaid no-reply={q3_metrics['sens_unpaid_nr_pct']:.2f}%"
    )

    # Export cleaned dataset and chart
    clean_cols = [
        "application_id", "applied_on", "role_category", "work_mode",
        "stipend_inr_month", "source", "steps_done", "shortlisted",
        "interviewed", "hired", "days_to_first_reply"
    ]
    df[clean_cols].to_csv(base_dir / "cleaned_applications.csv", index=False)
    make_q3_chart(df, base_dir / "chart_q3_paid_vs_unpaid.png")
    print("\nSaved cleaned_applications.csv and chart_q3_paid_vs_unpaid.png successfully.")


if __name__ == "__main__":
    main()
