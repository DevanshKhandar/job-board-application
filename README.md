# Student Job Board Application Funnel Analysis (Jul–Sep 2026)

Anonymous submission (no name, college, or personal data included).

## Files Included
- `job-board-applications.csv` — Original raw dataset (1,206 rows).
- `analysis.py` — Fully reproducible Python script that cleans the data, prints all answers and statistical checks, exports `cleaned_applications.csv`, and generates `chart_q3_paid_vs_unpaid.png`.
- `cleaned_applications.csv` — Deduplicated, standardised dataset (1,200 rows) produced by `analysis.py`.
- `chart_q3_paid_vs_unpaid.png` — Two-panel chart answering Question 3 (Paid vs. Unpaid reply speed and no-reply gap).
- `manager_note.md` — ~200-word note to the manager covering all three answers, next-month recommendation, and data caveats.

## How to Reproduce Every Number and Chart
Requires Python 3.9+ with `pandas`, `numpy`, `scipy`, and `matplotlib`:

```bash
python analysis.py
```

Running this single command regenerates `cleaned_applications.csv`, `chart_q3_paid_vs_unpaid.png`, and prints the exact cleaning audit and answers to Questions 1–3.

## Summary of Data Cleaning Fixes
1. **Removed 6 exact duplicate rows** (`A100227`, `A100274`, `A100497`, `A100972`, `A101102`, `A101161`), yielding 1,200 unique applications (`A100000`–`A101199`).
2. **Standardised 9 `work_mode` values** (`On-site` → `on_site`, `Hybrid` → `hybrid`, `Remote` → `remote`).
3. **Handled 15 missing `source` values** by keeping them in the dataset for Questions 2 and 3, while excluding them from named source rankings in Question 1.
4. **Verified funnel and date integrity:** all dates fall between `2026-07-01` and `2026-09-30`, with zero stage-order contradictions. Flagged 11 `steps_done = 0` applications that were still shortlisted (3 hired) as a tracking anomaly.

## Tool Use Disclosure (edit before submitting)
An AI assistant helped write `analysis.py`, design the chart layout, and draft `manager_note.md`. **[Before submitting, note what you reviewed, verified, or modified yourself.]**
