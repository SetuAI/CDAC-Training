
# CDAC-Training

Teaching materials for the CDAC training programme — slide content, runnable teaching code, and capstone case studies.

Maintained by [Tarka Upskilling and Engineering Co.](https://tarkaupskilling.com)

---

## Day 1 — Descriptive Statistics

`Day1_materials/`

Mean, median, standard deviation, skewness and kurtosis, taught through a single running case study (Meridian Retail) so the same numbers carry across every file.

### Teaching code

Each file is standalone. Open any one and run it top to bottom — no imports between them, no setup beyond the dependencies below.

| File                          | Covers                                                            |
| ----------------------------- | ----------------------------------------------------------------- |
| `01_case_study_and_data.py` | The case study, five-number summary, full statistical profile     |
| `02_mean.py`                | Mean, deviations summing to zero, weighted mean, trimmed mean     |
| `03_median_and_mode.py`     | Median, breakdown point, percentiles, IQR fence, mode, bimodality |
| `04_standard_deviation.py`  | Variance, standard deviation, n vs n−1, CV, z-scores             |
| `05_skewness.py`            | Skewness, right and left skew, log transform                      |
| `06_kurtosis.py`            | Kurtosis, tail regimes, tail risk, full diagnostic workflow       |

Run any file:

```bash
python3 01_case_study_and_data.py
```

Every from-scratch calculation is checked against pandas with an `assert`, so an edit that breaks the maths fails loudly instead of printing a plausible wrong number.

### Capstone case study

| File                    | What it is                                                      |
| ----------------------- | --------------------------------------------------------------- |
| `PROJECT.md`          | The brief — business situation, data dictionary, ten questions |
| `meridian_orders.csv` | The dataset: 5,000 orders, 18 columns                           |

Start with `PROJECT.md`. It opens with an integrity check to run before computing anything.

The ten questions map to the topics above: centre (Q1, Q2), weighted mean (Q3), spread (Q4, Q5), percentiles (Q6), skewness (Q7, Q8), kurtosis (Q9), and segmentation with outlier detection (Q10).

**About the data.** Synthetic, generated with a fixed random seed so every run reproduces the same file. Meridian Retail is not a real company. The statistical patterns in it are deliberately engineered — a genuine right skew in order value, a genuine left skew in customer ratings, genuine fat tails in delivery times — so the answers to the ten questions exist to be found.

`order_value` reconciles exactly to `units × unit_price × (1 − discount_pct/100)`. Verify it before you trust it.

---

## Requirements

```bash
pip install pandas numpy scipy openpyxl
```

Python 3.9 or later.

---

## Licence

MIT — see [LICENSE](LICENSE).
