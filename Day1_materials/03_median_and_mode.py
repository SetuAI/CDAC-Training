"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL
================================================================================

Meridian Retail pulls one day of orders from its Andheri store. Nine orders
came through, in thousands of rupees:

        12, 14, 15, 15, 18, 20, 22, 25, 99

Eight are ordinary walk-in orders. The ninth is a corporate bulk order.
In file 02 we saw the mean report 26.67 -- a value higher than eight of the
nine orders. This file builds the statistic that does not have that problem.

--------------------------------------------------------------------------------
THIS FILE: THE MEDIAN, PERCENTILES AND THE MODE
--------------------------------------------------------------------------------

The median is the value at the middle POSITION of the sorted data. It does not
care how large the largest value is -- only that it is largest. That single
difference is what makes it resistant.

Sections:
        1. Computing the median by hand (odd and even n)
        2. Robustness -- the breakdown point
        3. Percentiles, quartiles and the IQR
        4. The IQR outlier fence
        5. The mode, and what bimodality is telling you
================================================================================
"""

import pandas as pd


ORDER_VALUES = [12, 14, 15, 15, 18, 20, 22, 25, 99]
ORDER_VALUES_NO_BULK = [12, 14, 15, 15, 18, 20, 22, 25]

orders = pd.Series(ORDER_VALUES, dtype=float)
orders_no_bulk = pd.Series(ORDER_VALUES_NO_BULK, dtype=float)


# ==============================================================================
# SECTION 1 -- COMPUTING THE MEDIAN FROM SCRATCH
# ==============================================================================

def median_from_scratch(values):
    """
    Find the middle value of the sorted data.

    Two cases:
        n is ODD  -> one value sits exactly in the middle. Take it.
        n is EVEN -> two values share the middle. Average them.

    Notice what this function never does: it never adds up the values. It only
    sorts them and counts positions. That is the entire reason the median
    survives an outlier -- the size of the largest value is irrelevant to it.
    """
    # sorted() returns a new sorted list, leaving the caller's list untouched.
    ordered = sorted(values)
    n = len(ordered)

    if n == 0:
        raise ValueError("cannot compute a median of an empty list")

    # The % operator gives the remainder after division.
    # n % 2 == 1 means n is odd.
    if n % 2 == 1:
        # Integer division // discards the remainder.
        # For n=9: 9 // 2 = 4, and ordered[4] is the 5th item (Python counts
        # from 0). That is the middle of nine items.
        middle_index = n // 2
        return ordered[middle_index]
    else:
        # For n=8: the two middle items are at index 3 and 4 (the 4th and 5th).
        lower_index = n // 2 - 1
        upper_index = n // 2
        return (ordered[lower_index] + ordered[upper_index]) / 2


def show_median_calculation(values, label):
    """Print the median calculation showing which position was selected."""
    ordered = sorted(values)
    n = len(ordered)
    median = median_from_scratch(values)

    print(f"\n{label}")
    print(f"  sorted: {[f'{v:g}' for v in ordered]}")
    print(f"  n = {n}, which is {'odd' if n % 2 else 'even'}")

    if n % 2 == 1:
        position = n // 2 + 1  # +1 to convert to human counting (1st, 2nd...)
        print(f"  middle position = (n + 1) / 2 = {position}")
        print(f"  median = the {position}th value = {median:g}")
    else:
        lower_pos = n // 2
        upper_pos = n // 2 + 1
        print(f"  two middle positions: {lower_pos}th and {upper_pos}th")
        print(f"  median = ({ordered[lower_pos - 1]:g} + {ordered[upper_pos - 1]:g}) / 2 = {median:g}")

    return median


# ==============================================================================
# SECTION 2 -- ROBUSTNESS AND THE BREAKDOWN POINT
# ==============================================================================

def show_breakdown_point(base_values):
    """
    Push one observation to absurd values and watch what each statistic does.

    The BREAKDOWN POINT is the proportion of the data you must corrupt before
    a statistic can be pushed anywhere you like.

        mean   -> breakdown point 0%.  One value does it, at any sample size.
        median -> breakdown point 50%. You must corrupt HALF the dataset.

    This is not a small technical difference. It is why statistical agencies
    and regulators report median household income rather than mean.
    """
    print(f"\n  {'largest value':>16} {'mean':>14} {'median':>10}")
    print("  " + "-" * 44)

    for replacement in [99, 990, 9_900, 99_000, 9_900_000]:
        modified = base_values[:]      # copy so we do not edit the original
        modified[-1] = replacement

        series = pd.Series(modified, dtype=float)
        print(f"  {replacement:>16,} {series.mean():>14,.1f} {series.median():>10.1f}")

    print("\n  The median is flat at 18 across all five rows.")
    print("  It is not being clever. It simply never looks at the magnitude")
    print("  of the largest value -- only at the fact that it is largest.")


# ==============================================================================
# SECTION 3 -- PERCENTILES, QUARTILES AND THE IQR
# ==============================================================================

def show_percentiles(series, label):
    """
    Show the five number summary and the interquartile range.

    A PERCENTILE generalises the median. The kth percentile is the value below
    which k% of the observations fall.

        Q1 = 25th percentile
        Q2 = 50th percentile = the median
        Q3 = 75th percentile

        IQR = Q3 - Q1 = the width of the middle half of the data

    The IQR is the resistant answer to "how spread out is this?", in the same
    way the median is the resistant answer to "where is the centre?". Both
    ignore the tails by construction.
    """
    q1 = series.quantile(0.25)
    q2 = series.quantile(0.50)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    print(f"\n{label}")
    print(f"  min    : {series.min():8.2f}")
    print(f"  Q1     : {q1:8.2f}   (25% of orders fall below this)")
    print(f"  median : {q2:8.2f}   (50% of orders fall below this)")
    print(f"  Q3     : {q3:8.2f}   (75% of orders fall below this)")
    print(f"  max    : {series.max():8.2f}")
    print(f"  IQR    : {iqr:8.2f}   (Q3 - Q1)")

    # The gap between Q3 and the maximum is visible evidence of a tail.
    # If the top quarter of the data is stretched far wider than the middle
    # half, you have a tail, and you know it without computing skewness.
    upper_gap = series.max() - q3
    print(f"\n  gap from Q3 to max: {upper_gap:.2f}, versus an IQR of {iqr:.2f}")
    if upper_gap > iqr:
        print("  The top quarter is stretched wider than the middle half.")
        print("  That is a right tail, visible from position data alone.")

    return q1, q3, iqr


# ==============================================================================
# SECTION 4 -- THE IQR OUTLIER FENCE
# ==============================================================================

def iqr_fences(series, multiplier=1.5):
    """
    Compute the standard outlier boundaries used by a box plot.

        lower fence = Q1 - 1.5 x IQR
        upper fence = Q3 + 1.5 x IQR

    Anything outside these is FLAGGED, not deleted. Flagging is a request to
    go and look at the observation. Sometimes it is a data entry error.
    Sometimes -- as here -- it is a genuine corporate customer who is worth
    more than everyone else combined.

    The 1.5 is a convention, not a derivation. Use 3.0 for a stricter fence.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return lower, upper


def show_outlier_flags(series, label):
    """Apply the IQR fence and report which observations fall outside it."""
    lower, upper = iqr_fences(series)

    print(f"\n{label}")
    print(f"  lower fence: {lower:8.2f}")
    print(f"  upper fence: {upper:8.2f}")

    # Boolean indexing: the expression inside the brackets produces a
    # True/False value for every row, and pandas keeps only the True rows.
    # The | symbol means OR, and each condition needs its own brackets.
    flagged = series[(series < lower) | (series > upper)]

    if len(flagged) == 0:
        print("  no observations flagged")
    else:
        print(f"  flagged: {flagged.tolist()}")
        print("  -> go and look at these. Do not delete them by reflex.")


# ==============================================================================
# SECTION 5 -- THE MODE
# ==============================================================================

def mode_from_scratch(values):
    """
    Find the most frequently occurring value or values.

    The mode is the only measure of centre that works on categorical data.
    You cannot average payment methods or failure codes, and you cannot take
    their median, but you can ask which one occurs most often.

    A dataset can have more than one mode, so this returns a list.
    """
    # Count how many times each distinct value appears.
    counts = {}
    for value in values:
        # dict.get(key, default) returns the default if the key is missing,
        # which saves writing an if-statement for the first occurrence.
        counts[value] = counts.get(value, 0) + 1

    # The highest count seen anywhere.
    highest_count = max(counts.values())

    # Every value that achieves that highest count is a mode.
    modes = [value for value, count in counts.items() if count == highest_count]

    return sorted(modes), highest_count, counts


def show_bimodality_warning():
    """
    Show what happens when two populations are stacked into one column.

    Meridian's Andheri store serves two distinct groups: walk-in retail
    customers and corporate bulk buyers. If we summarise them as one column,
    every statistic we produce describes an average customer who does not
    exist.

    Two peaks in a histogram is not a curiosity. It is an instruction to
    split the data before computing anything else.
    """
    # Two clearly separate groups of order values.
    walk_in = [12, 14, 15, 15, 16, 18, 18, 19, 20, 22]
    corporate = [88, 92, 95, 99, 101, 104, 108]

    mixed = walk_in + corporate
    mixed_series = pd.Series(mixed, dtype=float)

    print(f"\n  walk-in orders   (n={len(walk_in)}): {walk_in}")
    print(f"  corporate orders (n={len(corporate)}): {corporate}")

    print(f"\n  Summarised as ONE column:")
    print(f"    mean   = {mixed_series.mean():.2f}")
    print(f"    median = {mixed_series.median():.2f}")

    # Count how many actual orders sit anywhere near the reported centre.
    mean_value = mixed_series.mean()
    near_centre = mixed_series[(mixed_series > mean_value - 10) &
                               (mixed_series < mean_value + 10)]
    print(f"\n    orders within 10 of that mean: {len(near_centre)} out of {len(mixed)}")

    print(f"\n  Summarised as TWO segments:")
    print(f"    walk-in   median = {pd.Series(walk_in, dtype=float).median():.2f}")
    print(f"    corporate median = {pd.Series(corporate, dtype=float).median():.2f}")
    print("\n  The second version describes real customers. The first does not.")


# ==============================================================================
# SECTION 6 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("THE MEDIAN, PERCENTILES AND THE MODE -- MERIDIAN RETAIL")
    print("=" * 78)

    # --- Section 1: the calculation ------------------------------------------
    print("\n" + "-" * 78)
    print("1. COMPUTING THE MEDIAN")
    print("-" * 78)

    median_all = show_median_calculation(ORDER_VALUES, "ALL NINE ORDERS (odd n)")
    median_clean = show_median_calculation(ORDER_VALUES_NO_BULK, "BULK ORDER REMOVED (even n)")

    # Verify against pandas.
    assert median_from_scratch(ORDER_VALUES) == orders.median()
    print(f"\n  our function agrees with pandas .median(): {orders.median():.2f}")

    # The headline comparison of the whole course.
    print("\n" + "-" * 78)
    print("   MEAN VERSUS MEDIAN, SIDE BY SIDE")
    print("-" * 78)
    print(f"\n  {'':<20} {'with bulk':>12} {'without':>12} {'change':>10}")
    print("  " + "-" * 56)
    print(f"  {'mean':<20} {orders.mean():>12.2f} {orders_no_bulk.mean():>12.2f} "
          f"{(orders.mean() - orders_no_bulk.mean()) / orders_no_bulk.mean() * 100:>9.0f}%")
    print(f"  {'median':<20} {orders.median():>12.2f} {orders_no_bulk.median():>12.2f} "
          f"{(orders.median() - orders_no_bulk.median()) / orders_no_bulk.median() * 100:>9.0f}%")

    # --- Section 2: breakdown point ------------------------------------------
    print("\n" + "-" * 78)
    print("2. ROBUSTNESS -- THE BREAKDOWN POINT")
    print("-" * 78)

    show_breakdown_point(ORDER_VALUES)

    # --- Section 3: percentiles ----------------------------------------------
    print("\n" + "-" * 78)
    print("3. PERCENTILES, QUARTILES AND THE IQR")
    print("-" * 78)

    show_percentiles(orders, "ALL NINE ORDERS")

    print("\n  Where percentiles are the standard in industry:")
    print("    engineering  -> p50, p95, p99 response time. Mean latency hides")
    print("                    exactly the requests that lose you customers.")
    print("    compensation -> median salary by band and location")
    print("    real estate  -> median price per square foot")
    print("    operations   -> p90 delivery time is the promise, not the mean")

    # --- Section 4: outlier fence --------------------------------------------
    print("\n" + "-" * 78)
    print("4. THE IQR OUTLIER FENCE")
    print("-" * 78)

    show_outlier_flags(orders, "ALL NINE ORDERS")
    show_outlier_flags(orders_no_bulk, "BULK ORDER REMOVED")

    # --- Section 5: the mode -------------------------------------------------
    print("\n" + "-" * 78)
    print("5. THE MODE")
    print("-" * 78)

    modes, frequency, counts = mode_from_scratch(ORDER_VALUES)

    print(f"\n  value counts: {counts}")
    print(f"  mode(s)     : {[f'{m:g}' for m in modes]}  (appearing {frequency} times)")
    print(f"\n  ordering on this dataset: mode {modes[0]:g} < median {median_all:g} "
          f"< mean {orders.mean():.2f}")
    print("  That ordering is the signature of a right-skewed distribution.")
    print("  File 05 measures it properly.")

    print("\n" + "-" * 78)
    print("   BIMODALITY -- WHEN THE MODE IS WARNING YOU")
    print("-" * 78)

    show_bimodality_warning()

    # --- Closing -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CHOOSING A MEASURE OF CENTRE")
    print("=" * 78)
    print("""
  symmetric, no extreme values   -> mean       uses all the information
  long tail or outliers present  -> median     resistant to extremes
  categorical data               -> mode       the only valid option
  two visible peaks              -> none yet   segment the data first
  components of unequal size     -> weighted mean
  reporting to a public audience -> median, or report both

  When in doubt, report both and let the gap between them speak.

  Next: 04_standard_deviation.py -- centre is only the first of four questions.
    """)
