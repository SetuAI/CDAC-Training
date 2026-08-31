"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL
================================================================================

Meridian Retail runs a chain of stores. The analytics team has been asked one
question by the CFO:

        "What is a typical order worth at our Andheri store?"

The team pulls one full day of orders from the store. Nine orders came through.
The values, in thousands of rupees, are:

        12, 14, 15, 15, 18, 20, 22, 25, 99

Eight of these are ordinary walk-in retail orders. The ninth is a bulk order
placed by a corporate client for an office event.

That single bulk order is the whole problem. Depending on which statistic the
team reports, the CFO will hear "a typical order is worth 27 thousand" or
"a typical order is worth 18 thousand". Both are computed correctly. Only one
of them is honest.

This file sets up the dataset and produces the full statistical profile. The
files that follow take one statistic each and rebuild it from scratch:

        02_mean.py                 -- centre by balance point
        03_median_and_mode.py      -- centre by position
        04_standard_deviation.py   -- spread
        05_skewness.py             -- symmetry
        06_kurtosis.py             -- tails

Run this file first. Everything after it refers back to these nine numbers.
================================================================================
"""

import numpy as np
import pandas as pd


# ==============================================================================
# SECTION 1 -- THE DATA
# ==============================================================================

# The nine order values from the Andheri store, in thousands of rupees.
# We keep this as a plain Python list first so it is obvious there is no magic
# happening inside a library object. It is just nine numbers.
ORDER_VALUES = [12, 14, 15, 15, 18, 20, 22, 25, 99]

# The same nine numbers with the bulk corporate order removed.
# We will keep comparing these two lists all through the course. Every statistic
# we compute will be computed on both, so students can see exactly how much
# damage one observation does.
ORDER_VALUES_NO_BULK = [12, 14, 15, 15, 18, 20, 22, 25]

# Wrap them in pandas Series. A Series is a labelled column of numbers -- the
# same thing you get when you write df['order_value'] on a real dataset.
# dtype=float matters: if we leave these as integers, some operations will
# quietly do integer arithmetic and give slightly wrong answers.
orders = pd.Series(ORDER_VALUES, dtype=float, name="order_value")
orders_no_bulk = pd.Series(ORDER_VALUES_NO_BULK, dtype=float, name="order_value")


# ==============================================================================
# SECTION 2 -- LOOKING AT THE DATA BEFORE COMPUTING ANYTHING
# ==============================================================================

def show_raw_data(series, label):
    """
    Print the data sorted, plus its size.

    The first thing to do with any numeric column is sort it and look at it.
    Sorting is not a formality -- it is how you spot a tail. If the largest
    value sits far away from the second largest, you have a tail, and you know
    it before you compute a single statistic.
    """
    # .sort_values() returns a new sorted Series. It does not modify the
    # original -- pandas operations return copies unless you say otherwise.
    sorted_values = series.sort_values()

    print(f"\n{label}")
    print(f"  n            : {len(series)}")
    # .tolist() converts the Series back to a plain list so it prints cleanly
    # on one line instead of as a formatted pandas table.
    print(f"  sorted values: {sorted_values.tolist()}")

    # The gap between the largest and second-largest value is a cheap tail check.
    # .iloc[-1] is the last element by position, .iloc[-2] is second to last.
    if len(series) >= 2:
        largest = sorted_values.iloc[-1]
        second_largest = sorted_values.iloc[-2]
        gap = largest - second_largest
        print(f"  top two      : {second_largest} then {largest}  (gap of {gap})")


# ==============================================================================
# SECTION 3 -- THE FIVE NUMBER SUMMARY
# ==============================================================================

def five_number_summary(series):
    """
    Return the five number summary: min, Q1, median, Q3, max.

    These five numbers describe the shape of a distribution using position
    only. No arithmetic on the values themselves -- just "which value sits at
    which point in the sorted list". That is why none of them can be dragged
    around by a single extreme value.

    This is exactly what a box plot draws.
    """
    return {
        "min": series.min(),
        # .quantile(0.25) finds the value below which 25% of the data falls.
        # This is Q1, the first quartile.
        "Q1": series.quantile(0.25),
        # .quantile(0.50) is the median by definition -- the 50th percentile.
        "median": series.quantile(0.50),
        "Q3": series.quantile(0.75),
        "max": series.max(),
    }


def print_five_number_summary(series, label):
    """Print the five number summary along with the derived IQR and range."""
    summary = five_number_summary(series)

    print(f"\n{label}")
    for name, value in summary.items():
        # :>7 right-aligns the label in a 7-character column so the output
        # lines up into a readable table.
        print(f"  {name:>7} : {value:8.2f}")

    # The interquartile range is the width of the middle 50% of the data.
    # It is the resistant answer to "how spread out is this?"
    iqr = summary["Q3"] - summary["Q1"]
    # The range uses only the two most extreme values, so it is the opposite:
    # maximally sensitive to exactly the observations we are least sure about.
    data_range = summary["max"] - summary["min"]

    print(f"  {'IQR':>7} : {iqr:8.2f}   (width of the middle 50%)")
    print(f"  {'range':>7} : {data_range:8.2f}   (uses only 2 of {len(series)} values)")


# ==============================================================================
# SECTION 4 -- THE FULL PROFILE: ALL FOUR QUESTIONS AT ONCE
# ==============================================================================

def full_profile(series):
    """
    Return every statistic covered in this course, as a dictionary.

    The four questions any numeric column has to answer:
        1. Centre    -- where do the values sit?      mean, median
        2. Spread    -- how far apart are they?       standard deviation
        3. Symmetry  -- is one side longer?           skewness
        4. Tails     -- how extreme are extremes?     kurtosis

    Each of these has its own file later. Here we just call the library
    versions so we have a reference answer to check our own code against.
    """
    return {
        "n": len(series),
        "mean": series.mean(),
        "median": series.median(),
        # ddof=1 means "divide by n-1", the sample standard deviation.
        # This is pandas' default, but we write it out so it is never a mystery.
        "std_sample": series.std(ddof=1),
        # ddof=0 means "divide by n", the population standard deviation.
        "std_population": series.std(ddof=0),
        # .skew() returns the sample-adjusted skewness.
        # Positive = long right tail. Negative = long left tail.
        "skewness": series.skew(),
        # .kurt() returns EXCESS kurtosis: a normal distribution scores 0, not 3.
        # This trips people up constantly. Always check which one your tool
        # reports before you interpret the number.
        "excess_kurtosis": series.kurt(),
    }


def print_profile(series, label):
    """Print the full profile in a readable block."""
    profile = full_profile(series)

    print(f"\n{label}")
    print(f"  {'n':<16}: {profile['n']:>8}")
    print(f"  {'mean':<16}: {profile['mean']:>8.2f}")
    print(f"  {'median':<16}: {profile['median']:>8.2f}")
    print(f"  {'std (sample)':<16}: {profile['std_sample']:>8.2f}")
    print(f"  {'std (population)':<16}: {profile['std_population']:>8.2f}")
    print(f"  {'skewness':<16}: {profile['skewness']:>8.2f}")
    print(f"  {'excess kurtosis':<16}: {profile['excess_kurtosis']:>8.2f}")


# ==============================================================================
# SECTION 5 -- THE DIAGNOSTIC THAT COSTS TWO LINES
# ==============================================================================

def mean_median_diagnostic(series):
    """
    Compare the mean against the median and say what the comparison implies.

    This is the single cheapest useful thing you can do to a new numeric
    column. Two function calls and one subtraction, and you already know
    the shape of the distribution before computing skewness.

        mean approximately equal to median  -> roughly symmetric
        mean noticeably above the median    -> long right tail
        mean noticeably below the median    -> long left tail
    """
    mean = series.mean()
    median = series.median()

    # Express the gap as a percentage of the median so it is comparable across
    # datasets with completely different scales. A gap of 8 rupees means
    # nothing on its own; a gap of 48% of the median means a lot.
    gap_pct = (mean - median) / median * 100

    print(f"\n  mean   = {mean:.2f}")
    print(f"  median = {median:.2f}")
    print(f"  the mean sits {gap_pct:+.1f}% away from the median")

    # A 10% threshold is a rule of thumb, not a law. It is deliberately loose:
    # this check is meant to raise a flag, not to settle the question.
    if abs(gap_pct) < 10:
        print("  -> roughly symmetric. The mean is a safe number to report.")
    elif gap_pct > 0:
        print("  -> long RIGHT tail. A minority of large values is pulling the mean up.")
        print("     Report the median as your headline number.")
    else:
        print("  -> long LEFT tail. A minority of small values is pulling the mean down.")
        print("     Report the median as your headline number.")


# ==============================================================================
# SECTION 6 -- RUN EVERYTHING
# ==============================================================================

# This guard means the code below runs only when you execute this file directly.
# If another file imports this one, the definitions above become available but
# nothing prints. That is the standard Python pattern for a runnable script.
if __name__ == "__main__":

    print("=" * 78)
    print("MERIDIAN RETAIL -- ONE DAY OF ORDERS AT THE ANDHERI STORE")
    print("all values in thousands of rupees")
    print("=" * 78)

    # --- Step 1: look at the raw numbers -------------------------------------
    show_raw_data(orders, "ALL NINE ORDERS")
    show_raw_data(orders_no_bulk, "SAME DAY, BULK CORPORATE ORDER REMOVED")

    # --- Step 2: the five number summary -------------------------------------
    print("\n" + "=" * 78)
    print("FIVE NUMBER SUMMARY")
    print("=" * 78)
    print_five_number_summary(orders, "ALL NINE ORDERS")
    print_five_number_summary(orders_no_bulk, "BULK ORDER REMOVED")

    # --- Step 3: the full profile, side by side ------------------------------
    print("\n" + "=" * 78)
    print("FULL STATISTICAL PROFILE")
    print("=" * 78)
    print_profile(orders, "ALL NINE ORDERS")
    print_profile(orders_no_bulk, "BULK ORDER REMOVED")

    # --- Step 4: what actually changed ---------------------------------------
    print("\n" + "=" * 78)
    print("WHAT ONE OBSERVATION OUT OF NINE DID")
    print("=" * 78)

    with_bulk = full_profile(orders)
    without_bulk = full_profile(orders_no_bulk)

    print(f"\n  {'statistic':<18} {'with bulk':>12} {'without':>12} {'change':>12}")
    print("  " + "-" * 56)

    # Loop through the statistics we care about and show the before/after.
    # Putting this in a loop rather than writing six print statements keeps the
    # code short and makes it obvious that the same comparison is being applied
    # to every statistic.
    for key, name in [
        ("mean", "mean"),
        ("median", "median"),
        ("std_sample", "std deviation"),
        ("skewness", "skewness"),
        ("excess_kurtosis", "excess kurtosis"),
    ]:
        before = with_bulk[key]
        after = without_bulk[key]
        # Percentage change is only meaningful when the baseline is not close
        # to zero. Skewness and kurtosis can legitimately sit near zero, so for
        # those we show the absolute shift instead of a percentage.
        if key in ("mean", "median", "std_sample"):
            change = f"{(before - after) / after * 100:+.0f}%"
        else:
            change = f"{before - after:+.2f}"
        print(f"  {name:<18} {before:>12.2f} {after:>12.2f} {change:>12}")

    print("\n  The median moved least. That is not luck -- it is the defining")
    print("  property of the median, and the reason we teach it separately.")

    # --- Step 5: the two-line diagnostic -------------------------------------
    print("\n" + "=" * 78)
    print("THE DIAGNOSTIC: MEAN VERSUS MEDIAN")
    print("=" * 78)

    print("\nALL NINE ORDERS")
    mean_median_diagnostic(orders)

    print("\nBULK ORDER REMOVED")
    mean_median_diagnostic(orders_no_bulk)

    # --- Step 6: answer the CFO ----------------------------------------------
    print("\n" + "=" * 78)
    print("ANSWERING THE CFO")
    print("=" * 78)
    print("""
  Question : "What is a typical order worth at our Andheri store?"

  Wrong answer : "About 27 thousand."
                 Correct arithmetic. Describes none of the nine orders.
                 Eight of the nine sit below it.

  Right answer : "The typical order is about 18 thousand. One corporate bulk
                 order of 99 thousand pulls the average up to 27 thousand, so
                 the average is not representative of walk-in demand.
                 If you are planning inventory, use 18. If you are planning
                 revenue, use the total."

  The second answer contains the same data and a different decision.
    """)
