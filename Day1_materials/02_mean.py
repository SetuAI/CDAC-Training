"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL
================================================================================

Meridian Retail pulls one day of orders from its Andheri store. Nine orders
came through, in thousands of rupees:

        12, 14, 15, 15, 18, 20, 22, 25, 99

Eight are ordinary walk-in orders. The ninth is a corporate bulk order.
The CFO wants to know what a typical order is worth.

--------------------------------------------------------------------------------
THIS FILE: THE MEAN
--------------------------------------------------------------------------------

The mean is the total spread equally across every unit. It answers:
"if every order were identical, how much would each one be?"

        mean = sum of all values / number of values

We build it from scratch, prove two properties that everything later depends
on, then break it on purpose so the failure is visible rather than theoretical.

Sections:
        1. Computing the mean by hand
        2. Property 1 -- deviations always sum to zero
        3. Property 2 -- the mean is not resistant
        4. The weighted mean
        5. The trimmed mean
================================================================================
"""

import pandas as pd


# The nine order values, in thousands of rupees.
ORDER_VALUES = [12, 14, 15, 15, 18, 20, 22, 25, 99]
ORDER_VALUES_NO_BULK = [12, 14, 15, 15, 18, 20, 22, 25]

orders = pd.Series(ORDER_VALUES, dtype=float)
orders_no_bulk = pd.Series(ORDER_VALUES_NO_BULK, dtype=float)


# ==============================================================================
# SECTION 1 -- COMPUTING THE MEAN FROM SCRATCH
# ==============================================================================

def mean_from_scratch(values):
    """
    Compute the arithmetic mean without using any statistics library.

    There are exactly two operations here: add everything up, divide by how
    many there are. Writing it out makes it obvious that the mean has no way
    to protect itself -- every value enters the sum with equal force, and a
    value of 99 contributes 99 no matter how unlike the others it is.
    """
    # total starts at zero and accumulates each value in turn.
    # We use an explicit loop rather than sum() so the accumulation is visible.
    total = 0.0
    for value in values:
        total = total + value

    # count is simply how many numbers we added.
    count = len(values)

    # Guard against an empty list. Dividing by zero would crash; an explicit
    # error message tells the student what went wrong instead of showing a
    # ZeroDivisionError traceback.
    if count == 0:
        raise ValueError("cannot compute a mean of an empty list")

    return total / count


def show_mean_calculation(values, label):
    """Print the mean calculation as it would be written on a whiteboard."""
    total = sum(values)
    count = len(values)
    mean = total / count

    print(f"\n{label}")
    # " + ".join(...) builds the string "12 + 14 + 15 + ..." so the arithmetic
    # is shown rather than just asserted.
    terms = " + ".join(f"{v:g}" for v in values)
    print(f"  {terms}")
    print(f"  = {total:g}")
    print(f"  mean = {total:g} / {count} = {mean:.2f}")

    return mean


# ==============================================================================
# SECTION 2 -- PROPERTY 1: DEVIATIONS ALWAYS SUM TO ZERO
# ==============================================================================

def show_deviations_sum_to_zero(values, label):
    """
    Show that the deviations from the mean always add up to zero.

    A deviation is the distance of one observation from the mean:

            deviation = value - mean

    Some are positive, some negative. They ALWAYS cancel exactly. This is not
    a property of this dataset -- it is true of every dataset that has ever
    existed, and it follows directly from the definition of the mean.

    Why this matters: it kills the obvious approach to measuring spread.
    You cannot "average the distances from the centre", because that average
    is zero for every dataset. This is precisely why the variance formula in
    file 04 squares the deviations before adding them.
    """
    mean = sum(values) / len(values)

    # Build the list of deviations. This is a list comprehension: for each
    # value in values, compute value - mean, and collect the results.
    deviations = [value - mean for value in values]

    print(f"\n{label}")
    print(f"  mean = {mean:.4f}")
    print(f"  {'value':>8} {'deviation':>12}")
    print("  " + "-" * 21)
    for value, deviation in zip(values, deviations):
        # zip() walks two lists side by side, giving one pair at a time.
        print(f"  {value:>8.2f} {deviation:>+12.4f}")

    total_deviation = sum(deviations)
    print("  " + "-" * 21)
    print(f"  {'SUM':>8} {total_deviation:>+12.4f}")

    # The result will not be exactly 0.0 on a computer -- it will be something
    # like 1e-15. That is floating point rounding, not a flaw in the maths.
    # round() to 10 decimal places makes the intended answer visible.
    print(f"\n  rounded: {round(total_deviation, 10)}")
    print("  The positives and negatives cancel exactly. Always.")


# ==============================================================================
# SECTION 3 -- PROPERTY 2: THE MEAN IS NOT RESISTANT
# ==============================================================================

def show_sensitivity_to_one_value(base_values, replacement_values):
    """
    Replace one observation with progressively larger values and watch the
    mean follow it.

    "Not resistant" means: no matter how far one observation moves, it keeps
    dragging the mean with it. There is no point at which the mean says
    "that value is too strange, I will ignore it."

    The technical name for this is a breakdown point of 0%. You need to corrupt
    zero percent of the data -- that is, one single observation in a dataset of
    any size -- to move the mean anywhere you like.
    """
    print(f"\n  original data      : {base_values}")
    print(f"  original mean      : {sum(base_values) / len(base_values):.2f}")
    print(f"\n  now replace the last value and recompute:\n")
    print(f"  {'last value':>14} {'mean':>12} {'median':>12}")
    print("  " + "-" * 40)

    for replacement in replacement_values:
        # Copy the list before modifying it. Without [:] we would be editing
        # the caller's list, which is a common and painful Python bug.
        modified = base_values[:]
        modified[-1] = replacement

        mean = sum(modified) / len(modified)
        # The median is included purely for contrast. It is the subject of
        # file 03, but seeing it refuse to move here is the whole argument
        # for why that file exists.
        median = pd.Series(modified, dtype=float).median()

        print(f"  {replacement:>14,.0f} {mean:>12,.2f} {median:>12,.2f}")

    print("\n  The mean chases the outlier without limit. The median does not move.")


# ==============================================================================
# SECTION 4 -- THE WEIGHTED MEAN
# ==============================================================================

def weighted_mean(values, weights):
    """
    Compute a mean where observations carry different importance.

            weighted mean = sum(weight * value) / sum(weights)

    Use this whenever the things you are averaging are not the same size.
    Averaging three product-line margins as if each line were equally large
    is a straightforward error, and it is one of the most common mistakes in
    business reporting.
    """
    # Both lists must line up, otherwise zip() silently drops the extra items
    # and you get a wrong answer with no error message.
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")

    # Sum of (weight x value) across all pairs.
    weighted_total = sum(w * v for v, w in zip(values, weights))
    total_weight = sum(weights)

    return weighted_total / total_weight


def show_weighted_mean_case():
    """
    Meridian Retail's blended gross margin across three product lines.

    The unweighted average treats a 40 crore apparel line as equal in
    importance to a 240 crore grocery line. It is not.
    """
    # Each row is one product line.
    lines = ["Electronics", "Apparel", "Grocery"]
    revenue = [120.0, 40.0, 240.0]   # in crores of rupees
    margin = [8.0, 32.0, 4.0]        # margin percentage on that line

    print(f"\n  {'line':<14} {'revenue (Cr)':>14} {'margin %':>10}")
    print("  " + "-" * 40)
    for name, rev, mar in zip(lines, revenue, margin):
        print(f"  {name:<14} {rev:>14.0f} {mar:>10.1f}")

    # The naive approach: average the three margin percentages directly.
    simple = sum(margin) / len(margin)

    # The correct approach: weight each margin by the revenue it applies to.
    weighted = weighted_mean(values=margin, weights=revenue)

    print(f"\n  simple average of margins   : {simple:.2f}%")
    print(f"  revenue-weighted margin     : {weighted:.2f}%")
    print(f"  the simple average overstates profitability by {simple / weighted:.1f} times")
    print("\n  Portfolio returns, index values and blended rates are all")
    print("  weighted means. Using a simple average on any of them is an error.")


# ==============================================================================
# SECTION 5 -- THE TRIMMED MEAN
# ==============================================================================

def trimmed_mean(values, trim_count):
    """
    Drop the highest and lowest observations, then take an ordinary mean.

    trim_count is how many observations to remove from EACH end.
    trim_count=1 on nine values leaves seven.

    This is the controlled compromise: you keep the arithmetic convenience of
    the mean, but you refuse to let the extremes dominate. Olympic scoring,
    consumer price index construction and sell-side consensus estimates all
    work this way.

    The cost is real: you are discarding genuine observations. Always state
    how much you trimmed when you report one.
    """
    # Sort first -- "highest and lowest" only means something on sorted data.
    ordered = sorted(values)

    # Check we are not trimming away the entire dataset.
    if trim_count * 2 >= len(ordered):
        raise ValueError("trimming that much would leave nothing behind")

    # Slice off trim_count items from the front and trim_count from the back.
    # ordered[1:-1] on a nine-item list gives items at positions 1 through 7.
    kept = ordered[trim_count : len(ordered) - trim_count]

    return sum(kept) / len(kept), kept


# ==============================================================================
# SECTION 6 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("THE MEAN -- MERIDIAN RETAIL")
    print("=" * 78)

    # --- Section 1: the calculation ------------------------------------------
    print("\n" + "-" * 78)
    print("1. COMPUTING THE MEAN")
    print("-" * 78)

    mean_all = show_mean_calculation(ORDER_VALUES, "ALL NINE ORDERS")
    mean_clean = show_mean_calculation(ORDER_VALUES_NO_BULK, "BULK ORDER REMOVED")

    print(f"\n  removing one order out of nine moved the reported average by "
          f"{(mean_all - mean_clean) / mean_clean * 100:.0f}%")

    # Check our from-scratch function against pandas. If these ever disagree,
    # our code is wrong -- pandas is not.
    assert abs(mean_from_scratch(ORDER_VALUES) - orders.mean()) < 1e-10
    print(f"  our function agrees with pandas .mean(): {orders.mean():.4f}")

    # --- Section 2: deviations sum to zero -----------------------------------
    print("\n" + "-" * 78)
    print("2. PROPERTY: DEVIATIONS FROM THE MEAN ALWAYS SUM TO ZERO")
    print("-" * 78)

    show_deviations_sum_to_zero(ORDER_VALUES, "ALL NINE ORDERS")
    print("\n  Consequence: we cannot measure spread by averaging these")
    print("  distances, because they always average to zero. File 04 solves")
    print("  this by squaring them first.")

    # --- Section 3: not resistant --------------------------------------------
    print("\n" + "-" * 78)
    print("3. PROPERTY: THE MEAN IS NOT RESISTANT")
    print("-" * 78)

    show_sensitivity_to_one_value(
        base_values=ORDER_VALUES,
        replacement_values=[25, 99, 500, 5000, 100000],
    )

    # --- Section 4: weighted mean --------------------------------------------
    print("\n" + "-" * 78)
    print("4. THE WEIGHTED MEAN -- WHEN OBSERVATIONS ARE NOT EQUAL")
    print("-" * 78)

    show_weighted_mean_case()

    # --- Section 5: trimmed mean ---------------------------------------------
    print("\n" + "-" * 78)
    print("5. THE TRIMMED MEAN -- KEEPING THE MEAN, DROPPING THE TAILS")
    print("-" * 78)

    trimmed, kept_values = trimmed_mean(ORDER_VALUES, trim_count=1)

    print(f"\n  original (n=9)          : {sorted(ORDER_VALUES)}")
    print(f"  after trimming 1 each end: {kept_values}")
    print(f"\n  untrimmed mean : {mean_all:.2f}")
    print(f"  trimmed mean   : {trimmed:.2f}")
    print(f"  median         : {orders.median():.2f}   (for comparison)")
    print("\n  The trimmed mean lands close to the median, which is the point.")

    # --- Closing -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("WHEN TO REPORT THE MEAN")
    print("=" * 78)
    print("""
  Report the mean when:
    - the distribution is roughly symmetric
    - you need the total to be recoverable (mean x n = sum)
    - the number feeds further analysis -- variance, regression and most
      modelling techniques are built on the mean
    - every observation genuinely deserves equal weight

  Avoid it when the tail is long and the audience will read "average" as
  "typical". On this dataset those two words mean different things by 51%.

  Next: 03_median_and_mode.py -- centre measured by position instead.
    """)
