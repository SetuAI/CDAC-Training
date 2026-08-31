"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL
================================================================================

Meridian Retail pulls one day of orders from its Andheri store. Nine orders
came through, in thousands of rupees:

        12, 14, 15, 15, 18, 20, 22, 25, 99

Alongside the store data, Meridian's operations team is comparing two delivery
hubs. Both report a mean fulfilment time of 50 minutes:

        Hub A : 48, 49, 50, 51, 52     minutes
        Hub B : 20, 35, 50, 65, 80     minutes

Identical centres. Hub A can promise a 55-minute delivery window. Hub B cannot
promise anything. The mean cannot tell these two operations apart. That is the
argument for this file.

--------------------------------------------------------------------------------
THIS FILE: VARIANCE AND STANDARD DEVIATION
--------------------------------------------------------------------------------

Sections:
        1. Same centre, different business
        2. From deviations to variance -- why we square
        3. Standard deviation, computed step by step
        4. n versus n-1 (Bessel's correction)
        5. Coefficient of variation -- comparing across scales
        6. The empirical rule and z-scores, and where z-scores fail
================================================================================
"""

import math
import pandas as pd


ORDER_VALUES = [12, 14, 15, 15, 18, 20, 22, 25, 99]
orders = pd.Series(ORDER_VALUES, dtype=float)

# A deliberately small, clean subset. Every number in the hand calculation
# below comes out whole until the final square root, which is the only way a
# step-by-step walkthrough stays readable on a whiteboard.
TEACHING_SET = [12, 14, 15, 18, 21]

HUB_A = [48, 49, 50, 51, 52]
HUB_B = [20, 35, 50, 65, 80]


# ==============================================================================
# SECTION 1 -- SAME CENTRE, DIFFERENT BUSINESS
# ==============================================================================

def compare_two_hubs(hub_a, hub_b):
    """
    Two datasets with identical means and completely different behaviour.

    This is the single clearest argument for why a centre on its own is not a
    description of anything. It is one point. A distribution is not one point.
    """
    a = pd.Series(hub_a, dtype=float)
    b = pd.Series(hub_b, dtype=float)

    # Print the raw data first, one hub per line, so the lists stay readable.
    print(f"\n  Hub A times (min): {hub_a}")
    print(f"  Hub B times (min): {hub_b}")

    print(f"\n  {'':<20} {'Hub A':>12} {'Hub B':>12}")
    print("  " + "-" * 46)
    print(f"  {'mean':<20} {a.mean():>12.1f} {b.mean():>12.1f}")
    print(f"  {'median':<20} {a.median():>12.1f} {b.median():>12.1f}")
    print(f"  {'std deviation':<20} {a.std():>12.2f} {b.std():>12.2f}")
    print(f"  {'range':<20} {a.max() - a.min():>12.0f} {b.max() - b.min():>12.0f}")

    print("\n  Same mean. Same median. Hub B's standard deviation is "
          f"{b.std() / a.std():.0f} times Hub A's.")
    print("  Hub A runs a process. Hub B runs a lottery.")


# ==============================================================================
# SECTION 2 -- FROM DEVIATIONS TO VARIANCE
# ==============================================================================

def show_why_we_square(values):
    """
    Demonstrate why the obvious approach to measuring spread fails, and what
    we do instead.

    We want the average distance from the centre. The obvious formula is:

            average of (value - mean)

    That is always zero, for every dataset, because of the property proved in
    file 02. So we need something that stops positive and negative distances
    from cancelling. Two options:

        1. Absolute values  -> mean absolute deviation. Intuitive, but the
           absolute value function has a sharp corner at zero, which makes it
           awkward in any formula that needs calculus.

        2. Squares          -> variance. Harder to interpret directly, but
           smooth, additive across independent sources, and the foundation of
           regression, ANOVA and essentially all inferential statistics.

    Statistics chose squares. That choice is the only reason we then have to
    take a square root at the end to get back to usable units.
    """
    mean = sum(values) / len(values)
    deviations = [v - mean for v in values]
    absolute_deviations = [abs(d) for d in deviations]
    squared_deviations = [d ** 2 for d in deviations]

    print(f"\n  mean = {mean:.4f}\n")
    print(f"  {'value':>8} {'deviation':>12} {'|deviation|':>13} {'deviation^2':>14}")
    print("  " + "-" * 50)

    for v, d, a, s in zip(values, deviations, absolute_deviations, squared_deviations):
        print(f"  {v:>8.2f} {d:>+12.4f} {a:>13.4f} {s:>14.4f}")

    print("  " + "-" * 50)
    print(f"  {'SUM':>8} {sum(deviations):>+12.4f} "
          f"{sum(absolute_deviations):>13.4f} {sum(squared_deviations):>14.4f}")

    print("\n  Column 2 sums to zero -- useless. Columns 3 and 4 do not.")
    print("  Note how much harder column 4 punishes the far observation:")
    largest_share = max(squared_deviations) / sum(squared_deviations) * 100
    print(f"  one observation out of {len(values)} contributes "
          f"{largest_share:.0f}% of the total squared deviation.")


# ==============================================================================
# SECTION 3 -- STANDARD DEVIATION, STEP BY STEP
# ==============================================================================

def standard_deviation_from_scratch(values, ddof=1):
    """
    Compute the standard deviation without a statistics library.

    ddof stands for "delta degrees of freedom" -- how much to subtract from n
    in the denominator.

        ddof=1 -> divide by (n - 1). SAMPLE standard deviation. The default.
        ddof=0 -> divide by n.       POPULATION standard deviation.

    Five steps:
        1. mean
        2. deviations from the mean
        3. square each deviation
        4. add the squares and divide  -> this is the VARIANCE
        5. square root                 -> this is the STANDARD DEVIATION
    """
    n = len(values)

    if n - ddof <= 0:
        raise ValueError("not enough observations for this ddof")

    # Step 1
    mean = sum(values) / n

    # Steps 2 and 3 combined into one comprehension.
    squared_deviations = [(v - mean) ** 2 for v in values]

    # Step 4 -- the variance. Note the denominator is (n - ddof), not n.
    variance = sum(squared_deviations) / (n - ddof)

    # Step 5 -- the square root brings us back to the original units.
    return math.sqrt(variance)


def show_sd_calculation(values, label):
    """Print the five-step standard deviation calculation in full."""
    n = len(values)
    mean = sum(values) / n
    deviations = [v - mean for v in values]
    squares = [d ** 2 for d in deviations]
    sum_squares = sum(squares)

    sample_variance = sum_squares / (n - 1)
    population_variance = sum_squares / n

    print(f"\n{label}")
    print(f"  data: {values}\n")

    print(f"  STEP 1  mean = {sum(values):g} / {n} = {mean:g}\n")

    print(f"  STEP 2  deviations from the mean:")
    print(f"          {[f'{d:+g}' for d in deviations]}\n")

    print(f"  STEP 3  square each one:")
    print(f"          {[f'{s:g}' for s in squares]}\n")

    print(f"  STEP 4  add them up: sum of squares = {sum_squares:g}")
    print(f"          sample variance     = {sum_squares:g} / ({n} - 1) = {sample_variance:g}")
    print(f"          population variance = {sum_squares:g} / {n} = {population_variance:g}\n")

    print(f"  STEP 5  take the square root:")
    print(f"          s (sample)     = {math.sqrt(sample_variance):.4f}")
    print(f"          sigma (pop)    = {math.sqrt(population_variance):.4f}")

    print(f"\n  Variance is in SQUARED units. If these were rupees, the variance")
    print(f"  is in rupees-squared, which means nothing to anyone. The square")
    print(f"  root exists purely to put the answer back into rupees.")


# ==============================================================================
# SECTION 4 -- WHY n MINUS 1
# ==============================================================================

def show_bessel_correction(values):
    """
    Show the size of the correction and explain why it exists.

    The deviations are measured from the SAMPLE mean, and the sample mean was
    itself computed from those same data points. The sample mean therefore
    sits at the exact centre of the sample -- closer to it than the true
    population mean would be.

    So squared deviations from the sample mean are systematically SMALLER than
    squared deviations from the population mean. Dividing by n would therefore
    underestimate the population variance every single time. Dividing by
    (n - 1) corrects for that bias.

    The (n - 1) is called DEGREES OF FREEDOM. Once the mean is fixed, only
    (n - 1) of the deviations are free to take any value -- the last one is
    forced, because they all have to add up to zero.
    """
    n = len(values)
    mean = sum(values) / n
    sum_squares = sum((v - mean) ** 2 for v in values)

    sample_sd = math.sqrt(sum_squares / (n - 1))
    population_sd = math.sqrt(sum_squares / n)

    print(f"\n  n = {n}")
    print(f"  s     (divide by n-1 = {n - 1}) = {sample_sd:.4f}")
    print(f"  sigma (divide by n   = {n}) = {population_sd:.4f}")
    print(f"  the correction is worth {(sample_sd / population_sd - 1) * 100:.1f}%")

    print("\n  How the correction shrinks as the sample grows:\n")
    print(f"  {'n':>8} {'s / sigma':>12} {'inflation':>12}")
    print("  " + "-" * 34)

    for sample_size in [3, 5, 10, 30, 100, 1000, 10000]:
        # The ratio between the two formulas depends only on n, not on the
        # data itself: it is always sqrt(n / (n-1)).
        ratio = math.sqrt(sample_size / (sample_size - 1))
        print(f"  {sample_size:>8} {ratio:>12.4f} {(ratio - 1) * 100:>11.2f}%")

    print("\n  It matters for small samples and is irrelevant for large ones.")
    print("  Use n-1 by default. It is what pandas, numpy's .std(ddof=1),")
    print("  Excel's STDEV.S and R's sd() all do.")


# ==============================================================================
# SECTION 5 -- COEFFICIENT OF VARIATION
# ==============================================================================

def coefficient_of_variation(series):
    """
    Standard deviation expressed relative to the mean.

            CV = s / mean

    A standard deviation of 7 is large for a variable that averages 10 and
    trivial for one that averages 10,000. CV removes the units entirely so
    two variables on different scales become comparable.

    Do not use it when the mean is near zero, or when the variable can be
    negative -- the ratio becomes unstable or meaningless.
    """
    return series.std() / series.mean()


def show_cv_case():
    """
    Two funds. Absolute volatility picks the wrong one; CV picks the right one.
    """
    funds = {
        "Fund A": {"mean_return": 8.0, "sd": 4.0},
        "Fund B": {"mean_return": 18.0, "sd": 7.2},
    }

    print(f"\n  {'fund':<10} {'mean return':>14} {'std dev':>10} {'CV':>10}")
    print("  " + "-" * 46)

    for name, stats in funds.items():
        cv = stats["sd"] / stats["mean_return"]
        print(f"  {name:<10} {stats['mean_return']:>13.1f}% {stats['sd']:>9.1f}% {cv:>10.2f}")

    print("\n  Fund B has nearly double the absolute volatility.")
    print("  Fund B also carries LESS risk per unit of return.")
    print("  Judging on standard deviation alone would have picked Fund A.")


# ==============================================================================
# SECTION 6 -- EMPIRICAL RULE AND Z-SCORES
# ==============================================================================

def z_score(value, mean, sd):
    """
    How many standard deviations a value sits from the mean.

            z = (value - mean) / s

    This standardises any variable onto a common scale, so a z of 2.5 means
    the same thing whether the underlying units are rupees, minutes or degrees.
    """
    return (value - mean) / sd


def show_empirical_rule_and_z_failure(series):
    """
    State the empirical rule, then show where z-scores break down.

    For an approximately NORMAL distribution:
        about 68% of values fall within mean +/- 1 standard deviation
        about 95% within +/- 2
        about 99.7% within +/- 3

    The trap: on skewed data, the outlier inflates the very standard deviation
    that is being used to judge it. The outlier ends up scoring an unremarkable
    z-value and slips through an automated 3-sigma filter.

    On skewed data, use the IQR fence from file 03 instead.
    """
    mean = series.mean()
    sd = series.std()

    print(f"\n  mean = {mean:.2f}, s = {sd:.2f}")
    print(f"\n  {'value':>8} {'z-score':>10}")
    print("  " + "-" * 20)

    for value in sorted(series):
        z = z_score(value, mean, sd)
        # Mark anything a naive 3-sigma filter would catch.
        flag = "  <- caught by a 3-sigma rule" if abs(z) > 3 else ""
        print(f"  {value:>8.0f} {z:>10.2f}{flag}")

    largest = series.max()
    z_largest = z_score(largest, mean, sd)

    print(f"\n  The bulk order of {largest:.0f} is more than five times the median.")
    print(f"  Its z-score is only {z_largest:.2f}. A 3-sigma filter misses it entirely.")
    print("\n  Why: that observation contributed most of the standard deviation")
    print("  itself. It inflated its own detector. This is the standard failure")
    print("  mode of z-score outlier detection on skewed data.")

    # Show the IQR fence catching what the z-score missed.
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    upper_fence = q3 + 1.5 * (q3 - q1)
    print(f"\n  The IQR fence puts the boundary at {upper_fence:.2f}, so it flags "
          f"the value of {largest:.0f} without difficulty.")


# ==============================================================================
# SECTION 7 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("VARIANCE AND STANDARD DEVIATION -- MERIDIAN RETAIL")
    print("=" * 78)

    # --- Section 1 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("1. SAME CENTRE, DIFFERENT BUSINESS")
    print("-" * 78)

    compare_two_hubs(HUB_A, HUB_B)

    # --- Section 2 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. WHY WE SQUARE THE DEVIATIONS")
    print("-" * 78)

    show_why_we_square(TEACHING_SET)

    # --- Section 3 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("3. STANDARD DEVIATION, STEP BY STEP")
    print("-" * 78)

    show_sd_calculation(TEACHING_SET, "CLEAN TEACHING SET (n=5)")

    # Verify against pandas before moving on.
    check = pd.Series(TEACHING_SET, dtype=float)
    assert abs(standard_deviation_from_scratch(TEACHING_SET, ddof=1) - check.std(ddof=1)) < 1e-10
    assert abs(standard_deviation_from_scratch(TEACHING_SET, ddof=0) - check.std(ddof=0)) < 1e-10
    print(f"\n  our function agrees with pandas: "
          f"s={check.std(ddof=1):.4f}, sigma={check.std(ddof=0):.4f}")

    # Now the same calculation on the real order data.
    print(f"\n  On the full nine-order dataset:")
    print(f"    mean   = {orders.mean():.2f}")
    print(f"    median = {orders.median():.2f}")
    print(f"    s      = {orders.std():.2f}")
    print(f"\n  The standard deviation is LARGER than the median. That alone")
    print(f"  tells you the distribution is badly behaved, before any further test.")

    # --- Section 4 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. WHY n MINUS 1 (BESSEL'S CORRECTION)")
    print("-" * 78)

    show_bessel_correction(TEACHING_SET)

    # --- Section 5 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("5. COEFFICIENT OF VARIATION")
    print("-" * 78)

    show_cv_case()

    print(f"\n  Meridian orders, CV = {coefficient_of_variation(orders):.2f}")
    print(f"  A CV above 1 means the spread exceeds the centre. Anything you")
    print(f"  forecast from this mean will be wrong by more than the mean itself.")

    # --- Section 6 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("6. THE EMPIRICAL RULE AND WHERE Z-SCORES FAIL")
    print("-" * 78)

    print("""
  For an approximately normal distribution:
      68%   of observations lie within  mean +/- 1s
      95%   within  mean +/- 2s
      99.7% within  mean +/- 3s

  Those percentages assume normality. Our data is not normal, and the
  consequences are immediate:""")

    show_empirical_rule_and_z_failure(orders)

    # --- Closing -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("""
  range    -- uses 2 observations, ignores the rest. Sanity check only.
  IQR      -- resistant, pairs with the median, drives the box plot.
  variance -- squared units, so never report it directly. It is a building
              block for everything downstream.
  s        -- variance back in original units. Report this one.
  CV       -- s relative to the mean. Use it to compare across scales.

  Never report a centre without a spread. Hub A and Hub B had the same
  mean and were not the same business.

  Next: 05_skewness.py -- centre and spread still cannot tell you which
  side of the distribution is longer.
    """)
