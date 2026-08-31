"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL
================================================================================

Meridian Retail pulls one day of orders from its Andheri store. Nine orders
came through, in thousands of rupees:

        12, 14, 15, 15, 18, 20, 22, 25, 99

We now know the centre (mean 26.67, median 18) and the spread (s = 27.44).
Neither of those tells us WHICH SIDE of the distribution is stretched. That
matters commercially: a cost distribution with a long right tail means the
downside is bounded and the upside risk is not, and you plan for that
differently.

--------------------------------------------------------------------------------
THIS FILE: SKEWNESS
--------------------------------------------------------------------------------

Skewness measures asymmetry -- the direction and degree to which one tail is
longer than the other.

        positive  -> long RIGHT tail. A minority of large values.
        zero      -> symmetric.
        negative  -> long LEFT tail. A minority of small values.

Sections:
        1. A text histogram, so shape is visible before it is measured
        2. Skewness from scratch -- and why the formula cubes
        3. Right skew, left skew, symmetric -- three datasets compared
        4. Reading the value against thresholds
        5. What skew changes about the decision -- the log transform
================================================================================
"""

import math
import numpy as np
import pandas as pd


ORDER_VALUES = [12, 14, 15, 15, 18, 20, 22, 25, 99]
ORDER_VALUES_NO_BULK = [12, 14, 15, 15, 18, 20, 22, 25]

orders = pd.Series(ORDER_VALUES, dtype=float)


# ==============================================================================
# SECTION 1 -- SEE THE SHAPE BEFORE MEASURING IT
# ==============================================================================

def text_histogram(values, bins=10, width=50):
    """
    Draw a histogram using text characters.

    No plotting library needed, so this runs in any terminal and in any
    notebook. The point is not the picture -- it is the habit. Always look at
    the distribution before you compute a statistic that summarises it.
    """
    series = pd.Series(values, dtype=float)

    # np.histogram splits the range into equal-width bins and counts how many
    # observations fall into each. It returns the counts and the bin edges.
    counts, edges = np.histogram(series, bins=bins)

    # Scale the longest bar to fit the requested width.
    largest_count = counts.max() if counts.max() > 0 else 1

    print()
    for i, count in enumerate(counts):
        low = edges[i]
        high = edges[i + 1]
        # Bar length proportional to the count in this bin.
        bar_length = int(count / largest_count * width)
        bar = "#" * bar_length
        print(f"  {low:8.1f} to {high:8.1f} | {bar} {count if count else ''}")
    print()


# ==============================================================================
# SECTION 2 -- SKEWNESS FROM SCRATCH
# ==============================================================================

def skewness_from_scratch(values):
    """
    Compute sample skewness, matching what pandas .skew() returns.

            g1 = [ n / ((n-1)(n-2)) ] * sum( ((x - mean) / s) ^ 3 )

    where s is the SAMPLE standard deviation (divide by n-1).

    Why the CUBE:
        - It PRESERVES SIGN. Negative deviations stay negative when cubed.
          Squaring would destroy exactly the information we are after.
        - It AMPLIFIES DISTANCE. A value 3 standard deviations out contributes
          27 units; a value 1 standard deviation out contributes 1.
        - So the sign of the total is decided by whichever tail reaches further.

    Dividing by s^3 makes the result dimensionless. Skewness has no units,
    which is why a skewness of 2.1 means the same thing on rupees, minutes
    and kilograms.

    The leading fraction n/((n-1)(n-2)) is a small-sample bias correction, the
    same idea as the n-1 in the variance. It needs at least three observations.
    """
    n = len(values)

    if n < 3:
        raise ValueError("skewness needs at least 3 observations")

    mean = sum(values) / n

    # Sample standard deviation -- note the n-1.
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(variance)

    if sd == 0:
        # Every value identical. There is no shape to describe.
        return 0.0

    # Standardise each observation, then cube it, then add them all up.
    cubed_z_sum = sum(((v - mean) / sd) ** 3 for v in values)

    # The bias correction factor.
    correction = n / ((n - 1) * (n - 2))

    return correction * cubed_z_sum


def show_skewness_contributions(values, label):
    """
    Show how much each observation contributes to the skewness total.

    This is the slide that makes skewness click. One observation out of nine
    supplies almost the entire signal, because the cube of its standardised
    distance dwarfs everything else.
    """
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))

    print(f"\n{label}")
    print(f"  mean = {mean:.2f}, s = {sd:.2f}\n")
    print(f"  {'value':>8} {'z = (x-mean)/s':>16} {'z cubed':>12} {'share':>10}")
    print("  " + "-" * 50)

    cubes = [((v - mean) / sd) ** 3 for v in values]
    # Share of the total is computed on absolute values, because positive and
    # negative contributions partly cancel and we want to show influence,
    # not net effect.
    total_absolute = sum(abs(c) for c in cubes)

    for v, c in zip(values, cubes):
        share = abs(c) / total_absolute * 100
        print(f"  {v:>8.0f} {(v - mean) / sd:>+16.3f} {c:>+12.2f} {share:>9.1f}%")

    print("  " + "-" * 50)
    print(f"  {'SUM':>8} {'':>16} {sum(cubes):>+12.2f}")

    correction = n / ((n - 1) * (n - 2))
    print(f"\n  skewness = [{n} / ({n-1} x {n-2})] x {sum(cubes):.2f} = {correction * sum(cubes):.4f}")


# ==============================================================================
# SECTION 3 -- THREE SHAPES SIDE BY SIDE
# ==============================================================================

def build_example_distributions(seed=42):
    """
    Generate three datasets with clearly different shapes.

    A fixed random seed means everyone in the class gets identical numbers,
    so the discussion is about the statistics and not about whose run differed.
    """
    rng = np.random.default_rng(seed)

    # RIGHT SKEW: a lognormal distribution. This is the natural shape of
    # anything bounded below at zero and unbounded above -- order values,
    # income, claim sizes, latency, time to resolution.
    right = pd.Series(rng.lognormal(mean=3.0, sigma=0.7, size=2000))

    # LEFT SKEW: an easy examination. Most students cluster near the ceiling,
    # a few score badly. Built by subtracting a right-skewed variable from
    # a maximum score.
    left = pd.Series(100 - rng.lognormal(mean=2.2, sigma=0.6, size=2000))

    # SYMMETRIC: a normal distribution, for reference.
    symmetric = pd.Series(rng.normal(loc=50, scale=10, size=2000))

    return right, left, symmetric


def describe_shape(series, label):
    """
    Print the mean/median/mode ordering alongside the skewness value.

    The ORDERING of the three centre measures is the fingerprint of the shape:

        right skew :  mode < median < mean
        symmetric  :  mode = median = mean
        left skew  :  mean < median < mode
    """
    mean = series.mean()
    median = series.median()
    skew = series.skew()

    print(f"\n  {label}")
    print(f"    mean     : {mean:8.2f}")
    print(f"    median   : {median:8.2f}")
    print(f"    skewness : {skew:8.2f}")

    if skew > 0.5:
        print(f"    -> mean sits ABOVE the median. Long RIGHT tail.")
    elif skew < -0.5:
        print(f"    -> mean sits BELOW the median. Long LEFT tail.")
    else:
        print(f"    -> mean and median close together. Roughly symmetric.")


# ==============================================================================
# SECTION 4 -- READING THE VALUE
# ==============================================================================

def interpret_skewness(value):
    """
    Translate a skewness number into a reporting decision.

    These thresholds are conventions, not laws. They exist to stop people
    staring at a number like 0.83 with no idea what to do next.
    """
    magnitude = abs(value)

    if magnitude < 0.5:
        return ("approximately symmetric",
                "the mean is safe to report on its own")
    elif magnitude < 1.0:
        return ("moderately skewed",
                "report the median alongside the mean")
    else:
        direction = "right" if value > 0 else "left"
        return (f"highly {direction}-skewed",
                "lead with the median; the mean on its own will mislead")


# ==============================================================================
# SECTION 5 -- WHAT SKEW CHANGES: THE LOG TRANSFORM
# ==============================================================================

def show_log_transform(series, label):
    """
    Apply a log transform to right-skewed data and measure what happens.

    Many modelling techniques assume roughly symmetric errors. Heavy right
    skew in a target variable breaks that assumption. Taking the natural log
    compresses the large values much more than the small ones, which pulls
    the tail in.

    Constraint: log is undefined at zero and negative numbers. If the data
    contains zeros, the usual fix is log(x + 1), available as np.log1p.
    """
    print(f"\n  {label}")
    print(f"    skewness before log : {series.skew():8.2f}")

    # np.log gives the natural logarithm. Applied to a Series, it operates
    # element by element and returns a new Series.
    logged = np.log(series)

    print(f"    skewness after log  : {logged.skew():8.2f}")
    print(f"    mean/median gap before: {(series.mean() / series.median() - 1) * 100:6.1f}%")
    print(f"    mean/median gap after : {(logged.mean() / logged.median() - 1) * 100:6.1f}%")
    print("\n    The transform did not remove any data. It changed the scale on")
    print("    which distance is measured, so the tail no longer dominates.")


# ==============================================================================
# SECTION 6 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("SKEWNESS -- MERIDIAN RETAIL")
    print("=" * 78)

    # --- Section 1 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("1. LOOK AT THE SHAPE FIRST")
    print("-" * 78)

    print("\n  ALL NINE ORDERS")
    text_histogram(ORDER_VALUES, bins=9)
    print("  Eight orders bunched at the left, one stranded on the right.")
    print("  That is a right tail, and you can see it before measuring it.")

    # --- Section 2 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. SKEWNESS FROM SCRATCH")
    print("-" * 78)

    show_skewness_contributions(ORDER_VALUES, "ALL NINE ORDERS")

    # Verify our implementation against pandas.
    ours = skewness_from_scratch(ORDER_VALUES)
    theirs = orders.skew()
    assert abs(ours - theirs) < 1e-10
    print(f"\n  our function agrees with pandas .skew(): {theirs:.4f}")

    print("\n  One observation out of nine supplies most of the signal.")
    print("  That is the cube doing its job -- it is designed to be dominated")
    print("  by whichever tail reaches further.")

    # --- Section 3 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("3. HOW MUCH ONE OBSERVATION MOVES IT")
    print("-" * 78)

    skew_with = skewness_from_scratch(ORDER_VALUES)
    skew_without = skewness_from_scratch(ORDER_VALUES_NO_BULK)

    print(f"\n  all nine orders      : skewness = {skew_with:+.2f}")
    print(f"  bulk order removed   : skewness = {skew_without:+.2f}")
    print(f"\n  {interpret_skewness(skew_with)[0]}  ->  {interpret_skewness(skew_with)[1]}")
    print(f"  {interpret_skewness(skew_without)[0]}  ->  {interpret_skewness(skew_without)[1]}")

    # --- Section 4 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. THREE SHAPES, SIDE BY SIDE")
    print("-" * 78)

    right, left, symmetric = build_example_distributions()

    describe_shape(right, "RIGHT SKEW -- order values, income, claim size, latency")
    text_histogram(right, bins=12)

    describe_shape(symmetric, "SYMMETRIC -- measurement error, standardised test scores")
    text_histogram(symmetric, bins=12)

    describe_shape(left, "LEFT SKEW -- scores on an easy exam, age at death, uptime %")
    text_histogram(left, bins=12)

    print("  Note the direction of the tail in each picture, and check it")
    print("  against the sign of the skewness value above it.")

    # --- Section 5 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("5. READING THE VALUE")
    print("-" * 78)

    print(f"\n  {'skewness':>12}  {'interpretation':<28} {'what to do'}")
    print("  " + "-" * 74)
    for test_value in [0.0, 0.3, -0.7, 1.4, 2.87, -2.1]:
        description, action = interpret_skewness(test_value)
        print(f"  {test_value:>+12.2f}  {description:<28} {action}")

    # --- Section 6 -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("6. WHAT SKEW CHANGES ABOUT THE DECISION")
    print("-" * 78)

    show_log_transform(right, "RIGHT-SKEWED ORDER VALUES (n=2000)")

    print("""
  Four things skewness changes:

  REPORTING   beyond +/- 1, lead with the median. Report the mean only with
              the median next to it.

  MODELLING   many methods assume roughly symmetric errors. Heavy right skew
              in a target variable usually calls for a log transform.

  OUTLIERS    under strong skew, large values may be genuine members of the
              population rather than errors. Do not delete them by reflex --
              Meridian's bulk buyer is a real customer.

  PLANNING    a right-skewed cost distribution means the downside is bounded
              and the upside is not. You plan against the tail, not the mean.

  Next: 06_kurtosis.py -- skewness tells you WHICH tail is longer. It does
  not tell you how heavy the tails are.
    """)
