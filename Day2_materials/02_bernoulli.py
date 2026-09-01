"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL, DAY 2
================================================================================

Meridian Retail's site converts 28% of visitors into buyers. The operations
team has four questions about next week. This file answers the first one:

        "The next visitor lands on the site. Will they buy?"

One visitor. Two possible outcomes. That is a BERNOULLI question.

--------------------------------------------------------------------------------
THIS FILE: THE BERNOULLI DISTRIBUTION
--------------------------------------------------------------------------------

The simplest distribution there is. One trial, two outcomes, one input.

        P(success) = p
        P(failure) = 1 - p

We call one outcome "success" and the other "failure", but these are only
labels. A loan default can be the success -- it just means the outcome we
are counting.

Sections:
        1. The distribution, computed by hand
        2. Mean and variance, and why the mean looks strange
        3. Simulating visitors
        4. Where Bernoulli shows up in a real dataset
        5. Why one trial is rarely enough
================================================================================
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)

# Meridian's conversion rate, from the Day 1 data.
CONVERSION_RATE = 0.28


# ==============================================================================
# SECTION 1 -- THE DISTRIBUTION
# ==============================================================================

def bernoulli_from_scratch(p):
    """
    Return the complete Bernoulli distribution as a dictionary.

    There are only two outcomes, so the whole distribution is two lines:

        outcome 1 (success) has probability p
        outcome 0 (failure) has probability 1 - p

    That is the entire distribution. No formula to memorise.
    """
    # Guard against an impossible input. A probability must sit between 0 and 1.
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    return {1: p, 0: 1 - p}


def show_bernoulli(p, success_label, failure_label):
    """Print the distribution as a small table with a check that it sums to 1."""
    distribution = bernoulli_from_scratch(p)

    print(f"\n  {'outcome':<22} {'value':>7} {'probability':>14}")
    print("  " + "-" * 46)
    print(f"  {success_label:<22} {1:>7} {distribution[1]:>14.4f}")
    print(f"  {failure_label:<22} {0:>7} {distribution[0]:>14.4f}")
    print("  " + "-" * 46)
    print(f"  {'TOTAL':<22} {'':>7} {sum(distribution.values()):>14.4f}")

    # Check our two-line version against the library.
    assert abs(stats.bernoulli.pmf(1, p) - distribution[1]) < 1e-12
    print(f"\n  scipy agrees: stats.bernoulli.pmf(1, p={p}) = "
          f"{stats.bernoulli.pmf(1, p):.4f}")


# ==============================================================================
# SECTION 2 -- MEAN AND VARIANCE
# ==============================================================================

def bernoulli_mean_and_variance(p):
    """
    Compute the mean and variance, and explain what they mean here.

        Mean     = p
        Variance = p x (1 - p)

    The mean of a Bernoulli is the probability itself. That looks odd at
    first, because no single trial ever produces a value of 0.28 -- every
    trial gives you exactly 0 or exactly 1.

    The mean is what you get if you average the outcomes over many trials.
    It is a long-run rate, not a description of one trial.
    """
    mean = p
    variance = p * (1 - p)
    sd = variance ** 0.5

    print(f"\n  p        = {p}")
    print(f"  mean     = p          = {mean:.4f}")
    print(f"  variance = p x (1-p)  = {variance:.4f}")
    print(f"  std dev  = sqrt(var)  = {sd:.4f}")

    print("\n  No single visitor is ever '0.28 bought'. Every visitor is a 0")
    print("  or a 1. The mean is the rate you converge to across many visitors.")

    return mean, variance


def show_where_variance_is_highest():
    """
    Show how the variance changes with p.

    Variance is largest at p = 0.5 and shrinks towards 0 at either extreme.
    This has a practical meaning: an outcome you are unsure about is the most
    unpredictable, and an outcome that almost always happens is easy to plan
    around.
    """
    print(f"\n  {'p':>8} {'variance':>12} {'std dev':>10}   {'':<20}")
    print("  " + "-" * 54)

    for p in [0.01, 0.10, 0.28, 0.50, 0.72, 0.90, 0.99]:
        variance = p * (1 - p)
        # A simple bar so the shape is visible without a plotting library.
        bar = "#" * int(variance * 100)
        print(f"  {p:>8.2f} {variance:>12.4f} {variance ** 0.5:>10.4f}   {bar}")

    print("\n  Variance peaks at p = 0.5, where the outcome is least predictable,")
    print("  and falls away at both ends. A 99% on-time delivery process is easy")
    print("  to plan around. A 50% one is not.")


# ==============================================================================
# SECTION 3 -- SIMULATING VISITORS
# ==============================================================================

def simulate_visitors(p, n_visitors, show_first=20):
    """
    Simulate visitors arriving and either buying or not.

    Each visitor is one Bernoulli trial. We draw a random number between 0
    and 1; if it lands below p, that visitor bought.
    """
    # Each entry is 1 (bought) or 0 (did not buy).
    outcomes = (rng.random(n_visitors) < p).astype(int)

    print(f"\n  First {show_first} visitors (1 = bought, 0 = did not):")
    print(f"    {list(outcomes[:show_first])}")

    bought = outcomes.sum()
    print(f"\n  visitors simulated : {n_visitors:,}")
    print(f"  bought             : {bought:,}")
    print(f"  observed rate      : {bought / n_visitors:.4f}")
    print(f"  true rate (p)      : {p:.4f}")

    return outcomes


# ==============================================================================
# SECTION 4 -- BERNOULLI IN A REAL DATASET
# ==============================================================================

def show_real_bernoulli_columns():
    """
    Every Yes/No column in a dataset is Bernoulli data.

    These are the actual rates from the Day 1 Meridian dataset. Estimating p
    is the easiest task in statistics: count the successes, divide by the
    total.
    """
    columns = [
        ("returned == 'Yes'", 0.1464, "an order is returned"),
        ("customer_rating <= 2", 0.2294, "a customer rates 1 or 2"),
        ("delivery_days > 4", 0.1478, "a delivery misses a 4-day promise"),
    ]

    print(f"\n  {'column':<24} {'p':>8} {'1-p':>8}   what a 'success' means")
    print("  " + "-" * 74)
    for name, p, meaning in columns:
        print(f"  {name:<24} {p:>8.4f} {1 - p:>8.4f}   {meaning}")

    print("\n  Note the third row. 'Success' here is a missed delivery. The label")
    print("  carries no judgement -- it just marks the outcome being counted.")


# ==============================================================================
# SECTION 5 -- WHY ONE TRIAL IS NOT ENOUGH
# ==============================================================================

def show_the_limitation(p, n=20):
    """
    Bernoulli describes one trial. Business questions are about many.

    Knowing that one visitor converts with probability 0.28 does not tell you
    how many of tomorrow's 20 visitors will convert. Adding up n Bernoulli
    trials is exactly what the binomial distribution does, which is the next
    file.
    """
    print(f"\n  Bernoulli answers : 'will this ONE visitor buy?'   -> p = {p}")
    print(f"  It cannot answer  : 'how many of {n} visitors will buy?'")

    # Show it directly: run 20 visitors ten separate times.
    print(f"\n  Ten separate days, {n} visitors each:\n")
    print(f"  {'day':>5} {'outcomes':>44} {'total':>7}")
    print("  " + "-" * 58)

    totals = []
    for day in range(1, 11):
        outcomes = (rng.random(n) < p).astype(int)
        total = outcomes.sum()
        totals.append(total)
        compact = "".join(str(x) for x in outcomes)
        print(f"  {day:>5} {compact:>44} {total:>7}")

    print(f"\n  The daily totals ranged from {min(totals)} to {max(totals)}.")
    print(f"  Predicting that range is a BINOMIAL question. See 03_binomial.py.")


# ==============================================================================
# SECTION 6 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("THE BERNOULLI DISTRIBUTION -- ONE EVENT, TWO OUTCOMES")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("1. THE DISTRIBUTION")
    print("-" * 78)
    show_bernoulli(CONVERSION_RATE, "visitor buys", "visitor leaves")

    print("\n" + "-" * 78)
    print("2. MEAN AND VARIANCE")
    print("-" * 78)
    bernoulli_mean_and_variance(CONVERSION_RATE)
    show_where_variance_is_highest()

    print("\n" + "-" * 78)
    print("3. SIMULATING VISITORS")
    print("-" * 78)
    simulate_visitors(CONVERSION_RATE, n_visitors=10_000)

    print("\n" + "-" * 78)
    print("4. BERNOULLI COLUMNS IN THE REAL DATASET")
    print("-" * 78)
    show_real_bernoulli_columns()

    print("\n" + "-" * 78)
    print("5. WHY ONE TRIAL IS NOT ENOUGH")
    print("-" * 78)
    show_the_limitation(CONVERSION_RATE)

    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRY IT YOURSELF")
    print("=" * 78)
    print("""
  1. Change CONVERSION_RATE at the top from 0.28 to 0.50 and re-run.
     Look at the ten simulated days -- the totals now swing much wider,
     because variance is highest at p = 0.5.

  2. Set it to 0.95 and re-run. The days become almost identical. High p
     means a predictable process.

  3. Run one line yourself:

         simulate_visitors(p=0.28, n_visitors=100)

     Then run it again with 100,000. Watch the observed rate settle.

  4. ADVANTAGES of Bernoulli: one input, trivial to estimate, and it is the
     building block for the next three distributions.
     DISADVANTAGE: it describes one trial only, which is almost never the
     question a business is actually asking.
    """)


# ==============================================================================
# EXPECTED OUTPUT AND HOW TO READ IT
# ==============================================================================
"""
--------------------------------------------------------------------------------
WHAT YOU SHOULD SEE
--------------------------------------------------------------------------------

SECTION 1  A two-row table: buys 0.2800, leaves 0.7200, total 1.0000.

SECTION 2  mean 0.2800, variance 0.2016, std dev 0.4490.
           Then a variance table with bars, peaking at p = 0.50.

SECTION 3  A list of 20 zeros and ones, then roughly 2,800 purchases out of
           10,000 visitors.

SECTION 4  Three real Yes/No columns with their p values.

SECTION 5  Ten simulated days of 20 visitors, with daily totals varying by
           several conversions.

--------------------------------------------------------------------------------
HOW TO READ IT
--------------------------------------------------------------------------------

THE MEAN OF 0.28
    This is the line that confuses people first. No visitor is ever "0.28
    bought" -- every visitor is a 0 or a 1. The mean of a Bernoulli is a
    long-run rate: average the outcomes over many visitors and you land on
    0.28.

    Practical version: you cannot use it to predict one visitor. You can use
    it to predict a week.

THE VARIANCE TABLE PEAKING AT p = 0.5
    Variance measures unpredictability. At p = 0.99 almost every trial is a
    success, so there is very little to be uncertain about and the variance
    is 0.0099. At p = 0.50 the outcome is a genuine coin flip and the
    variance is at its maximum of 0.25.

    Operational reading: a 99% on-time process is easy to staff for. A 50%
    one is not, and no amount of averaging will make it so.

SECTION 3 LANDING NEAR 2,800
    The simulated rate will be close to 0.28 but not exactly 0.28. That gap
    is normal sampling variation, and it shrinks as the number of visitors
    grows. This is the same idea demonstrated in file 01.

THE TEN DAYS IN SECTION 5 -- THIS IS THE IMPORTANT ONE
    Each day has the same underlying 28% rate, yet the daily totals differ
    by several conversions. Nothing changed between those days. No campaign
    launched, nothing broke.

    This is the practical warning of the whole file: a slow day is usually
    not a signal. It is what a fixed conversion rate looks like on a small
    sample. Deciding whether a difference is real is what A/B testing does,
    and A/B testing is built on the binomial distribution -- the next file.
"""
