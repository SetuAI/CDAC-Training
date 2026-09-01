"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL, DAY 2
================================================================================

Yesterday we described Meridian Retail's order book. Mean, median, standard
deviation, skewness and kurtosis all answered the same kind of question:

        "What does the data we already have look like?"

Today the questions change. The operations team is planning next week and needs
answers about things that have NOT happened yet:

        1. The next visitor lands on the site. Will they buy?
        2. Of the next 20 visitors, how many will buy?
        3. How many complaint calls will arrive at the desk in the next hour?
        4. What fraction of parcels will be too heavy for the standard rate?

None of these can be answered by looking at a column of past data. They need a
PROBABILITY DISTRIBUTION -- a rule that assigns a probability to every possible
outcome.

What we know from the Day 1 data:

        conversion rate      28% of visitors place an order
        complaint calls      6 per hour on average
        parcel weight        mean 1.85 kg, standard deviation 0.42 kg

--------------------------------------------------------------------------------
THIS FILE: THE FOUNDATION
--------------------------------------------------------------------------------

        1. What a probability distribution actually is
        2. Discrete or continuous -- the first question to ask
        3. What "probability 0.28" means, demonstrated
        4. A decision guide that picks the distribution for you

The files that follow take one distribution each:

        02_bernoulli.py    one yes/no event
        03_binomial.py     successes across n trials
        04_poisson.py      events in a window of time
        05_normal.py       measurements
        06_case_study.py   all four questions answered
================================================================================
"""

import numpy as np

# Fixed seed so every student sees identical numbers when we simulate.
rng = np.random.default_rng(2024)

# The three facts from the Day 1 dataset that drive everything today.
CONVERSION_RATE = 0.28
CALLS_PER_HOUR = 6
WEIGHT_MEAN = 1.85
WEIGHT_SD = 0.42


# ==============================================================================
# SECTION 1 -- WHAT A PROBABILITY DISTRIBUTION IS
# ==============================================================================

def build_a_distribution_by_hand():
    """
    Build a probability distribution from scratch, out of counts.

    A probability distribution is a rule with two parts:
        1. a list of every outcome that can happen
        2. a probability attached to each one

    The probabilities must be between 0 and 1, and they must add up to
    exactly 1. That is the entire definition. Nothing more.

    Here we build one from Meridian's customer rating counts.
    """
    # These are the actual rating counts from the Day 1 dataset.
    counts = {1: 703, 2: 444, 3: 508, 4: 1620, 5: 1725}

    total = sum(counts.values())

    print(f"\n  Ratings given by {total:,} customers:\n")
    print(f"  {'rating':>8} {'count':>8} {'probability':>14}")
    print("  " + "-" * 32)

    # Convert each count into a probability by dividing by the total.
    probabilities = {}
    for rating, count in counts.items():
        probability = count / total
        probabilities[rating] = probability
        print(f"  {rating:>8} {count:>8,} {probability:>14.4f}")

    # The check that every distribution must pass.
    total_probability = sum(probabilities.values())
    print("  " + "-" * 32)
    print(f"  {'TOTAL':>8} {total:>8,} {total_probability:>14.4f}")

    print("\n  The probabilities add to 1. That is what makes this a")
    print("  probability distribution rather than just a table of counts.")
    print("\n  If your probabilities ever add to something other than 1,")
    print("  you have made a mistake. It is the fastest sanity check there is.")

    return probabilities


# ==============================================================================
# SECTION 2 -- DISCRETE OR CONTINUOUS
# ==============================================================================

def explain_discrete_vs_continuous():
    """
    The first question to ask. It cuts your choice from four to two.

    DISCRETE   -- you are COUNTING. Whole numbers only, nothing in between.
                  Bernoulli, Binomial and Poisson are all discrete.

    CONTINUOUS -- you are MEASURING. Any value in a range is possible.
                  Normal is continuous.

    The test: can the answer be 2.5? If not, it is discrete.
    """
    examples = [
        ("number of customers who bought", "COUNT", "Discrete",
         "there is no such thing as 2.7 customers"),
        ("number of complaint calls", "COUNT", "Discrete",
         "calls arrive whole, never half a call"),
        ("number of parcels returned", "COUNT", "Discrete",
         "you cannot return 1.4 parcels"),
        ("parcel weight in kg", "MEASURE", "Continuous",
         "1.85, 1.8513, any value is possible"),
        ("delivery time in days", "MEASURE", "Continuous",
         "2.3 days is a real answer"),
        ("order value in rupees", "MEASURE", "Continuous",
         "treated as continuous in practice"),
    ]

    print(f"\n  {'variable':<34} {'you are':<9} {'type':<12} why")
    print("  " + "-" * 92)
    for variable, action, kind, reason in examples:
        print(f"  {variable:<34} {action:<9} {kind:<12} {reason}")

    print("\n  Counts are NEVER normal. This single rule prevents the most")
    print("  common mistake people make with distributions.")


# ==============================================================================
# SECTION 3 -- WHAT "PROBABILITY 0.28" ACTUALLY MEANS
# ==============================================================================

def demonstrate_what_probability_means(p=CONVERSION_RATE):
    """
    Show that a probability is a LONG-RUN rate, not a promise about any
    small number of trials.

    We simulate visitors arriving at the site. Each one buys with
    probability p. Then we check what fraction actually bought.

    With 10 visitors the fraction bounces around wildly. With 100,000 it
    settles on p. That settling is what a probability means.
    """
    print(f"\n  True conversion rate: {p}")
    print(f"\n  {'visitors simulated':>20} {'bought':>10} {'actual rate':>14} {'off by':>10}")
    print("  " + "-" * 58)

    for n_visitors in [10, 50, 100, 1_000, 10_000, 100_000]:
        # rng.random(n) gives n random numbers between 0 and 1.
        # A number below p counts as a purchase. Over many draws, the
        # proportion below p converges to p.
        purchases = (rng.random(n_visitors) < p).sum()
        actual_rate = purchases / n_visitors
        error = actual_rate - p

        print(f"  {n_visitors:>20,} {purchases:>10,} {actual_rate:>14.4f} {error:>+10.4f}")

    print("\n  Read the last column. With 10 visitors the observed rate can be")
    print("  far from 0.28. With 100,000 it is almost exactly 0.28.")
    print("\n  This matters commercially: a bad morning with 2 conversions out")
    print("  of 20 is not evidence that something broke. It is what a 28% rate")
    print("  looks like on a small sample.")


# ==============================================================================
# SECTION 4 -- WHICH DISTRIBUTION SHOULD I USE?
# ==============================================================================

def choose_distribution(counting, fixed_number_of_trials=None, symmetric=None):
    """
    A decision guide in code form.

    Three questions, asked in order:
        1. Are you counting or measuring?
        2. If counting: is the number of trials fixed in advance?
        3. If measuring: is the data symmetric?

    Returns the name of the distribution and the reason.
    """
    if counting:
        # Counting -> discrete -> Bernoulli, Binomial or Poisson.
        if fixed_number_of_trials is None:
            return ("Need more information",
                    "you are counting, so tell me whether n is fixed")
        if fixed_number_of_trials:
            return ("BINOMIAL",
                    "fixed number of trials, counting how many succeed "
                    "(use BERNOULLI if there is only one trial)")
        else:
            return ("POISSON",
                    "no fixed number of trials, counting events in a window")
    else:
        # Measuring -> continuous -> Normal, but only if symmetric.
        if symmetric is None:
            return ("Need more information",
                    "you are measuring, so tell me whether the data is symmetric")
        if symmetric:
            return ("NORMAL", "measuring, and the data is symmetric")
        else:
            return ("NOT NORMAL",
                    "measuring, but the data is skewed -- a normal model would "
                    "give confident wrong answers. Transform the data first, or "
                    "use percentiles instead")


def run_decision_guide():
    """Run the guide against the four case study questions plus two traps."""
    questions = [
        ("Will the next single visitor buy?",
         dict(counting=True, fixed_number_of_trials=True)),
        ("Of the next 20 visitors, how many buy?",
         dict(counting=True, fixed_number_of_trials=True)),
        ("How many complaint calls in the next hour?",
         dict(counting=True, fixed_number_of_trials=False)),
        ("What will the next parcel weigh?",
         dict(counting=False, symmetric=True)),
        ("What will the next order be worth?",
         dict(counting=False, symmetric=False)),
        ("How many orders will be returned out of 500 shipped?",
         dict(counting=True, fixed_number_of_trials=True)),
    ]

    for question, answers in questions:
        distribution, reason = choose_distribution(**answers)
        print(f"\n  Q: {question}")
        print(f"     -> {distribution}")
        print(f"        {reason}")


# ==============================================================================
# SECTION 5 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("PROBABILITY DISTRIBUTIONS -- FOUNDATION")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("1. WHAT A PROBABILITY DISTRIBUTION IS")
    print("-" * 78)
    build_a_distribution_by_hand()

    print("\n" + "-" * 78)
    print("2. DISCRETE OR CONTINUOUS -- THE FIRST QUESTION")
    print("-" * 78)
    explain_discrete_vs_continuous()

    print("\n" + "-" * 78)
    print("3. WHAT 'PROBABILITY 0.28' ACTUALLY MEANS")
    print("-" * 78)
    demonstrate_what_probability_means()

    print("\n" + "-" * 78)
    print("4. WHICH DISTRIBUTION SHOULD I USE?")
    print("-" * 78)
    run_decision_guide()

    # --------------------------------------------------------------------------
    # TRY IT YOURSELF
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRY IT YOURSELF")
    print("=" * 78)
    print("""
  1. Change CONVERSION_RATE at the top of this file from 0.28 to 0.05.
     Re-run. Notice that with a rare event you need far more simulated
     visitors before the observed rate settles down.

  2. Call the decision guide with your own question:

         print(choose_distribution(counting=True, fixed_number_of_trials=False))

  3. Change one rating count in build_a_distribution_by_hand() and re-run.
     The probabilities still add to 1 -- they always will, because each one
     is a share of the same total.
    """)


# ==============================================================================
# EXPECTED OUTPUT AND HOW TO READ IT
# ==============================================================================
"""
--------------------------------------------------------------------------------
WHAT YOU SHOULD SEE
--------------------------------------------------------------------------------

SECTION 1  A table of five ratings with their probabilities, and a TOTAL row
           showing 1.0000.

SECTION 2  Six variables sorted into Discrete or Continuous.

SECTION 3  A table where the "actual rate" column starts far from 0.28 and
           gets closer as the number of simulated visitors grows. Your exact
           numbers will match, because the random seed is fixed.

SECTION 4  Six questions, each answered with a distribution name.

--------------------------------------------------------------------------------
HOW TO READ IT
--------------------------------------------------------------------------------

THE TOTAL OF 1.0000 IN SECTION 1
    This is not a coincidence or a rounding result. Every probability is a
    share of the same total, so the shares must add to the whole. If you ever
    compute a set of probabilities that do not add to 1, stop and find the
    error before going further.

THE SHRINKING ERROR COLUMN IN SECTION 3
    At 10 visitors the observed rate can be 0.10 or 0.40. At 100,000 it is
    within a fraction of a percent of 0.28.

    A probability is a statement about the long run. It says nothing
    reliable about the next 10 visitors. This is the single most
    misunderstood idea in the whole topic, and it is why a slow morning is
    usually not evidence of a problem.

THE TWO "NOT NORMAL" STYLE ANSWERS IN SECTION 4
    "What will the next order be worth?" returns NOT NORMAL. That is correct
    and it comes straight from Day 1: order_value had a skewness of 12.2.
    A normal model applied to it would produce precise, confident, wrong
    answers -- for example, it would assign a real probability to a negative
    order value.

    Parcel weight returns NORMAL because we verified yesterday that it is
    symmetric: skewness -0.05, excess kurtosis -0.02.

    The lesson: yesterday's shape check is what makes today's distributions
    safe to use.
"""
