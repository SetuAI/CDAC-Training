"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL, DAY 2
================================================================================

Meridian Retail's site converts 28% of visitors. The marketing team ran a
campaign yesterday and 8 of the 20 visitors converted, well above the usual
5 or 6. They want to declare the campaign a success.

Before anyone does that, someone has to answer:

        "Of the next 20 visitors, how many will buy?"
        "And how often would 8 or more happen anyway, with no campaign at all?"

Twenty trials, each a yes/no. That is a BINOMIAL question.

--------------------------------------------------------------------------------
THIS FILE: THE BINOMIAL DISTRIBUTION
--------------------------------------------------------------------------------

Run the same yes/no trial n times and count the successes.

        P(X = k) = C(n, k) x p^k x (1-p)^(n-k)

Two inputs: n (number of trials) and p (probability of success each time).

Sections:
        1. The four conditions that must hold
        2. The formula, built one piece at a time
        3. A worked calculation you can do by hand
        4. The full distribution for 20 visitors
        5. Answering the marketing team
        6. Where the binomial breaks
================================================================================
"""

import math
import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)

CONVERSION_RATE = 0.28
VISITORS = 20


# ==============================================================================
# SECTION 1 -- THE FOUR CONDITIONS
# ==============================================================================

def show_conditions():
    """
    Check these before using the binomial. If any one fails, the answer is
    wrong -- and it will still look precise, which is the dangerous part.
    """
    conditions = [
        ("1. Fixed number of trials",
         "You decide n in advance. 20 visitors, not 'however many turn up'."),
        ("2. Two outcomes per trial",
         "Success or failure. Nothing in between, no third option."),
        ("3. Same probability every trial",
         "p does not change from trial 1 to trial 20."),
        ("4. Trials are independent",
         "One visitor buying does not change whether the next one buys."),
    ]

    for name, explanation in conditions:
        print(f"\n  {name}")
        print(f"     {explanation}")

    print("""
  CONDITION 4 IS THE ONE THAT BREAKS MOST OFTEN.

  If customers influence each other -- a product goes viral, a group books
  together, a queue forms -- the trials are not independent, and the binomial
  will UNDERSTATE how much the results bounce around.

  CONDITION 3 BREAKS TOO.

  Meridian's conversion rate is not really 28% all day. It is higher in the
  evening and higher during a sale. Using one average p across a whole day
  hides that.""")


# ==============================================================================
# SECTION 2 -- THE FORMULA, PIECE BY PIECE
# ==============================================================================

def count_arrangements(n, k):
    """
    C(n, k) -- how many different ways k successes can occur among n trials.

    Also written "n choose k". If 5 parcels ship and exactly 4 arrive on
    time, the late one could be any of the five, so there are 5 arrangements.

    math.comb does this directly. The formula behind it is:

            C(n, k) = n! / (k! x (n-k)!)
    """
    return math.comb(n, k)


def binomial_probability(k, n, p):
    """
    Compute P(exactly k successes in n trials) from scratch.

    Three pieces multiplied together:

        C(n, k)        how many ways k successes can be arranged
        p^k            probability those k trials all succeed
        (1-p)^(n-k)    probability the remaining trials all fail
    """
    arrangements = count_arrangements(n, k)
    success_part = p ** k
    failure_part = (1 - p) ** (n - k)

    return arrangements * success_part * failure_part


def show_arrangements_matter():
    """
    Show why C(n, k) is in the formula at all.

    Getting 4 on-time deliveries out of 5 is more likely than getting 5 out
    of 5, even though each individual delivery is 90% likely. The reason is
    that there are five different ways to get 4, and only one way to get 5.
    """
    n, p = 5, 0.9

    print(f"\n  {n} parcels, each {p:.0%} likely to arrive on time.\n")
    print(f"  {'k on time':>10} {'arrangements':>14} {'p^k':>10} "
          f"{'(1-p)^(n-k)':>14} {'probability':>13}")
    print("  " + "-" * 66)

    for k in range(n + 1):
        arrangements = count_arrangements(n, k)
        success_part = p ** k
        failure_part = (1 - p) ** (n - k)
        probability = arrangements * success_part * failure_part

        print(f"  {k:>10} {arrangements:>14} {success_part:>10.4f} "
              f"{failure_part:>14.4f} {probability:>13.5f}")

    print("\n  There is only 1 way for all 5 to be on time, but 5 different")
    print("  ways for exactly 4 to be on time. That count is what C(n, k) is.")


# ==============================================================================
# SECTION 3 -- A WORKED CALCULATION
# ==============================================================================

def worked_example():
    """
    Meridian ships 5 parcels. Each is 90% likely to arrive on time.
    What is the probability that exactly 4 arrive on time?

    Deliberately small numbers so this can be done on a whiteboard.
    """
    n, p, k = 5, 0.9, 4

    arrangements = count_arrangements(n, k)
    success_part = p ** k
    failure_part = (1 - p) ** (n - k)
    answer = arrangements * success_part * failure_part

    print(f"\n  n = {n} parcels, p = {p} on time, k = {k}\n")
    print(f"  STEP 1  count the arrangements")
    print(f"          C({n}, {k}) = {arrangements}")
    print(f"          (the late parcel could be any of the {n})\n")

    print(f"  STEP 2  probability those {k} arrive on time")
    print(f"          {p}^{k} = {success_part:.4f}\n")

    print(f"  STEP 3  probability the remaining {n - k} is late")
    print(f"          {1 - p:.1f}^{n - k} = {failure_part:.4f}\n")

    print(f"  STEP 4  multiply")
    print(f"          {arrangements} x {success_part:.4f} x {failure_part:.4f} "
          f"= {answer:.5f}\n")

    print(f"  ANSWER: {answer:.1%} chance that exactly 4 of 5 arrive on time.")

    # Verify against scipy before moving on.
    assert abs(answer - stats.binom.pmf(k, n, p)) < 1e-12
    print(f"\n  scipy agrees: stats.binom.pmf({k}, {n}, {p}) = "
          f"{stats.binom.pmf(k, n, p):.5f}")


# ==============================================================================
# SECTION 4 -- THE FULL DISTRIBUTION
# ==============================================================================

def show_full_distribution(n, p):
    """
    Print the probability of every possible outcome from 0 to n.

    This is the distribution: a probability attached to every outcome, all
    adding to 1. Seeing the whole thing at once is more useful than any
    single probability.
    """
    print(f"\n  {n} visitors, each {p:.0%} likely to buy.\n")
    print(f"  {'conversions':>12} {'probability':>13}  {'':<40}")
    print("  " + "-" * 70)

    probabilities = []
    for k in range(n + 1):
        probability = binomial_probability(k, n, p)
        probabilities.append(probability)

    # Scale the bars so the tallest is 40 characters wide.
    tallest = max(probabilities)

    for k, probability in enumerate(probabilities):
        # Skip the far tail where probabilities round to nothing, to keep
        # the output readable.
        if probability < 0.0005 and k > n * p:
            continue
        bar = "#" * int(probability / tallest * 40)
        print(f"  {k:>12} {probability:>13.4f}  {bar}")

    print("  " + "-" * 70)
    print(f"  {'TOTAL':>12} {sum(probabilities):>13.4f}")

    return probabilities


def binomial_mean_and_variance(n, p):
    """
        Mean     = n x p
        Variance = n x p x (1 - p)

    Both follow directly from the Bernoulli versions, because a binomial is
    just n Bernoulli trials added together.
    """
    mean = n * p
    variance = n * p * (1 - p)
    sd = variance ** 0.5

    print(f"\n  mean     = n x p        = {n} x {p}    = {mean:.2f} conversions")
    print(f"  variance = n x p x (1-p) = {variance:.3f}")
    print(f"  std dev  = sqrt(var)     = {sd:.3f} conversions")

    print(f"\n  A typical day sees about {mean:.1f} conversions, swinging roughly")
    print(f"  {sd:.1f} either side. Both numbers matter -- that is yesterday's")
    print(f"  lesson applied here.")

    return mean, sd


# ==============================================================================
# SECTION 5 -- ANSWERING THE MARKETING TEAM
# ==============================================================================

def answer_marketing(n, p, observed=8):
    """
    The campaign produced 8 conversions out of 20. Is that evidence it worked?

    The question to ask is: how often would 8 or more happen WITHOUT any
    campaign, just from ordinary day-to-day variation?

    Note the use of cdf. It gives the probability of a value OR LESS, so for
    "or more" we subtract from 1.
    """
    # P(8 or more) = 1 - P(7 or fewer)
    p_at_least = 1 - stats.binom.cdf(observed - 1, n, p)
    p_exactly = stats.binom.pmf(observed, n, p)

    print(f"\n  Observed: {observed} conversions from {n} visitors")
    print(f"  Expected on an ordinary day: {n * p:.1f}\n")

    print(f"  P(exactly {observed})   = {p_exactly:.4f}")
    print(f"  P({observed} or more)   = {p_at_least:.4f}  ->  {p_at_least:.1%}")

    days_per_week = p_at_least * 7

    print(f"""
  WHAT TO TELL THE MARKETING TEAM

    A result of {observed} or more happens {p_at_least:.1%} of the time with no
    campaign at all -- roughly {days_per_week:.1f} days a week.

    So {observed} conversions is a good day, not evidence that the campaign
    worked. To show a real effect you would need either a much larger result
    or many more visitors.

    This is exactly what A/B testing does: it separates a real difference
    from ordinary variation, and it is built on the binomial distribution.""")

    # Show the other tail too, so the point is symmetric.
    p_at_most_2 = stats.binom.cdf(2, n, p)
    print(f"\n    For balance: a BAD day of 2 or fewer conversions happens")
    print(f"    {p_at_most_2:.1%} of the time. That is not evidence of a problem either.")


def theory_versus_simulation(n, p, n_days=10_000):
    """
    Run 10,000 simulated days and compare the results against the formula.

    If the theory is right, the simulated proportions should land very close
    to the computed probabilities. This is the most convincing check there is,
    because it needs no algebra.
    """
    # Each simulated day: n visitors, count how many bought.
    simulated = rng.binomial(n, p, size=n_days)

    print(f"\n  {n_days:,} simulated days of {n} visitors each.\n")
    print(f"  {'conversions':>12} {'formula says':>14} {'simulation gave':>17} "
          f"{'difference':>12}")
    print("  " + "-" * 60)

    for k in range(0, 13):
        formula = binomial_probability(k, n, p)
        observed = (simulated == k).mean()
        print(f"  {k:>12} {formula:>14.4f} {observed:>17.4f} "
              f"{observed - formula:>+12.4f}")

    print(f"\n  simulated mean : {simulated.mean():.3f}   (formula: {n * p:.3f})")
    print(f"  simulated sd   : {simulated.std(ddof=1):.3f}   "
          f"(formula: {(n * p * (1 - p)) ** 0.5:.3f})")


# ==============================================================================
# SECTION 6 -- WHERE THE BINOMIAL BREAKS
# ==============================================================================

def show_where_it_breaks():
    """
    Demonstrate what happens when independence fails.

    We simulate days where visitors influence each other -- if the first few
    buy, later visitors are more likely to buy too. The mean stays about the
    same, but the spread grows sharply.
    """
    n, p, n_days = 20, 0.28, 10_000

    # Proper binomial: every visitor independent.
    independent = rng.binomial(n, p, size=n_days)

    # Clustered: each day gets its own conversion rate, drawn around 0.28.
    # This is what happens when a shared factor -- weather, a viral post, a
    # site slowdown -- affects every visitor that day.
    daily_rate = np.clip(rng.normal(p, 0.12, size=n_days), 0.01, 0.99)
    clustered = rng.binomial(n, daily_rate)

    print(f"\n  {'':<28} {'mean':>10} {'std dev':>10}")
    print("  " + "-" * 50)
    print(f"  {'independent visitors':<28} {independent.mean():>10.2f} "
          f"{independent.std(ddof=1):>10.2f}")
    print(f"  {'visitors influence each other':<28} {clustered.mean():>10.2f} "
          f"{clustered.std(ddof=1):>10.2f}")

    print(f"""
  Both have the same average. The second has a much wider spread.

  If you use the binomial when trials are not independent, your average will
  be right and your risk estimate will be too small. You will be surprised by
  bad days that your model said were rare.""")


# ==============================================================================
# SECTION 7 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("THE BINOMIAL DISTRIBUTION -- COUNTING SUCCESSES IN n TRIALS")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("1. THE FOUR CONDITIONS")
    print("-" * 78)
    show_conditions()

    print("\n" + "-" * 78)
    print("2. WHY THE FORMULA COUNTS ARRANGEMENTS")
    print("-" * 78)
    show_arrangements_matter()

    print("\n" + "-" * 78)
    print("3. WORKED CALCULATION")
    print("-" * 78)
    worked_example()

    print("\n" + "-" * 78)
    print("4. THE FULL DISTRIBUTION FOR 20 VISITORS")
    print("-" * 78)
    show_full_distribution(VISITORS, CONVERSION_RATE)
    binomial_mean_and_variance(VISITORS, CONVERSION_RATE)

    print("\n" + "-" * 78)
    print("5. ANSWERING THE MARKETING TEAM")
    print("-" * 78)
    answer_marketing(VISITORS, CONVERSION_RATE, observed=8)

    print("\n" + "-" * 78)
    print("   THEORY VERSUS SIMULATION")
    print("-" * 78)
    theory_versus_simulation(VISITORS, CONVERSION_RATE)

    print("\n" + "-" * 78)
    print("6. WHERE THE BINOMIAL BREAKS")
    print("-" * 78)
    show_where_it_breaks()

    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRY IT YOURSELF")
    print("=" * 78)
    print("""
  1. Change VISITORS from 20 to 100 and re-run. The distribution becomes
     more symmetric and starts to look like a bell curve. That is not a
     coincidence -- it is the normal distribution appearing, which is
     05_normal.py.

  2. Ask your own question:

         from scipy import stats
         1 - stats.binom.cdf(14, 50, 0.28)     # P(15 or more out of 50)

  3. Set observed=13 in answer_marketing() and re-run. Now the result IS
     rare enough to take seriously. Find the smallest number of conversions
     that would convince you.

  4. WHERE ELSE THIS APPLIES: defective units in a batch of 500, loan
     defaults in a portfolio of 200, email opens from 10,000 sends,
     patients responding in a trial of 300.
    """)


# ==============================================================================
# EXPECTED OUTPUT AND HOW TO READ IT
# ==============================================================================
"""
--------------------------------------------------------------------------------
WHAT YOU SHOULD SEE
--------------------------------------------------------------------------------

SECTION 2  A six-row table for 5 parcels. The arrangements column reads
           1, 5, 10, 10, 5, 1.

SECTION 3  A four-step calculation ending at 0.32805.

SECTION 4  A bar chart of conversions from 0 upward, tallest around 5, with
           the probabilities totalling 1.0000.
           Then mean 5.60, std dev 2.008.

SECTION 5  P(8 or more) = 0.1707, described as about 1.2 days a week.

           Then a theory-versus-simulation table where the two columns agree
           to within about 0.01.

SECTION 6  Two rows with the same mean near 5.6, but standard deviations of
           roughly 2.0 and 2.8.

--------------------------------------------------------------------------------
HOW TO READ IT
--------------------------------------------------------------------------------

THE ARRANGEMENTS COLUMN: 1, 5, 10, 10, 5, 1
    This is the whole reason C(n, k) sits in the formula. There is exactly
    one way for all 5 parcels to be on time, but ten different ways for
    exactly 3 to be on time. Outcomes in the middle have more ways to happen,
    which is why the distribution bulges in the middle.

THE TOTAL OF 1.0000 IN SECTION 4
    Same check as always. Every possible outcome from 0 to 20 is listed, so
    the probabilities must account for everything.

MEAN 5.60 WITH STANDARD DEVIATION 2.008
    Read these together, never separately. "About 5.6 conversions" on its own
    invites someone to treat 5.6 as a target. The standard deviation of 2.0
    says an ordinary day runs anywhere from roughly 4 to 8.

P(8 OR MORE) = 0.1707 -- THE COMMERCIALLY IMPORTANT NUMBER
    The marketing team saw 8 conversions and wanted to claim the campaign
    worked. This says 8 or more happens 17% of the time with no campaign at
    all -- more than once a week.

    So 8 is a good day, not proof of anything. This is the single most
    valuable use of the binomial in business: it tells you when a result is
    ordinary variation and when it is a signal worth acting on.

    Note the balancing number underneath. A bad day of 2 or fewer happens
    about 5% of the time, and is equally not evidence of a problem.

THEORY VERSUS SIMULATION AGREEING
    The formula column and the simulation column match to about two decimal
    places. This is the proof that the maths describes reality. If you ever
    doubt a probability calculation, simulate it -- 10,000 runs will tell you
    quickly whether your formula is right.

SECTION 6: SAME MEAN, DIFFERENT SPREAD
    When visitors influence each other, the average is unchanged but the
    spread grows by roughly 40%. The binomial would have told you the quiet
    days and the busy days were rarer than they really are.

    This is the practical danger with the binomial. It does not fail loudly.
    It returns a confident, precise number that understates your risk.
"""
