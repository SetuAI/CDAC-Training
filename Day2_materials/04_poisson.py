"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL, DAY 2
================================================================================

Meridian's support desk receives an average of 6 complaint calls per hour.
The operations manager currently staffs enough agents to handle 6 calls an
hour and is being asked why the queue keeps overflowing.

        "How many complaint calls will arrive in the next hour?"
        "How many agents do we need to cover 95% of hours?"

Notice what is different about this question. There is no fixed number of
trials. Calls are not 20 attempts that either happen or do not -- they simply
arrive. That makes this a POISSON question, not a binomial one.

--------------------------------------------------------------------------------
THIS FILE: THE POISSON DISTRIBUTION
--------------------------------------------------------------------------------

Counting events in a fixed window of time or space, when there is no fixed n.

        P(X = k) = (e^(-lambda) x lambda^k) / k!

One input: lambda, the average number of events per window.

Sections:
        1. Why this is not a binomial question
        2. The formula and a worked calculation
        3. The full distribution for 6 calls an hour
        4. The mean = variance property, and the free model check it gives you
        5. Answering the staffing question
        6. Overdispersion -- where Poisson breaks
================================================================================
"""

import math
import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)

CALLS_PER_HOUR = 6


# ==============================================================================
# SECTION 1 -- WHY THIS IS NOT A BINOMIAL QUESTION
# ==============================================================================

def binomial_or_poisson():
    """
    The distinguishing question: is there a fixed number of trials?

    Binomial needs an n you decide in advance. Poisson does not have one at
    all -- events just arrive at some rate.
    """
    examples = [
        ("Of 20 visitors, how many buy?", "BINOMIAL",
         "n = 20, fixed in advance"),
        ("How many calls arrive this hour?", "POISSON",
         "no n -- calls are not a fixed set of attempts"),
        ("Of 500 parcels shipped, how many are returned?", "BINOMIAL",
         "n = 500, fixed in advance"),
        ("How many machine breakdowns this shift?", "POISSON",
         "no n -- breakdowns are not attempts"),
        ("How many potholes per kilometre of road?", "POISSON",
         "the window is space, not time -- still works"),
    ]

    print(f"\n  {'question':<44} {'use':<10} why")
    print("  " + "-" * 96)
    for question, answer, reason in examples:
        print(f"  {question:<44} {answer:<10} {reason}")

    print("\n  The test: can you name the number of trials? If not, it is Poisson.")


# ==============================================================================
# SECTION 2 -- THE FORMULA
# ==============================================================================

def poisson_probability(k, lam):
    """
    Compute P(exactly k events) from scratch.

            P(X = k) = (e^(-lambda) x lambda^k) / k!

    where:
        lam   is the average number of events per window
        k     is the number of events you are asking about
        e     is 2.71828..., a fixed mathematical constant
        k!    is k factorial: 4! = 4 x 3 x 2 x 1 = 24

    You will rarely type this out in practice. What matters is knowing when
    it applies, and that lambda is the only thing you need to supply.
    """
    if lam <= 0:
        raise ValueError("lambda must be positive")

    # math.exp(-lam) is e raised to the power of -lam.
    # math.factorial(k) is k!.
    return (math.exp(-lam) * lam ** k) / math.factorial(k)


def worked_example():
    """
    A support desk receives an average of 3 calls per hour.
    What is the probability of exactly 2 calls in the next hour?

    Small numbers so this works on a whiteboard.
    """
    lam, k = 3, 2

    step1 = math.exp(-lam)
    step2 = lam ** k
    step3 = math.factorial(k)
    answer = (step1 * step2) / step3

    print(f"\n  lambda = {lam} calls per hour, k = {k}\n")
    print(f"  STEP 1  e^(-{lam})        = {step1:.4f}")
    print(f"  STEP 2  {lam}^{k}            = {step2}")
    print(f"  STEP 3  {k}!              = {step3}")
    print(f"  STEP 4  ({step1:.4f} x {step2}) / {step3} = {answer:.4f}\n")
    print(f"  ANSWER: {answer:.1%} chance of exactly {k} calls in the next hour.")

    assert abs(answer - stats.poisson.pmf(k, lam)) < 1e-12
    print(f"\n  scipy agrees: stats.poisson.pmf({k}, mu={lam}) = "
          f"{stats.poisson.pmf(k, lam):.4f}")


# ==============================================================================
# SECTION 3 -- THE FULL DISTRIBUTION
# ==============================================================================

def show_full_distribution(lam, max_k=16):
    """
    Print the probability of every call count from 0 upward.

    Unlike the binomial, there is no upper limit -- 50 calls in an hour is
    possible, just extremely unlikely. We stop the table where the
    probabilities become negligible.
    """
    print(f"\n  Average {lam} calls per hour.\n")
    print(f"  {'calls':>7} {'probability':>13} {'cumulative':>12}  {'':<36}")
    print("  " + "-" * 74)

    probabilities = [poisson_probability(k, lam) for k in range(max_k + 1)]
    tallest = max(probabilities)
    running_total = 0.0

    for k, probability in enumerate(probabilities):
        running_total += probability
        bar = "#" * int(probability / tallest * 36)
        print(f"  {k:>7} {probability:>13.4f} {running_total:>12.4f}  {bar}")

    print("  " + "-" * 74)
    print(f"\n  The cumulative column reaches {running_total:.4f} by {max_k} calls.")
    print(f"  It never quite reaches 1, because there is no upper limit --")
    print(f"  20 calls in an hour is possible, just vanishingly unlikely.")

    return probabilities


# ==============================================================================
# SECTION 4 -- MEAN = VARIANCE, AND THE FREE MODEL CHECK
# ==============================================================================

def show_mean_equals_variance(lam, n_hours=100_000):
    """
    The property that makes Poisson distinctive:

            Mean = lambda
            Variance = lambda

    They are the same number. No other common distribution does this.

    This gives you a free diagnostic. Take real count data, compute its mean
    and its variance:

        roughly equal          -> Poisson is a reasonable model
        variance >> mean       -> the data is OVERDISPERSED and Poisson will
                                  understate your risk
    """
    simulated = rng.poisson(lam, size=n_hours)

    print(f"\n  Theory:     mean = lambda = {lam}")
    print(f"              variance = lambda = {lam}\n")
    print(f"  Simulated {n_hours:,} hours:")
    print(f"              mean     = {simulated.mean():.4f}")
    print(f"              variance = {simulated.var(ddof=1):.4f}")

    ratio = simulated.var(ddof=1) / simulated.mean()
    print(f"\n  variance / mean = {ratio:.3f}   (should be close to 1)")

    print("""
  THE TWO-LINE CHECK YOU CAN RUN ON ANY COUNT DATA

      ratio = data.var() / data.mean()

      close to 1     -> Poisson fits
      well above 1   -> overdispersed, Poisson understates the tail
      well below 1   -> underdispersed, rarer in practice

  This is the most useful thing in this file. It costs one line and it tells
  you whether you are allowed to use the model at all.""")


# ==============================================================================
# SECTION 5 -- ANSWERING THE STAFFING QUESTION
# ==============================================================================

def answer_staffing(lam):
    """
    The manager staffs for 6 calls an hour because 6 is the average.
    Show what that actually delivers, and what would be needed instead.
    """
    # P(more than 6) = 1 - P(6 or fewer)
    overwhelmed = 1 - stats.poisson.cdf(lam, lam)

    print(f"\n  Current plan: staff for {lam} calls, the average.\n")
    print(f"  P(more than {lam} calls arrive) = {overwhelmed:.4f}  ->  "
          f"{overwhelmed:.1%} of hours")
    print(f"  That is roughly {overwhelmed * 8:.1f} hours of every 8-hour shift")
    print(f"  where the queue overflows.")

    print(f"\n  How much capacity is needed for different coverage levels:\n")
    print(f"  {'coverage target':>16} {'staff for':>11} {'hours overflowing':>20}")
    print("  " + "-" * 52)

    for coverage in [0.50, 0.80, 0.90, 0.95, 0.99]:
        # ppf is the inverse of cdf: give it a probability, it returns the
        # count at that percentile.
        capacity = int(stats.poisson.ppf(coverage, lam))
        actual_overflow = 1 - stats.poisson.cdf(capacity, lam)
        print(f"  {coverage:>15.0%} {capacity:>11} {actual_overflow:>19.1%}")

    print(f"\n  Some specific probabilities:")
    for k in [10, 12, 15]:
        p = 1 - stats.poisson.cdf(k - 1, lam)
        print(f"    P({k} or more calls in an hour) = {p:.4f}  ({p:.1%})")

    capacity_95 = int(stats.poisson.ppf(0.95, lam))

    print(f"""
  WHAT TO TELL THE OPERATIONS MANAGER

    Staffing for the average of {lam} leaves the desk overwhelmed
    {overwhelmed:.0%} of hours. The average was never a capacity plan --
    it is the point where you are overwhelmed roughly half the time.

    To cover 95% of hours, staff for {capacity_95} calls, not {lam}.

    This is the same conclusion as yesterday's delivery promise question:
    plan against the tail, not the average.""")


# ==============================================================================
# SECTION 6 -- OVERDISPERSION, WHERE POISSON BREAKS
# ==============================================================================

def show_overdispersion(lam, n_hours=20_000):
    """
    Real events cluster. One system outage produces fifty calls at once.

    When events cluster, the variance grows well beyond the mean and Poisson
    understates the busy hours -- which are exactly the hours you were trying
    to plan for.
    """
    # A well-behaved Poisson process.
    clean = rng.poisson(lam, size=n_hours)

    # A clustered process: most hours are quiet, but 5% of hours have an
    # incident that generates a burst of extra calls.
    incident = rng.random(n_hours) < 0.05
    clustered = rng.poisson(lam * 0.7, size=n_hours) + incident * rng.poisson(30, size=n_hours)

    print(f"\n  {'':<26} {'mean':>9} {'variance':>11} {'var/mean':>10} {'max':>7}")
    print("  " + "-" * 66)
    print(f"  {'clean Poisson process':<26} {clean.mean():>9.2f} "
          f"{clean.var(ddof=1):>11.2f} {clean.var(ddof=1) / clean.mean():>10.2f} "
          f"{clean.max():>7}")
    print(f"  {'calls cluster in bursts':<26} {clustered.mean():>9.2f} "
          f"{clustered.var(ddof=1):>11.2f} "
          f"{clustered.var(ddof=1) / clustered.mean():>10.2f} {clustered.max():>7}")

    # What a Poisson model would predict versus what actually happens.
    threshold = 15
    poisson_says = 1 - stats.poisson.cdf(threshold - 1, clustered.mean())
    reality = (clustered >= threshold).mean()

    print(f"\n  P({threshold} or more calls in an hour):")
    print(f"    Poisson model predicts : {poisson_says:.4f}")
    print(f"    actually happens       : {reality:.4f}")
    if poisson_says > 0:
        print(f"    the model understates it by {reality / poisson_says:.0f} times")

    print("""
  THE FIX
    When the variance / mean ratio is well above 1, use the NEGATIVE BINOMIAL
    distribution instead. It has a second parameter that handles the extra
    spread.

    Worth knowing the name. The mechanics are beyond today, but recognising
    that you need it is the important part -- and the two-line check from
    section 4 is what tells you.""")


# ==============================================================================
# SECTION 7 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("THE POISSON DISTRIBUTION -- COUNTING EVENTS IN A WINDOW")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("1. WHY THIS IS NOT A BINOMIAL QUESTION")
    print("-" * 78)
    binomial_or_poisson()

    print("\n" + "-" * 78)
    print("2. WORKED CALCULATION")
    print("-" * 78)
    worked_example()

    print("\n" + "-" * 78)
    print("3. THE FULL DISTRIBUTION FOR 6 CALLS AN HOUR")
    print("-" * 78)
    show_full_distribution(CALLS_PER_HOUR)

    print("\n" + "-" * 78)
    print("4. MEAN = VARIANCE, AND THE FREE MODEL CHECK")
    print("-" * 78)
    show_mean_equals_variance(CALLS_PER_HOUR)

    print("\n" + "-" * 78)
    print("5. ANSWERING THE STAFFING QUESTION")
    print("-" * 78)
    answer_staffing(CALLS_PER_HOUR)

    print("\n" + "-" * 78)
    print("6. OVERDISPERSION -- WHERE POISSON BREAKS")
    print("-" * 78)
    show_overdispersion(CALLS_PER_HOUR)

    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRY IT YOURSELF")
    print("=" * 78)
    print("""
  1. Change CALLS_PER_HOUR from 6 to 2 and re-run. With a low rate the
     distribution is lopsided, with a long tail to the right. Now try 40 --
     it becomes symmetric and bell-shaped. A Poisson with a large lambda
     looks like a normal distribution, which is 05_normal.py.

  2. Ask your own question:

         from scipy import stats
         1 - stats.poisson.cdf(24, mu=18)     # P(25 or more when avg is 18)
         stats.poisson.ppf(0.99, mu=18)       # capacity to cover 99% of hours

  3. Run the two-line check on any count column you have:

         ratio = data.var() / data.mean()

     Anything much above 1 means Poisson will understate your busy periods.

  4. WHERE ELSE THIS APPLIES: insurance claims per month, server requests
     per second, defects per square metre of fabric, patients admitted per
     hour, stockouts per week.
    """)


# ==============================================================================
# EXPECTED OUTPUT AND HOW TO READ IT
# ==============================================================================
"""
--------------------------------------------------------------------------------
WHAT YOU SHOULD SEE
--------------------------------------------------------------------------------

SECTION 2  A four-step calculation ending at 0.2240.

SECTION 3  A bar chart of call counts 0 to 16, tallest at 5 and 6, with a
           cumulative column that climbs towards but never reaches 1.

SECTION 4  Simulated mean and variance both close to 6, and a variance/mean
           ratio close to 1.000.

SECTION 5  P(more than 6 calls) around 0.394, then a coverage table showing
           that 95% coverage needs capacity for 10 calls.

SECTION 6  Two rows with similar means but very different variances, and a
           model that understates the busy hours by a large multiple.

--------------------------------------------------------------------------------
HOW TO READ IT
--------------------------------------------------------------------------------

THE TALLEST BARS AT 5 AND 6, NOT ONLY AT 6
    The average is 6, but the single most likely outcomes are 5 and 6, each
    at about 16%. There is no outcome that happens most of the time. Even the
    most likely call count occurs in only one hour out of six.

    This is why "we average 6 calls an hour" is close to useless as a plan.

THE CUMULATIVE COLUMN NOT REACHING 1
    Poisson has no upper limit. Twenty calls in an hour is possible. This is
    the structural difference from the binomial, where the count can never
    exceed n.

VARIANCE / MEAN CLOSE TO 1.000
    This is the signature of a genuine Poisson process, and it is your model
    check. Run it on real data before trusting any Poisson result.

P(MORE THAN 6 CALLS) = 0.394 -- THE OPERATIONALLY IMPORTANT NUMBER
    The manager staffed for the average and was surprised the queue overflows.
    This says it overflows 39% of hours -- about 3 hours of every 8-hour shift.

    An average is not a capacity plan. By construction, staffing to the average
    means being overwhelmed a large share of the time.

    To cover 95% of hours you need capacity for 10 calls, not 6. That is a
    67% increase over the average, and it is entirely invisible if you only
    ever look at the mean.

SECTION 6: THE MODEL UNDERSTATING BUSY HOURS
    When calls arrive in bursts, the variance/mean ratio climbs far above 1
    and the Poisson model badly underestimates how often a very busy hour
    occurs.

    This is the practical failure mode. Real support calls DO cluster --
    an outage generates a burst. So run the variance/mean check first, and if
    it comes back well above 1, treat any Poisson answer as optimistic.
"""
