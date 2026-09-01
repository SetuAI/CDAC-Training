"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL, DAY 2
================================================================================

All four questions in one place.

The operations team is planning next week. From the Day 1 data we know:

        conversion rate      28% of visitors place an order
        complaint calls      6 per hour on average
        parcel weight        mean 1.85 kg, standard deviation 0.42 kg
                             (skewness -0.05, so genuinely symmetric)

        1. The next visitor lands on the site. Will they buy?
        2. Of the next 20 visitors, how many will buy?
        3. How many complaint calls will arrive in the next hour?
        4. What fraction of parcels will exceed the 2.5 kg surcharge limit?

--------------------------------------------------------------------------------
THIS FILE: CHOOSING, ANSWERING, AND CHECKING
--------------------------------------------------------------------------------

        1. How the four distributions connect to each other
        2. A decision guide
        3. All four questions answered end to end
        4. The recommendation for each team
        5. Common mistakes, demonstrated rather than described
================================================================================
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)

CONVERSION_RATE = 0.28
VISITORS = 20
CALLS_PER_HOUR = 6
WEIGHT_MEAN = 1.85
WEIGHT_SD = 0.42
SURCHARGE_LIMIT = 2.5


# ==============================================================================
# SECTION 1 -- HOW THE FOUR CONNECT
# ==============================================================================

def show_the_chain():
    """
    The four distributions are not unrelated tools. They form a chain, and
    each step has a condition attached.
    """
    print("""
  BERNOULLI          one trial, yes or no
      |
      |  add up n independent trials
      v
  BINOMIAL           how many successes out of n
      |
      |  let n get very large and p very small
      v
  POISSON            events arriving in a window, no fixed n
      |
      |  let the average get large
      v
  NORMAL             a smooth, symmetric curve
""")

    # Demonstrate the last two links with real numbers.
    n, p = 2000, 0.003
    lam = n * p

    print(f"  BINOMIAL BECOMING POISSON")
    print(f"    2,000 customers, each 0.3% likely to complain this hour.")
    print(f"    Binomial mean = n x p = {lam:.1f}, which is our lambda.\n")
    print(f"    {'complaints':>11} {'binomial':>11} {'poisson':>11} {'difference':>12}")
    print("    " + "-" * 47)
    for k in range(0, 8):
        b = stats.binom.pmf(k, n, p)
        po = stats.poisson.pmf(k, lam)
        print(f"    {k:>11} {b:>11.5f} {po:>11.5f} {b - po:>+12.5f}")

    print("\n    Nearly identical. This is why Poisson works for rare events")
    print("    across many opportunities -- it IS a binomial in that situation,")
    print("    with one input instead of two.\n")

    print(f"  POISSON BECOMING NORMAL")
    for lam_test in [2, 10, 50]:
        sample = rng.poisson(lam_test, 50_000)
        print(f"    lambda = {lam_test:>3}  ->  skewness of the counts = "
              f"{stats.skew(sample):>5.2f}")
    print("\n    As the average grows the shape becomes symmetric, which means")
    print("    a normal curve describes it well.")


# ==============================================================================
# SECTION 2 -- THE DECISION GUIDE
# ==============================================================================

def choose_distribution(counting, fixed_n=None, symmetric=None):
    """
    Three questions in order:
        1. counting or measuring?
        2. if counting: is n fixed in advance?
        3. if measuring: is the data symmetric?
    """
    if counting:
        if fixed_n:
            return "BINOMIAL (or BERNOULLI if n = 1)"
        return "POISSON"
    if symmetric:
        return "NORMAL"
    return "NOT NORMAL -- transform first, or use percentiles"


def show_decision_table():
    """Print the guide as a lookup table."""
    rows = [
        ("Will this one thing happen?", "counting, n = 1", "BERNOULLI"),
        ("How many of n tries succeed?", "counting, n fixed", "BINOMIAL"),
        ("How many events this hour?", "counting, no n", "POISSON"),
        ("What will this measurement be?", "measuring, symmetric", "NORMAL"),
        ("What will this measurement be?", "measuring, skewed", "none of them"),
    ]

    print(f"\n  {'your question':<34} {'situation':<24} {'use'}")
    print("  " + "-" * 76)
    for question, situation, answer in rows:
        print(f"  {question:<34} {situation:<24} {answer}")

    print("""
  THE THREE CHECKS THAT CATCH MOST ERRORS

    1. Am I counting or measuring?   Counts are never normal.
    2. Is n fixed in advance?        If not, it is not binomial.
    3. Is the data symmetric?        If not, it is not normal.""")


# ==============================================================================
# SECTION 3 -- ALL FOUR QUESTIONS ANSWERED
# ==============================================================================

def question_1():
    """The next visitor. One trial, two outcomes -> Bernoulli."""
    p = CONVERSION_RATE

    print(f"\n  Distribution: BERNOULLI   (one trial, two outcomes)")
    print(f"  Input: p = {p}\n")
    print(f"    P(buys)          = {p:.4f}")
    print(f"    P(does not buy)  = {1 - p:.4f}")
    print(f"    mean             = {p:.4f}")
    print(f"    variance         = {p * (1 - p):.4f}")

    print(f"\n  READING IT: there is a 28% chance this visitor buys. The mean of")
    print(f"  0.28 is a long-run rate -- this one visitor will be a 0 or a 1.")


def question_2():
    """Twenty visitors. Fixed n, counting successes -> Binomial."""
    n, p = VISITORS, CONVERSION_RATE

    mean = n * p
    sd = (n * p * (1 - p)) ** 0.5

    print(f"\n  Distribution: BINOMIAL   (n fixed at {n}, counting successes)")
    print(f"  Inputs: n = {n}, p = {p}\n")
    print(f"    expected conversions = {mean:.2f}")
    print(f"    standard deviation   = {sd:.2f}")
    print(f"    P(exactly 5)         = {stats.binom.pmf(5, n, p):.4f}")
    print(f"    P(8 or more)         = {1 - stats.binom.cdf(7, n, p):.4f}")
    print(f"    P(2 or fewer)        = {stats.binom.cdf(2, n, p):.4f}")

    p_high = 1 - stats.binom.cdf(7, n, p)

    print(f"\n  READING IT: expect about 5 or 6 conversions, swinging roughly 2")
    print(f"  either side. A result of 8 or more happens {p_high:.0%} of the time")
    print(f"  with nothing changed -- about {p_high * 7:.1f} days a week. So 8")
    print(f"  conversions is a good day, not proof that a campaign worked.")


def question_3():
    """Complaint calls. No fixed n, events in a window -> Poisson."""
    lam = CALLS_PER_HOUR

    overwhelmed = 1 - stats.poisson.cdf(lam, lam)
    capacity_95 = int(stats.poisson.ppf(0.95, lam))

    print(f"\n  Distribution: POISSON   (no fixed n, events per hour)")
    print(f"  Input: lambda = {lam}\n")
    print(f"    P(exactly 6 calls)   = {stats.poisson.pmf(6, lam):.4f}")
    print(f"    P(10 or more)        = {1 - stats.poisson.cdf(9, lam):.4f}")
    print(f"    P(12 or more)        = {1 - stats.poisson.cdf(11, lam):.4f}")
    print(f"    P(more than {lam}, the average) = {overwhelmed:.4f}")
    print(f"    capacity for 95% of hours = {capacity_95} calls")

    print(f"\n  READING IT: staffing for the average of {lam} leaves the desk")
    print(f"  overwhelmed {overwhelmed:.0%} of hours. To cover 95% of hours you")
    print(f"  need capacity for {capacity_95} calls. An average is not a plan.")


def question_4():
    """Parcel weight. A measurement, and symmetric -> Normal."""
    mu, sigma, limit = WEIGHT_MEAN, WEIGHT_SD, SURCHARGE_LIMIT

    z = (limit - mu) / sigma
    share = 1 - stats.norm.cdf(limit, mu, sigma)

    print(f"\n  Distribution: NORMAL   (a measurement, and skewness is -0.05)")
    print(f"  Inputs: mu = {mu}, sigma = {sigma}\n")
    print(f"    z for {limit} kg        = {z:.3f}")
    print(f"    P(over {limit} kg)      = {share:.4f}")
    print(f"    68% of parcels weigh {mu - sigma:.2f} to {mu + sigma:.2f} kg")
    print(f"    95% of parcels weigh {mu - 2 * sigma:.2f} to {mu + 2 * sigma:.2f} kg")
    print(f"    limit catching only 1%  = {stats.norm.ppf(0.99, mu, sigma):.2f} kg")

    print(f"\n  READING IT: about {share:.0%} of parcels attract the surcharge.")
    print(f"  The permission to use a normal model came from the shape check --")
    print(f"  without a skewness near zero, none of these numbers would hold.")


def verify_against_reality():
    """
    The check that decides whether any of this was worth doing.

    These are the actual figures from the 5,000 parcels in the Day 1 dataset.
    """
    mu, sigma = WEIGHT_MEAN, WEIGHT_SD

    checks = [
        ("parcels over 2.5 kg",
         1 - stats.norm.cdf(2.5, mu, sigma), 0.0608),
        ("parcels between 1.4 and 2.3 kg",
         stats.norm.cdf(2.3, mu, sigma) - stats.norm.cdf(1.4, mu, sigma), 0.7144),
    ]

    print(f"\n  {'':<34} {'predicted':>12} {'actual':>10} {'gap':>9}")
    print("  " + "-" * 68)
    for label, predicted, actual in checks:
        print(f"  {label:<34} {predicted:>11.2%} {actual:>10.2%} "
              f"{predicted - actual:>+9.2%}")

    print("""
  We predicted these from two numbers, without looking at a single parcel,
  and landed within a fifth of a percentage point of what 5,000 real parcels
  did.

  That is the entire value of a distribution: answers about data you have
  not collected yet.""")


# ==============================================================================
# SECTION 4 -- WHAT TO TELL EACH TEAM
# ==============================================================================

def recommendations():
    """The business output. A number nobody acts on is not an answer."""
    lam = CALLS_PER_HOUR
    overwhelmed = 1 - stats.poisson.cdf(lam, lam)
    capacity_95 = int(stats.poisson.ppf(0.95, lam))
    p_high = 1 - stats.binom.cdf(7, VISITORS, CONVERSION_RATE)
    share = 1 - stats.norm.cdf(SURCHARGE_LIMIT, WEIGHT_MEAN, WEIGHT_SD)

    print(f"""
  TO MARKETING
    Yesterday's 8 conversions from 20 visitors is not evidence the campaign
    worked. That result or better occurs {p_high:.0%} of the time with no
    campaign at all. Before claiming an effect, run a proper A/B test with
    enough visitors -- 20 is far too few to detect anything.

  TO OPERATIONS
    Staffing the support desk for the average of {lam} calls an hour leaves
    it overwhelmed {overwhelmed:.0%} of hours. Staff for {capacity_95} to cover
    95% of hours. Also run the variance-to-mean check on the real call log
    first: if calls cluster around incidents, even {capacity_95} will be
    optimistic.

  TO FINANCE
    Budget for about {share:.0%} of parcels attracting the weight surcharge.
    If the courier will move the threshold to
    {stats.norm.ppf(0.99, WEIGHT_MEAN, WEIGHT_SD):.2f} kg, that drops to 1%.
    That is a concrete number to negotiate with.

  TO EVERYONE
    Every one of these answers rests on an assumption -- constant rates,
    independent events, a symmetric shape. State the assumption whenever you
    state the number. A probability without its assumption is not a finding.""")


# ==============================================================================
# SECTION 5 -- COMMON MISTAKES, DEMONSTRATED
# ==============================================================================

def mistake_normal_on_counts():
    """Using a normal model where a count belongs."""
    lam = CALLS_PER_HOUR

    # A normal fitted to the call counts. Poisson variance = lambda, so
    # sigma = sqrt(lambda).
    sigma = lam ** 0.5
    p_negative = stats.norm.cdf(-0.5, lam, sigma)

    print(f"\n  MISTAKE 1: using NORMAL for call counts")
    print(f"    A normal model fitted to calls (mean {lam}, sd {sigma:.2f}) gives")
    print(f"    P(fewer than 0 calls) = {p_negative:.4f}")
    print(f"    Calls cannot be negative, and cannot be 6.3. Counts are never normal.")


def mistake_binomial_without_fixed_n():
    """Using binomial where there is no n."""
    print(f"\n  MISTAKE 2: using BINOMIAL for complaint calls")
    print(f"    To use a binomial you must supply n. What is n for 'complaints")
    print(f"    tomorrow'? Every customer? Everyone who might call? There is no")
    print(f"    defensible answer, and that is the signal to use Poisson.")


def mistake_ignoring_overdispersion():
    """Using Poisson on clustered data."""
    lam, n_hours = CALLS_PER_HOUR, 20_000

    incident = rng.random(n_hours) < 0.05
    clustered = rng.poisson(lam * 0.7, n_hours) + incident * rng.poisson(30, n_hours)

    ratio = clustered.var(ddof=1) / clustered.mean()
    poisson_says = 1 - stats.poisson.cdf(14, clustered.mean())
    reality = (clustered >= 15).mean()

    print(f"\n  MISTAKE 3: using POISSON when events cluster")
    print(f"    variance / mean = {ratio:.1f}   (should be near 1 for Poisson)")
    print(f"    P(15+ calls): Poisson says {poisson_says:.4f}, reality is {reality:.4f}")
    if poisson_says > 0:
        print(f"    The model understates busy hours by {reality / poisson_says:.0f} times.")


def mistake_normal_on_skewed():
    """The most common and most expensive error."""
    mu, sigma = 7931, 25606
    p_negative = stats.norm.cdf(0, mu, sigma)

    print(f"\n  MISTAKE 4: using NORMAL on skewed data")
    print(f"    order_value has a skewness of 12.2. Fit a normal to it and it")
    print(f"    reports P(order value below 0) = {p_negative:.1%}.")
    print(f"    It also claims 50% of orders fall below the mean. The real")
    print(f"    figure is 92%.")


# ==============================================================================
# SECTION 6 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("CHOOSING A DISTRIBUTION, AND THE FULL CASE STUDY")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("1. HOW THE FOUR DISTRIBUTIONS CONNECT")
    print("-" * 78)
    show_the_chain()

    print("\n" + "-" * 78)
    print("2. THE DECISION GUIDE")
    print("-" * 78)
    show_decision_table()

    print("\n" + "=" * 78)
    print("3. THE FOUR QUESTIONS")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("QUESTION 1 -- Will the next visitor buy?")
    print("-" * 78)
    question_1()

    print("\n" + "-" * 78)
    print("QUESTION 2 -- Of the next 20 visitors, how many will buy?")
    print("-" * 78)
    question_2()

    print("\n" + "-" * 78)
    print("QUESTION 3 -- How many complaint calls in the next hour?")
    print("-" * 78)
    question_3()

    print("\n" + "-" * 78)
    print("QUESTION 4 -- What fraction of parcels exceed 2.5 kg?")
    print("-" * 78)
    question_4()

    print("\n" + "-" * 78)
    print("   CHECKING THE MODEL AGAINST 5,000 REAL PARCELS")
    print("-" * 78)
    verify_against_reality()

    print("\n" + "=" * 78)
    print("4. WHAT TO TELL EACH TEAM")
    print("=" * 78)
    recommendations()

    print("\n" + "=" * 78)
    print("5. COMMON MISTAKES")
    print("=" * 78)
    mistake_normal_on_counts()
    mistake_binomial_without_fixed_n()
    mistake_ignoring_overdispersion()
    mistake_normal_on_skewed()

    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRY IT YOURSELF")
    print("=" * 78)
    print("""
  1. Change VISITORS from 20 to 200 in this file and re-run question 2.
     With 200 visitors, is 80 conversions still an ordinary result? Larger
     samples make real effects detectable.

  2. Use the decision guide on a question from your own work:

         choose_distribution(counting=True, fixed_n=False)

  3. Change CALLS_PER_HOUR to 20 and re-run question 3. Notice the capacity
     needed for 95% coverage no longer sits so far above the average, in
     relative terms. Higher-volume processes are proportionally easier to
     plan for.
    """)


# ==============================================================================
# EXPECTED OUTPUT AND HOW TO READ IT
# ==============================================================================
"""
--------------------------------------------------------------------------------
WHAT YOU SHOULD SEE
--------------------------------------------------------------------------------

SECTION 1  A binomial column and a Poisson column agreeing to about four
           decimal places, then Poisson skewness falling from about 0.71 at
           lambda = 2 to about 0.14 at lambda = 50.

SECTION 3  Four answers:
             Q1  P(buys) = 0.2800
             Q2  mean 5.60, P(8 or more) = 0.1707
             Q3  P(10 or more) = 0.0839, capacity for 95% = 10 calls
             Q4  z = 1.548, P(over 2.5 kg) = 0.0609

           Then a model-versus-reality table with gaps under 0.2 percentage
           points.

SECTION 5  Four demonstrated mistakes, each producing a visibly wrong number.

--------------------------------------------------------------------------------
HOW TO READ IT
--------------------------------------------------------------------------------

BINOMIAL AND POISSON AGREEING IN SECTION 1
    These are not two separate models that happen to be close. When n is
    large and p is small, the Poisson IS the binomial, expressed with one
    input instead of two.

    Practical use: if you are counting rare events across many opportunities
    and you do not know n exactly, you do not need it. Use the average and
    reach for Poisson.

POISSON SKEWNESS FALLING AS LAMBDA GROWS
    At lambda = 2 the distribution is lopsided. At lambda = 50 it is nearly
    symmetric, which is why a normal curve approximates a high-volume count
    process well. This is the last link in the chain.

THE FOUR ANSWERS TOGETHER
    Notice that each question needed a different distribution, and the choice
    was driven entirely by the shape of the question rather than by the data:

        one trial               -> Bernoulli
        fixed n                 -> Binomial
        no fixed n              -> Poisson
        a measurement, symmetric -> Normal

    Getting this choice right is most of the skill. The arithmetic afterwards
    is one line of scipy.

PREDICTED 6.09% VERSUS ACTUAL 6.08%
    This is the payoff of the whole day. Two numbers -- a mean and a standard
    deviation -- predicted the behaviour of 5,000 parcels we never examined,
    to within a fifth of a percentage point.

THE FOUR MISTAKES
    Every one produces a number that looks fine. A normal model on call counts
    returns a clean probability. A normal model on order value returns a clean
    probability. Neither announces that it is nonsense.

    That is why the checks come first:
        counting or measuring?
        is n fixed?
        is the data symmetric?
        is variance close to the mean?

    Four questions, each one line of code, and they prevent every mistake in
    this section.
"""
