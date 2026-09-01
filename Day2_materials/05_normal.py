"""
================================================================================
CASE STUDY -- MERIDIAN RETAIL, DAY 2
================================================================================

Meridian's courier charges a surcharge on any parcel above 2.5 kg. Finance
wants to know what share of parcels will attract that surcharge next quarter,
so they can budget for it.

        "What fraction of parcels will exceed 2.5 kg?"

Parcel weight is a MEASUREMENT, not a count. A parcel can weigh 1.85 kg or
1.8513 kg. That makes this a continuous question, and the normal distribution
is the tool -- but only after we check one thing first.

From the Day 1 data:
        parcel weight    mean 1.85 kg, standard deviation 0.42 kg
                         skewness -0.05, excess kurtosis -0.02

That skewness near zero is the permission slip. Without it we could not use
the normal distribution at all.

--------------------------------------------------------------------------------
THIS FILE: THE NORMAL DISTRIBUTION
--------------------------------------------------------------------------------

Two inputs: mu (the mean, which sets the centre) and sigma (the standard
deviation, which sets the width). Those two numbers describe it completely.

Sections:
        1. Check the shape BEFORE choosing the distribution
        2. The empirical rule
        3. Z-scores
        4. Answering the finance question
        5. The Central Limit Theorem -- why normal appears everywhere
        6. Where the normal distribution fails
================================================================================
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)

WEIGHT_MEAN = 1.85
WEIGHT_SD = 0.42
SURCHARGE_LIMIT = 2.5


# ==============================================================================
# SECTION 1 -- CHECK THE SHAPE FIRST
# ==============================================================================

def check_before_using_normal(name, skewness, kurtosis):
    """
    The normal distribution is symmetric. If your data is not symmetric,
    a normal model will give confident, precise, wrong answers.

    The check is yesterday's material:
        skewness near 0          -> symmetric, normal is allowed
        skewness beyond +/- 1    -> skewed, normal is not allowed
    """
    safe = abs(skewness) < 0.5

    verdict = "SAFE to use normal" if safe else "DO NOT use normal"
    print(f"  {name:<20} skew {skewness:>7.2f}   kurtosis {kurtosis:>7.2f}   {verdict}")

    return safe


def run_shape_checks():
    """Run the check on three real columns from the Day 1 dataset."""
    print()
    check_before_using_normal("package_weight_kg", -0.05, -0.02)
    check_before_using_normal("order_value", 12.17, 266.70)
    check_before_using_normal("delivery_days", 8.31, 118.00)

    print("""
  Only parcel weight passes. Using a normal model on order_value would
  produce a real probability of a NEGATIVE order value, which is nonsense.

  This is why Day 1 comes before Day 2. The shape check is what makes the
  distribution choice safe.""")


# ==============================================================================
# SECTION 2 -- THE EMPIRICAL RULE
# ==============================================================================

def empirical_rule(mu, sigma):
    """
    For a normal distribution:

        68%   of values fall within mu +/- 1 sigma
        95%   within mu +/- 2 sigma
        99.7% within mu +/- 3 sigma

    These three numbers let you answer most practical questions without any
    software at all.
    """
    print(f"\n  Parcel weight: mean {mu} kg, standard deviation {sigma} kg\n")
    print(f"  {'range':<14} {'from':>8} {'to':>8} {'contains':>10}")
    print("  " + "-" * 44)

    for multiples, share in [(1, "68%"), (2, "95%"), (3, "99.7%")]:
        low = mu - multiples * sigma
        high = mu + multiples * sigma
        print(f"  mu +/- {multiples} sigma {low:>8.2f} {high:>8.2f} {share:>10}")

    print(f"\n  So {68}% of parcels weigh between {mu - sigma:.2f} and "
          f"{mu + sigma:.2f} kg.")
    print(f"  A parcel outside {mu - 3 * sigma:.2f} to {mu + 3 * sigma:.2f} kg is")
    print(f"  either a data entry error or something genuinely unusual.")


# ==============================================================================
# SECTION 3 -- Z-SCORES
# ==============================================================================

def z_score(value, mu, sigma):
    """
            z = (value - mu) / sigma

    How many standard deviations a value sits from the mean. This puts any
    measurement onto a common scale, so you can compare across variables with
    completely different units.
    """
    return (value - mu) / sigma


def show_z_scores(mu, sigma):
    """Convert several parcel weights into z-scores and probabilities."""
    print(f"\n  {'weight (kg)':>12} {'z-score':>10} {'P(heavier)':>13} {'':<26}")
    print("  " + "-" * 66)

    for weight in [1.0, 1.43, 1.85, 2.27, 2.5, 3.0]:
        z = z_score(weight, mu, sigma)
        # cdf gives P(value or less), so P(heavier) is 1 minus that.
        p_heavier = 1 - stats.norm.cdf(weight, mu, sigma)

        if abs(z) < 0.1:
            note = "the average parcel"
        elif p_heavier < 0.01:
            note = "very unusual"
        elif p_heavier < 0.10:
            note = "uncommon"
        else:
            note = ""

        print(f"  {weight:>12.2f} {z:>10.2f} {p_heavier:>13.4f} {note:<26}")

    print("\n  A z-score of 0 is exactly average. A z of +2 means only about")
    print("  2.5% of parcels are heavier.")


# ==============================================================================
# SECTION 4 -- ANSWERING THE FINANCE QUESTION
# ==============================================================================

def answer_finance(mu, sigma, limit, parcels_per_quarter=12_500):
    """
    What share of parcels exceed the surcharge limit, and what does that cost?
    """
    z = z_score(limit, mu, sigma)
    share_over = 1 - stats.norm.cdf(limit, mu, sigma)

    print(f"\n  Surcharge applies above {limit} kg.\n")
    print(f"  z = ({limit} - {mu}) / {sigma} = {z:.3f}")
    print(f"  P(parcel over {limit} kg) = {share_over:.4f}  ->  {share_over:.1%}")

    affected = share_over * parcels_per_quarter
    print(f"\n  On {parcels_per_quarter:,} parcels a quarter, that is about "
          f"{affected:,.0f} surcharged parcels.")

    # A second, more useful question: where should the limit sit?
    print(f"\n  Turning the question around -- what limit would catch only 1%?\n")
    print(f"  {'target share':>14} {'weight limit':>14}")
    print("  " + "-" * 32)
    for target in [0.10, 0.05, 0.01]:
        # ppf is the inverse of cdf: give it a probability, get the value.
        weight_limit = stats.norm.ppf(1 - target, mu, sigma)
        print(f"  {target:>13.0%} {weight_limit:>14.2f} kg")

    print(f"""
  WHAT TO TELL FINANCE

    About {share_over:.0%} of parcels breach the {limit} kg limit -- roughly
    {affected:,.0f} parcels a quarter.

    If the courier would move the limit to {stats.norm.ppf(0.99, mu, sigma):.2f} kg,
    only 1% would be surcharged. That is the number to negotiate with, and it
    came from the distribution rather than from guesswork.""")


def check_model_against_reality(mu, sigma, limit):
    """
    A model is only worth using if it matches what actually happened.

    These are the real figures from the 5,000 parcels in the Day 1 dataset.
    """
    comparisons = [
        (f"parcels over {limit} kg",
         1 - stats.norm.cdf(limit, mu, sigma),
         0.0608),
        ("parcels between 1.4 and 2.3 kg",
         stats.norm.cdf(2.3, mu, sigma) - stats.norm.cdf(1.4, mu, sigma),
         0.7144),
    ]

    print(f"\n  {'':<34} {'model predicts':>16} {'actual data':>14} {'gap':>8}")
    print("  " + "-" * 76)
    for label, predicted, actual in comparisons:
        print(f"  {label:<34} {predicted:>15.2%} {actual:>14.2%} "
              f"{predicted - actual:>+8.2%}")

    print("""
  The model is accurate to within a fraction of a percentage point.

  It worked because parcel weight is genuinely symmetric. Apply the same
  approach to order_value and it would fail badly -- see section 6.""")


# ==============================================================================
# SECTION 5 -- THE CENTRAL LIMIT THEOREM
# ==============================================================================

def central_limit_theorem(n_samples=5000):
    """
    The result that makes the normal distribution unavoidable.

    Take repeated samples from almost ANY population, compute the average of
    each sample, and those averages will follow a normal distribution -- even
    when the original data is nowhere near normal.

    We demonstrate it on the worst-behaved column we have: order value, with
    a skewness of about 12.
    """
    # Build a population shaped like Meridian's order values: heavily
    # right-skewed, most orders small, a few enormous.
    population = rng.lognormal(mean=7.6, sigma=1.1, size=200_000)

    print(f"\n  The population (individual order values):")
    print(f"    mean     : {population.mean():>12,.0f}")
    print(f"    median   : {np.median(population):>12,.0f}")
    print(f"    skewness : {stats.skew(population):>12.2f}   <- severely skewed")

    print(f"\n  Now take {n_samples:,} samples and average each one:\n")
    print(f"  {'sample size':>12} {'mean of means':>15} {'sd of means':>13} "
          f"{'skew of means':>15}")
    print("  " + "-" * 60)

    for sample_size in [1, 2, 5, 30, 100, 500]:
        # Draw n_samples samples, each of sample_size observations,
        # and take the mean of each.
        samples = rng.choice(population, size=(n_samples, sample_size))
        sample_means = samples.mean(axis=1)

        print(f"  {sample_size:>12} {sample_means.mean():>15,.0f} "
              f"{sample_means.std(ddof=1):>13,.0f} {stats.skew(sample_means):>15.2f}")

    print("""
  Read the last column downwards. At a sample size of 1 the skewness is the
  population's own severe skew of about 4.8. It falls steadily: roughly 1.3
  at a sample size of 30, 0.6 at 100, and 0.3 at 500.

  NOTE THE RULE OF THUMB BREAKING
    Textbooks say "a sample of 30 is enough for the Central Limit Theorem".
    Here it is not -- at 30 the skewness is still 1.3, which is well outside
    the symmetric range. This population is severely skewed, and the more
    skewed the population, the larger the sample you need.

    For mildly skewed data 30 really is fine. For order values it is not.
    Check rather than assume.

  WHAT THIS MEANS IN PRACTICE

    Individual order values are NOT normal, and never will be.
    The AVERAGE of 100 order values IS approximately normal.

    This is why the normal distribution is unavoidable: almost every
    confidence interval, t-test, A/B test and regression result rests on
    the average being normal, not the raw data.

  Also note the third column: the spread of the sample means SHRINKS as the
  sample grows. Bigger samples give more stable averages, which is the whole
  reason larger samples are better.""")


# ==============================================================================
# SECTION 6 -- WHERE THE NORMAL DISTRIBUTION FAILS
# ==============================================================================

def show_failure_on_skewed_data():
    """
    Apply a normal model to skewed data and show the specific nonsense it
    produces.

    We use order value: mean 7,931 and standard deviation 25,606, both taken
    from the real Day 1 dataset.
    """
    mu, sigma = 7931, 25606

    # A normal distribution is symmetric and unbounded, so it happily assigns
    # probability to values below zero.
    p_negative = stats.norm.cdf(0, mu, sigma)

    print(f"\n  Fitting a normal model to order_value (mean {mu:,}, sd {sigma:,}):\n")
    print(f"    P(order value below 0) = {p_negative:.4f}  ->  {p_negative:.1%}")
    print(f"\n  The model says {p_negative:.0%} of orders have a NEGATIVE value.")
    print(f"  That is impossible. The model is not slightly off -- it is wrong.")

    print(f"\n  It also gets the ordinary cases wrong:\n")
    print(f"  {'question':<38} {'normal says':>13} {'actual':>10}")
    print("  " + "-" * 64)
    print(f"  {'orders below the mean of 7,931':<38} "
          f"{stats.norm.cdf(mu, mu, sigma):>12.1%} {'92.3%':>10}")
    print(f"  {'orders below 2,000':<38} "
          f"{stats.norm.cdf(2000, mu, sigma):>12.1%} {'52.4%':>10}")

    print("""
  WHY IT FAILS
    The normal distribution is symmetric, so it assumes half the orders sit
    below the mean. In the real data 92% do, because the tail drags the mean
    far above the typical order.

  WHAT TO DO INSTEAD
    - Take logs first. A log transform brought order_value's skewness from
      12.2 down to 0.48 on Day 1, and the logged values are close to normal.
    - Or abandon the distribution and use percentiles directly, which make
      no shape assumption at all.

  THE RULE
    Check the shape, then choose the distribution. Never the other way round.""")


# ==============================================================================
# SECTION 7 -- RUN EVERYTHING
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("THE NORMAL DISTRIBUTION -- MEASUREMENTS, NOT COUNTS")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("1. CHECK THE SHAPE BEFORE CHOOSING THE DISTRIBUTION")
    print("-" * 78)
    run_shape_checks()

    print("\n" + "-" * 78)
    print("2. THE EMPIRICAL RULE")
    print("-" * 78)
    empirical_rule(WEIGHT_MEAN, WEIGHT_SD)

    print("\n" + "-" * 78)
    print("3. Z-SCORES")
    print("-" * 78)
    show_z_scores(WEIGHT_MEAN, WEIGHT_SD)

    print("\n" + "-" * 78)
    print("4. ANSWERING THE FINANCE QUESTION")
    print("-" * 78)
    answer_finance(WEIGHT_MEAN, WEIGHT_SD, SURCHARGE_LIMIT)

    print("\n" + "-" * 78)
    print("   CHECKING THE MODEL AGAINST THE REAL 5,000 PARCELS")
    print("-" * 78)
    check_model_against_reality(WEIGHT_MEAN, WEIGHT_SD, SURCHARGE_LIMIT)

    print("\n" + "-" * 78)
    print("5. THE CENTRAL LIMIT THEOREM")
    print("-" * 78)
    central_limit_theorem()

    print("\n" + "-" * 78)
    print("6. WHERE THE NORMAL DISTRIBUTION FAILS")
    print("-" * 78)
    show_failure_on_skewed_data()

    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRY IT YOURSELF")
    print("=" * 78)
    print("""
  1. Change SURCHARGE_LIMIT from 2.5 to 2.2 and re-run. Watch the share of
     surcharged parcels jump. Small changes to a threshold near the middle
     of a distribution have large effects.

  2. Ask your own question:

         from scipy import stats
         stats.norm.cdf(2.0, loc=1.85, scale=0.42)    # P(under 2.0 kg)
         stats.norm.ppf(0.90, loc=1.85, scale=0.42)   # the 90th percentile

  3. In central_limit_theorem(), add 1000 to the list of sample sizes. The
     skewness gets even closer to zero and the spread shrinks further.

  4. ADVANTAGES: two parameters describe it completely, the empirical rule
     gives fast answers, and the Central Limit Theorem makes it apply to
     averages of almost anything.
     DISADVANTAGES: it is symmetric so it cannot model skewed data, its
     tails are too thin for financial returns, and it allows negative values
     for quantities that cannot be negative.
    """)


# ==============================================================================
# EXPECTED OUTPUT AND HOW TO READ IT
# ==============================================================================
"""
--------------------------------------------------------------------------------
WHAT YOU SHOULD SEE
--------------------------------------------------------------------------------

SECTION 1  Three columns checked. Only package_weight_kg is marked SAFE.

SECTION 2  Three ranges: 1.43 to 2.27, 1.01 to 2.69, 0.59 to 3.11 kg.

SECTION 3  A table of weights with z-scores from about -2.02 up to +2.74.

SECTION 4  z = 1.548, P(over 2.5 kg) = 0.0609, about 761 parcels a quarter.
           Then a model-versus-reality table where the gaps are under 0.2 of
           a percentage point.

SECTION 5  A table where the "skew of means" column falls from about 4.8 at a
           sample size of 1 down to about 0.3 at 500, and the "sd of means"
           column shrinks steadily.

SECTION 6  A normal model assigning roughly 38% probability to a negative
           order value.

--------------------------------------------------------------------------------
HOW TO READ IT
--------------------------------------------------------------------------------

ONLY ONE COLUMN PASSING THE SHAPE CHECK
    Three columns, one usable. That ratio is realistic. The normal
    distribution is the one people reach for by default, and most business
    data does not qualify for it.

    Yesterday's skewness calculation is what earns you the right to use
    today's tool.

MODEL 6.09% VERSUS ACTUAL 6.08% -- THE MOST IMPORTANT NUMBER IN THIS FILE
    We predicted the surcharge rate from two numbers, mu and sigma, without
    looking at a single parcel. The prediction was within 0.2 percentage
    points of what 5,000 real parcels actually did.

    That is what a distribution buys you: answers about data you have not
    collected yet. It only works because the shape assumption held.

THE SKEW COLUMN FALLING IN SECTION 5
    This is the Central Limit Theorem happening in front of you. The
    population has a skewness of about 4.8. Averaging 100 values at a time
    brings the skewness of those averages down to about 0.6.

    The practical statement: individual order values will never be normal.
    The average of 100 of them is. Every confidence interval and A/B test you
    will ever run depends on that distinction.

    Also watch the "sd of means" column shrink. That is why bigger samples
    give more stable answers, expressed as a number rather than an assertion.

38% PROBABILITY OF A NEGATIVE ORDER VALUE
    This is the failure mode stated as plainly as it can be. The model is not
    approximately right or a little conservative. It assigns 38% probability
    to something that cannot happen.

    It also claims half of orders sit below the mean, when the real figure is
    92%.

    Nothing in the output announces an error. The numbers come back formatted
    and confident. The only protection is checking the shape first.
"""
