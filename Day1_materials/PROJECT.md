# PROJECT — Meridian Retail: Reading the FY26 Order Book

**Capstone case study — Descriptive Statistics**
Dataset: `meridian_orders.xlsx` (sheet `Orders`) or `meridian_orders.csv`

---

## The situation

Meridian Retail is an online retailer operating across Mumbai, Bengaluru and Delhi NCR. It ships five product lines through three fulfilment hubs, and sells to two very different kinds of buyer: individual shoppers and corporate bulk buyers.

The financial year has closed. The leadership team is preparing for a board review and has asked the analytics team for a read on the order book. Four people want four different things, and they have all been given the same file.

- **The CFO** wants a single number for what a typical order is worth, to anchor next year's revenue plan.
- **The COO** wants to know which fulfilment hub is performing best, and what delivery promise the company can safely put on the website.
- **The Head of Category** wants to know which product line is the most volatile to plan inventory against.
- **The Head of Customer Experience** wants to know what the ratings data is really saying.

Every one of those questions can be answered wrongly with a correct calculation. That is what this case study is about.

---

## The dataset

5,000 orders placed between 1 April 2025 and 31 March 2026. One row per order, 18 columns.

| Column | Type | What it holds |
|---|---|---|
| `order_id` | Text | Unique order reference |
| `order_date` | Date | Date the order was placed |
| `customer_id` | Text | Customer reference |
| `customer_segment` | Category | `Retail` or `Corporate` |
| `city` | Category | Delivery city — Mumbai, Bengaluru, Delhi NCR |
| `fulfilment_hub` | Category | Warehouse that shipped the order |
| `channel` | Category | Mobile App, Website, Marketplace |
| `payment_method` | Category | UPI, Credit Card, Debit Card, Cash on Delivery, Net Banking, Wallet |
| `product_category` | Category | Electronics, Apparel, Grocery, Home & Kitchen, Beauty |
| `units` | Whole number | Quantity ordered |
| `unit_price` | Number (INR) | List price of one unit before discount |
| `discount_pct` | Number (%) | Discount applied |
| `order_value` | Number (INR) | Amount billed |
| `gross_margin_pct` | Number (%) | Gross margin on the order |
| `delivery_days` | Number | Days from order placed to delivered |
| `customer_rating` | Ordinal | Post-delivery rating, 1 to 5 |
| `package_weight_kg` | Number | Shipped parcel weight |
| `returned` | Category | `Yes` or `No` |

**Internal consistency:** `order_value = units × unit_price × (1 − discount_pct / 100)`, rounded to two decimals. You can verify this, and you should — checking that a file says what it claims to say is the first step of any analysis.

**Provenance:** this is a synthetic dataset built for teaching, generated with a fixed random seed so every run reproduces the same numbers. Meridian Retail is not a real company. The patterns inside it are engineered to be found.

---

## The ten questions

Each question names the concepts it exercises. Answer in two parts every time: **the number**, and **what you would tell the executive who asked**. A correct number with the wrong recommendation is a failed answer.

---

### Q1. What is a typical order worth?
*Concepts: mean, median, mean–median gap*

The CFO wants one number to anchor the revenue plan.

Compute the mean and the median of `order_value`. Report the gap between them as a percentage. Decide which number the CFO should use for planning, and state what would go wrong if she used the other one. Note that "which one to use" may depend on whether she is planning inventory or planning total revenue.

---

### Q2. Which payment method and channel dominate?
*Concepts: mode, valid statistics by data type*

Find the most common `payment_method` and the most common `channel`.

Then answer a second question that matters more than the first: why can you not compute a mean or a median for these columns? Give the general rule that decides which measures of centre are legal for which type of data. Test the rule against `customer_rating`, which is a third case again.

---

### Q3. What is the blended gross margin?
*Concepts: weighted mean, simple average*

An analyst reports the company's blended gross margin by averaging `gross_margin_pct` across the five product categories.

Compute that simple average. Then compute the revenue-weighted margin, using total `order_value` per category as the weights. Explain the size of the gap, identify which category is responsible for most of it, and state which figure belongs in the board pack.

---

### Q4. Which fulfilment hub performs best?
*Concepts: standard deviation, range, why a centre alone is not a description*

Group by `fulfilment_hub` and compute the count, mean, median and standard deviation of `delivery_days`.

The three hubs have nearly identical mean delivery times. Rank them anyway, and justify the ranking. Then answer the COO's real question: which hub would you route a time-critical order through, and why does the mean fail to identify it?

---

### Q5. Which product line is most volatile?
*Concepts: coefficient of variation, why raw standard deviation cannot compare across scales*

The Head of Category wants to know which product line has the least predictable order sizes.

Restrict to `customer_segment == 'Retail'` first, and explain why that restriction is necessary before you compute anything. Then compute the mean, standard deviation and coefficient of variation of `order_value` for each `product_category`.

Rank the categories by standard deviation, then rank them by CV. Explain why the two rankings can disagree and which one answers the question asked.

---

### Q6. What delivery promise can Meridian put on the website?
*Concepts: percentiles, quartiles, IQR, five-number summary*

Compute the five-number summary of `delivery_days`, plus the 90th and 95th percentiles, overall and per hub.

Marketing wants to advertise "delivered in 4 days". Using the percentiles, state what proportion of orders that promise would break, company-wide and for the worst hub. Then recommend a promise the company can actually keep, and say which percentile you based it on.

---

### Q7. How skewed is the order book, and what should be done about it?
*Concepts: skewness, log transform, mean–median ordering*

Compute the skewness of `order_value`. Classify it against the usual thresholds.

Then apply a natural log transform to `order_value` and compute the skewness again. Report both numbers. Explain what the transform did and did not do — specifically, state whether any data was removed. Finally, explain what a skew of this size means for anyone building a forecast on the mean.

---

### Q8. What are the ratings really saying?
*Concepts: skewness (negative), ordinal data, mean vs median on a bounded scale*

Compute the distribution of `customer_rating` — the count at each level — plus its mean, median and skewness.

The rating skew points the opposite way to the order value skew. Explain what that means in plain terms. Then investigate: cross-tabulate rating against `delivery_days` or `returned` and identify what is producing the low ratings. Recommend what the Head of Customer Experience should do, and state whether "average rating 3.7" is a useful thing to put on a dashboard.

---

### Q9. How often does a delivery go badly wrong?
*Concepts: kurtosis, excess vs raw kurtosis, empirical rule, tail risk*

Compute the skewness and excess kurtosis of `delivery_days`.

The operations planning model assumes delivery times are normally distributed. Test that assumption: count how many orders fall beyond 2, 3 and 4 standard deviations from the mean, and compare each count against what a normal distribution predicts (4.55%, 0.27%, 0.0063%). Express the result as "how many times more often than the model expected".

State what this means for staffing the customer service desk, and explain why standard deviation alone would not have revealed it.

---

### Q10. Is this one business or two?
*Concepts: bimodality, segmentation, outlier detection, z-score vs IQR fence*

Everything above treated the order book as one population. Test that assumption.

1. Split `order_value` by `customer_segment` and compute the full profile — n, mean, median, standard deviation, skewness, excess kurtosis — for each group separately.
2. Compute how many orders sit within ±20% of the company-wide mean order value. Comment on the number.
3. Run outlier detection on the full `order_value` column two ways: a z-score rule flagging anything beyond 3 standard deviations, and the IQR fence at Q1 − 1.5×IQR / Q3 + 1.5×IQR. Report how many orders each method flags.
4. Explain why the two methods disagree so sharply, and which one you would trust here.

Then write the closing recommendation: should Meridian report one set of order statistics to the board, or two? Defend the answer.

---

## What to hand in

For each question:

- the code that produced the number
- the number itself, stated plainly
- two or three sentences of interpretation aimed at the executive who asked

The interpretation is the graded part. Every number in this case study can be produced by one line of pandas. Deciding which number to produce, and knowing what it does and does not license you to say, is the actual skill.

---

## The check to run before you start

```python
import pandas as pd

df = pd.read_csv("meridian_orders.csv", parse_dates=["order_date"])

print(df.shape)                 # expect (5000, 18)
print(df.isna().sum().sum())    # expect 0
print(df.dtypes)

# Verify the file says what it claims:
recomputed = (df["units"] * df["unit_price"] * (1 - df["discount_pct"] / 100)).round(2)
print((recomputed - df["order_value"]).abs().max())   # expect ~0
```

If that last line does not come back near zero, stop and find out why before computing anything else.
