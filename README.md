Apps for probability on Streamlit.io

## cumulative.py

App for calculating the cumulative probability of *r* successes in *n* trials.
Determine the number of trials needed to reach a certain probability threshold.

[Streamlit.io - Cumulative Probability Calculator](https://share.streamlit.io/hqn006/streamlit-probability/main/cumulative.py)

## [Definition of Cumulative Probability](https://en.wikipedia.org/wiki/Cumulative_distribution_function)

The cumulative probability of a random variable *X* evaluated at *x* is defined as
the probability that *X* will take a value less than or equal to *x*. The cumulative
distribution function is given by

$$ F_X(x) = P(X \\leq x) $$

The variables *x* and *r* are used interchangeably in this document.


## Calculating Cumulative Probability

### Probability of "exactly *r* successes"

To calculate the the probability of *x* successes in *n* trials, use the binomial
theorem.

$$ P(X = x) = \\binom{n}{x} p^x (1-p)^{n-x} $$

This module uses the [`math.comb`](https://docs.python.org/3/library/math.html)
function for "n choose k" type calculations.


### Probability of "at most *r* successes"

Sum all probabilities that *X* will take values within a range of "at most *r*
successes" with

$$ P(X \\leq x) = P(X = 0) + P(X = 1) + \\cdots + P(X = x) $$

It follows that the probability of "less than *r* successes" is

$$ P(X < x) = P(X = 0) + P(X = 1) + \\cdots + P(X = x-1) $$


### Probability of "at least *r* successes"

Obtain the probability of "greather than *r* successes" using the complemenent
of "at most *r* successes"

$$ P(X > x) = 1 - P(X \\leq x) $$

Then the probability of "at least *r* successes" is the complement of "less than
*r* successes"

$$ P(X \\geq x) = 1 - P(X < x) $$
