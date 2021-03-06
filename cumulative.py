"""App for calculating the cumulative probability of *r* successes in *n* trials.
Determine the number of trials needed to reach a certain probability threshold
or find the probability at a specified number of trials.

[Streamlit](https://share.streamlit.io/hqn006/streamlit-probability/main/cumulative.py)  
[GitHub](https://github.com/hqn006/streamlit-probability)


## [Definition of Cumulative Probability](https://en.wikipedia.org/wiki/Cumulative_distribution_function)

The cumulative probability of a random variable *X* evaluated at *x* is defined as
the probability that *X* will take a value less than or equal to *x*. The cumulative
distribution function is given by

$$ F_X(x) = P(X \\leq x) $$

The variables *x* and *r* are used interchangeably in this document.


## Calculating Cumulative Probability

This module examines the case where *X* is a binomial distribution with *r* being
the number of desired successes, *p* being the probability of success for each trial,
and *n* ranging from *r* to a maximum input value.

### Probability of "exactly *r* successes"

To calculate the the probability of *x* successes in *n* trials, use the binomial
theorem.

$$ P(X = x) = \\binom{n}{x} p^x (1-p)^{n-x} $$

This module uses the [`math.comb`](https://docs.python.org/3/library/math.html)
function for "n choose k" type calculations.


### Probability of "at most *r* successes"

Sum all probabilities that *X* will take values within a range of "at most *r*
successes" with

$$ P(X \\leq x) = \\sum_{k=0}^{r} P(X = k) $$

It follows that the probability of "less than *r* successes" is

$$ P(X < x) = \\sum_{k=0}^{r-1} P(X = k) $$


### Probability of "at least *r* successes"

Obtain the probability of "greather than *r* successes" using the complemenent
of "at most *r* successes"

$$ P(X > x) = 1 - P(X \\leq x) $$

Then the probability of "at least *r* successes" is the complement of "less than
*r* successes"

$$ P(X \\geq x) = 1 - P(X < x) $$

"""

from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st


def main():
    """Main function. Set up widgets, calculate, plot."""

    st.set_page_config(layout='wide')

    # Set up sidebar input widgets
    with st.sidebar:
        P_des, n_des, p, r, n_max = params()
        complementary, inclusive, out_txt = range_cond()
    
    # Proportions of output screen
    left_column, right_column = st.columns([1,3])

    # Calculations
    probs = Cumulative(r, n_max)
    probs.calc(p, r, complementary, inclusive)
    probs.find_desired(P_des, n_des, complementary)

    # DataFrame
    with left_column:
        df = probs.show_data()
        download_df(df)
    
    # Plot
    with right_column:
        probs.plot_graph(r, out_txt)

    return None


class Cumulative:

    def __init__(self, r, n_max):
        """Initialize Cumulative class containing probability calculations.

        Parameters
        ----------
        r : int
            Number of successes
        n_max : int
            Max number of trials
        """

        self.N = np.arange(r-1, n_max, 1)
        """Array containing numbers of trials ascending"""
        self.P = np.zeros(self.N.shape)
        """Array of cumulative probabilities corresponding to `N`"""
        self.n_found = -1
        """Number of trials closest to desired cumulative probability (see `P_des`)"""
        self.P_closest = 0
        """Cumulative probability closest to desired (see `P_des`)"""
        self.P_found = -1
        """Cumulative probability found at desired number of trials (see `n_des`)"""
        self.n_closest = 0
        """Number of trials closest to desired (equal to `n_des`)"""


    def calc(self, p, r, complementary, inclusive):
        """Send parameters to cached function outside of class to calculate.
        Write the resulting probability array to the class parameter `P`.

        Parameters
        ----------
        p : float
            Probability of one successful event
        r : int
            Number of successes
        complementary : bool
            Specifies cumulative probability or its complement
        inclusive : bool
            Specifies whether edge case is inclusive
        """

        self.P = calc_prob(self.N, p, r, complementary, inclusive)
        
        return None
    

    def find_desired(self, P_des, n_des, complementary):
        """Find closest number of trials that crosses the input probability threshold.
        Find probability at the input number of trials.
    
        Parameters
        ----------
        P_des : float
            Desired cumulative probability if wanting to find number of trials needed
        n_des : int
            Desired number of trials if wanting to find cumulative probability at a point
        complementary : bool
            Specifies cumulative probability or its complement
        """

        i = 0 # index in P array
        for n in self.N:

            # Store point at input desired probability
            if self.n_found <= 0:
                if (
                        (complementary and self.P[i] > P_des) or
                        (not complementary and self.P[i] < P_des)
                    ):
                    self.n_found = n
                    self.P_closest = self.P[i]
            
            # Store point at input number of trials
            if self.P_found <= 0:
                if n == n_des:
                    self.P_found = self.P[i]
                    self.n_closest = n_des
            
            # Only need to retrieve first time threshold is crossed
            if self.n_found > 0 and self.P_found > 0:
                break
            
            i += 1

        return None
    

    def show_data(self):
        """Display whether the desired cumulative probability threshold is crossed
        as "FOUND" and the closest data point. Output total probability DataFrame.
        
        Returns
        -------
        df : pandas.DataFrame
            Dataframe containing `N` and `P` arrays
        """

        # Found point at input desired probability
        if self.n_found > 0:
            st.success("Desired Cumulative Probability **FOUND**")
        else:
            st.warning("Desired Cumulative Probability **NOT FOUND**")

        st.write("Cumulative Probability:", self.P_closest)
        st.write("Number of trials:", self.n_found)

        # Found point at input number of trials
        if self.P_found > 0:
            st.success("Probability at Number of Trials **FOUND**")
        else:
            st.warning("Probability at Number of Trials **NOT FOUND**")
        
        st.write("Number of trials:", self.n_closest)
        st.write("Cumulative Probability:", self.P_found)

        df = pd.DataFrame({'N': self.N, 'P': self.P})
        st.dataframe(df, None, 700)

        return df
    

    def plot_graph(self, r, out_txt):
        """Plot cumulative probability distribution using Matplotlib.
        
        Parameters
        ----------
        r : int
            Number of successes
        out_txt : str
            Output text description of range conditions
        """

        fig, ax = plt.subplots()

        ax.set_title(f"Cumulative Probability of {out_txt} {r} Successes in n Trials")
        ax.set_xlabel("Number of Trials")
        ax.set_ylabel("Cumulative Probability")
        ax.plot(self.N, self.P)

        # Point at input desired probability
        ax.hlines(self.P_closest, 0, self.N[-1], 'r', 'dashed')
        ax.text(self.n_found, self.P_closest + 0.01,
                f'({self.n_found}, {self.P_closest:.3f})')

        # Point at input number of trials
        ax.plot(self.n_closest, self.P_found, 'r+')
        ax.text(self.n_closest, self.P_found + 0.01,
                f'({self.n_closest}, {self.P_found:.3f})')

        ax.set_ylim(0,1)

        st.pyplot(fig)

        return None


@st.experimental_memo
def calc_prob(N, p, r, complementary, inclusive):
    """Calculate cumulative probabilities.

    Parameters
    ----------
    N : ndarray
        Array containing numbers of trials ascending
    p : float
        Probability of one successful event
    r : int
        Number of successes
    complementary : bool
        Specifies cumulative probability or its complement
    inclusive : bool
        Specifies whether edge case is inclusive

    Returns
    -------
    P : ndarray
        Array of cumulative probabilities corresponding to `N`
    """

    P = np.zeros(N.shape)

    i = 0 # index in P array
    for n in N:

        # Must consider that 1 - P will be executed later
        if complementary: # At Least or Greater Than
            r_include = (r-1 if inclusive else r)
        else: # At most or Less Than
            r_include = (r if inclusive else r-1)

        # Sum up all exactly x successes
        sum_exactly = 0
        for k in range(r_include, -1, -1):
            exactly = comb(n,k) * p**k * (1-p)**(n-k)
            sum_exactly += exactly

        # Probability array
        if complementary:
            P[i] = 1 - sum_exactly
        else:
            P[i] = sum_exactly

        i += 1

    return P


def params():
    """Widgets for inputting parameters.

    Returns
    -------
    P_des : float
        Desired cumulative probability if wanting to find number of trials needed
    n_des : int
        Desired number of trials if wanting to find cumulative probability at a point
    p : float
        Probability of one successful event
    r : int
        Number of successes
    n_max : int
        Max number of trials
    """

    st.header("Cumulative Probability Calculator")

    P_des = st.number_input(
        "Desired Cumulative Probability",
        0.0, 1.0, 0.5,
        step=0.1, format="%.3f"
        )
    n_des = st.number_input(
        "Find Probability at Number of Trials",
        value=-1,
        step=1
        )
    
    "---"

    p = st.number_input(
        "Probability of event",
        0.0, 1.0, 0.01,
        step=0.001, format="%.8f"
        )
    r = st.number_input("Number of successes", 1)
    n_max = st.number_input("Max number of trials", 1, value=500, step=100) + 1

    return P_des, n_des, p, r, n_max


def range_cond():
    """Widgets for range conditions and edge case.

    Returns
    -------
    complementary : bool
        Specifies cumulative probability or its complement
    inclusive : bool
        Specifies whether edge case is inclusive
    out_txt : str
        Output text description of range conditions
    """

    "Range Conditions"
    # Default: At Most
    complementary = st.checkbox("Complementary", False)
    inclusive = st.checkbox("Inclusive", True)

    if complementary:
        out_txt = ("At Least" if inclusive else "Greater Than")
    else:
        out_txt = ("At Most" if inclusive else "Less Than")

    st.info("Description: " + out_txt + " *r* Successes")

    return complementary, inclusive, out_txt


def download_df(df):
    """Button to download DataFrame as CSV.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing `N` and `P` arrays
    """

    @st.experimental_memo
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)
    st.download_button("Download as CSV", csv, 'cumulative_df.csv', 'text/csv')

    return None


if __name__ == '__main__':
    main()


# EOF
