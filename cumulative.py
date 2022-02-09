"""
App for calculating the cumulative probability of `r` successes in `n` trials.

Huy Nguyen
hqn006@ucsd.edu

"""


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st


def main( ):
    """Main function."""

    st.set_page_config(layout='wide')

    # Sidebar input widgets
    with st.sidebar:
        P_des, p, r, n_max = params()
        complementary, inclusive, out_txt = range_cond()
    
    # Output screen proportions
    left_column, right_column = st.columns([1,3])

    # Calculations
    N, P, n_found, P_found = calc_prob(P_des, p, r, n_max, complementary, inclusive)

    # DataFrane
    with left_column:
        df = output_data(N, P, n_found, P_found)
        download_df( df )
    
    # Plot
    with right_column:
        plot_graph(N, P, n_found, P_found, r, out_txt)

    return None


def params( ):
    """Widgets for inputting parameters.

    Returns
    -------
    P_des : float
        Desired cumulative probability
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
    p = st.number_input(
        "Probability of event",
        0.0, 1.0, 0.01,
        step=0.001, format="%.3f"
        )
    r = st.number_input("Number of successes", 1)
    n_max = st.number_input("Max number of trials", 1, value=500, step=100)

    return P_des, p, r, n_max


def range_cond( ):
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

    "Output text: ", out_txt, " *r* Successes"

    return complementary, inclusive, out_txt


def calc_prob( P_des, p, r, n_max, complementary, inclusive ):
    """Calculate cumulative probabilities.

    Parameters
    ----------
    P_des : float
        Desired cumulative probability
    p : float
        Probability of one successful event
    r : int
        Number of successes
    n_max : int
        Max number of trials
    complementary : bool
        Specifies cumulative probability or its complement
    inclusive : bool
        Specifies whether edge case is inclusive

    Returns
    -------
    N : ndarray
        Array of number of trials ascending
    P : ndarray
        Array of cumulative probabilities corresponding to `N`
    n_found : int
        Number of trials closest to desired cumulative probability
    P_found : float
        Cumulative probability closest to desired
    """

    r_minus = r - 1 # Used depending on edge case

    N = np.arange(r_minus, n_max, 1)
    P = np.zeros(N.shape)
    n_found = -1
    P_found = 0
    i = 0 # index in P array
    for n in N:

        # Must consider that 1 - P will be executed later
        if complementary: # At Least or Greater Than
            r_inclusive = (r_minus if inclusive else r)
        else: # At most or Less Than
            r_inclusive = (r if inclusive else r_minus)

        # Sum up all exactly x successes
        sum_exactly = 0
        for k in range(r_inclusive, -1, -1):
            exactly = math.comb(n,k) * p**k * (1-p)**(n-k)
            sum_exactly += exactly

        # Probability array
        if complementary:
            P[i] = 1 - sum_exactly
        else:
            P[i] = sum_exactly

        # Store found values
        if 1 - sum_exactly > P_des and n_found <= 0:
            n_found = n
            P_found = P[i]

        i += 1
    
    return N, P, n_found, P_found


def output_data( N, P, n_found, P_found ):
    """Output found cumulative probability and Pandas DataFrame.
    
    Parameters
    ----------
    N : ndarray
        Array of number of trials ascending
    P : ndarray
        Array of cumulative probabilities corresponding to `N`
    n_found : int
        Number of trials closest to desired cumulative probability
    P_found : float
        Cumulative probability closest to desired

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing `N` and `P` arrays
    """

    "Desired Cumulative Probability", "**FOUND**" if n_found > 0 else "**NOT FOUND**"
    "Number of trials:", n_found
    "Cumulative Probability:", P_found

    df = pd.DataFrame({'N': N, 'P': P})
    st.dataframe(df, None, 1000)

    return df


def download_df( df ):
    """Button to download DataFrame as CSV.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing `N` and `P` arrays
    """

    csv = df.to_csv().encode('utf-8')
    st.download_button("Download as CSV", csv, 'cumulative_df.csv', 'text/csv')

    return None


def plot_graph( N, P, n_found, P_found, r, out_txt ):
    """Plot graph using Matplotlib.
    
    Parameters
    ----------
    N : ndarray
        Array of number of trials ascending
    P : ndarray
        Array of cumulative probabilities corresponding to `N`
    n_found : int
        Number of trials closest to desired cumulative probability
    P_found : float
        Cumulative probability closest to desired
    r : int
        Number of successes
    out_txt : str
        Output text description of range conditions
    """
    fig, ax = plt.subplots()

    ax.set_title(f"Cumulative Probability of {out_txt} {r} Successes in n Trials")
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Cumulative Probability")
    ax.plot(N, P)
    ax.plot(n_found, P_found, 'ro')
    ax.text(n_found, P_found, '({}, {:.3f})'.format(n_found, P_found))
    ax.set_ylim(0,1)

    st.pyplot(fig)

    return None


if __name__ == '__main__':
    main()


# EOF
