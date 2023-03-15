from __future__ import print_function
import numpy as np
import itertools as itr
import scipy.stats as ss
import scipy.special as spec
import seaborn as sns

# import statsmodels.tools.tools as stt
import statsmodels.distributions.empirical_distribution as smd
import matplotlib.pyplot as plt
import collections as cls
import pandas as pd
import joblib as job
import psutil as ps
import os

from function_finance_metrics import *

PBO = cls.namedtuple(
    "PBO",
    [
        "pbo",
        "prob_oos_loss",
        "linear_model",
        "stochastic",
        "Cs",
        "J",
        "J_bar",
        "R",
        "R_bar",
        "R_rank",
        "R_bar_rank",
        "rn",
        "rn_bar",
        "w_bar",
        "logits",
        "R_n_star",
        "R_bar_n_star",
    ],
)

PBOCore = cls.namedtuple(
    "PBOCore",
    [
        "J",
        "J_bar",
        "R",
        "R_bar",
        "R_rank",
        "R_bar_rank",
        "rn",
        "rn_bar",
        "w_bar",
        "logits",
    ],
)


def pbo(
        M,
        S,
        metric_func,
        name_exp,
        threshold,
        n_jobs=1,
        verbose=False,
        plot=False,
        hist=False,
):
    """
    Based on http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

    Features:
    * training and test sets are of equal size, providing comparable accuracy
    to both IS and OOS Sharpe ratios.
    * CSCV is symmetric, decline in performance can only result from
    overfitting, not arbitrary discrepancies between the training and test
    sets.
    * CSCV respects the time-dependence and other season-dependent features
    present in the data.
    * Results are deterministic, can be replicated.
    * Dispersion in the distribution of logits conveys relevant info regarding
    the robustness of the strategy selection process.
    * Model-free, non-parametric. Logits distribution resembles the cumulative
    Normal distribution if w_bar are close to uniform distribution (i.e. the
    backtest appears to be information-less). Therefore, for good backtesting,
    the distribution of logits will be centered in a significantly positive
    value, and its tail will marginally cover the region of negative logit
    values.

    Limitations:
    * CSCV is symmetric, for some strategies, K-fold CV might be better.
    * Not suitable for time series with strong auto-correlation, especially
    when S is large.
    * Assumes all the sample statistics carry the same weight.
    * Entirely possible that all the N strategy configs have high but similar
    Sharpe ratios. Therefore, PBO may appear high, however, 'overfitting' here
    is among many 'skilful' strategies.

    Parameters:

    M:
        returns data, numpy or dataframe format.
    S:
        chuncks to devided M into, must be even number. Paper suggests setting
        S = 16. See paper for details of choice of S.
    metric_func:
        evaluation function for returns data
    threshold:
        used as prob. of OOS Loss calculation cutoff. For Sharpe ratio,
        this should be 0 to indicate probabilty of loss.
    n_jobs:
        if greater than 1 then enable parallel mode
    hist:
        Default False, whether to plot histogram for rank of logits.
        Some problems exist when S >= 10. Need to look at why numpy /
        matplotlib does it.

    Returns:
    PBO result in namedtuple, instance of PBO.
    """
    if S % 2 == 1:
        raise ValueError(
            "S must be an even integer, {:.1f} was given".format(S)
        )

    n_jobs = int(n_jobs)
    if n_jobs < 0:
        n_jobs = max(1, ps.cpu_count(logical=False))

    if isinstance(M, pd.DataFrame):
        # conver to numpy values
        if verbose:
            print("Convert from DataFrame to numpy array.")
        M = M.values

    # Paper suggests T should be 2x the no. of observations used by investor
    # to choose a model config, due to the fact that CSCV compares combinations
    # of T/2 observations with their complements.
    T, N = M.shape
    residual = T % S
    if residual != 0:
        M = M[residual:]
        T, N = M.shape

    sub_T = T // S

    if verbose:
        print("Total sample size: {:,d}, chunck size: {:,d}".format(T, sub_T))

    # generate subsets, each of length sub_T
    Ms = []
    Ms_values = []
    for i in range(S):
        start, end = i * sub_T, (i + 1) * sub_T
        Ms.append((i, M[start:end, :]))
        Ms_values.append(M[start:end, :])
    Ms_values = np.array(Ms_values)

    if verbose:
        print("No. of Chuncks: {:,d}".format(len(Ms)))

    # generate combinations
    Cs = [x for x in itr.combinations(Ms, S // 2)]
    if verbose:
        print("No. of combinations = {:,d}".format(len(Cs)))

    # Ms_index used to find J_bar (complementary OOS part)
    Ms_index = set([x for x in range(len(Ms))])

    # create J and J_bar
    if n_jobs < 2:
        J = []
        J_bar = []
        for i in range(len(Cs)):
            # make sure chucks are concatenated in their original order
            order = [x for x, _ in Cs[i]]
            sort_ind = np.argsort(order)

            Cs_values = np.array([v for _, v in Cs[i]])
            # if verbose:
            #     print('Cs index = {}, '.format(order), end='')
            joined = np.concatenate(Cs_values[sort_ind, :])
            J.append(joined)

            # find Cs_bar
            Cs_bar_index = list(sorted(Ms_index - set(order)))
            # if verbose:
            # print('Cs_bar_index = {}'.format(Cs_bar_index))
            J_bar.append(np.concatenate(Ms_values[Cs_bar_index, :]))

        # compute matrices for J and J_bar, e.g. Sharpe ratio
        R = [metric_func(j) for j in J]
        R_bar = [metric_func(j) for j in J_bar]

        # print('----------  big  r (performance) ---------')
        # print(R)
        # print(R_bar)

        # compute ranks of metrics
        R_rank = [ss.rankdata(x) for x in R]
        R_bar_rank = [ss.rankdata(x) for x in R_bar]

        # print('----------  big  r (ranked) ---------')
        # print(R_rank)
        # print(R_bar_rank)

        # find highest metric, rn contains the index position of max value
        # in each set of R (IS)
        rn = [np.argmax(r) for r in R_rank]

        # print('-------------------- small r --------------------')
        # print(rn)

        # use above index to find R_bar (OOS) in same index position
        # i.e. the same config / setting

        rn_bar = [R_bar_rank[i][rn[i]] for i in range(len(R_bar_rank))]
        #
        # print(rn_bar)

        # formula in paper used N+1 as the denominator for w_bar. For good reason
        # to avoid 1.0 in w_bar which leads to inf in logits. Intuitively, just
        # because all of the samples have outperformed one cannot be 100% sure.

        w_bar = [float(r) / (N + 1) for r in rn_bar]

        # logit(.5) gives 0 so if w_bar value is equal to median logits is 0
        logits = [spec.logit(w) for w in w_bar]

    else:
        # use joblib for parallel calc
        # print('Run in parallel mode.')
        cores = job.Parallel(n_jobs=n_jobs)(
            job.delayed(pbo_core_calc)(
                Cs_x, Ms, Ms_values, Ms_index, metric_func, verbose
            )
            for Cs_x in Cs
        )
        # core_df = pd.DataFrame(cores, columns=PBOCore._fields)
        # convert to values needed.
        # # core_df = pd.DataFrame.from_records(cores)

        # J = core_df.J.values
        # J_bar = core_df.J_bar.values
        # R = core_df.R.values
        # R_bar = core_df.R_bar.values
        # R_rank = core_df.R_rank.values
        # R_bar_rank = core_df.R_bar_rank.values
        # rn = core_df.rn.values
        # rn_bar = core_df.rn_bar.values
        # w_bar = core_df.w_bar.values
        # logits = core_df.logits.values

        J = [c.J for c in cores]
        J_bar = [c.J_bar for c in cores]
        R = [c.R for c in cores]
        R_bar = [c.R_bar for c in cores]
        R_rank = [c.R_rank for c in cores]
        R_bar_rank = [c.R_bar_rank for c in cores]
        rn = [c.rn for c in cores]
        rn_bar = [c.rn_bar for c in cores]
        w_bar = [c.w_bar for c in cores]
        logits = [c.logits  for c in cores]

    ############ exited if / else ##########3

    # prob of overfitting
    phi = np.array([1.0 if lam <= 0 else 0.0 for lam in logits]) / len(Cs)
    pbo_test = np.sum(phi)

    # print("Test")
    # for i in range(len(R)):
    #     print('-----------')
    #     print(i)
    #     print(rn[i])
    #     print(R[i])
    #     print(R[i][0][rn[i]])
    # print('test works')

    R_n_star = np.array([R[i][rn[i]] for i in range(len(R))])
    R_bar_n_star = np.array([R_bar[i][rn[i]] for i in range(len(R_bar))])
    lm = ss.linregress(x=R_n_star, y=R_bar_n_star)

    prob_oos_loss = np.sum(
        [1.0 if r < threshold else 0.0 for r in R_bar_n_star]
    ) / len(R_bar_n_star)

    # Stochastic dominance
    y = np.linspace(
        min(R_bar_n_star), max(R_bar_n_star), endpoint=True, num=1000
    )
    R_bar_n_star_cdf = smd.ECDF(R_bar_n_star)
    optimized = R_bar_n_star_cdf(y)

    R_bar_cdf = smd.ECDF(np.concatenate(R_bar))
    non_optimized = R_bar_cdf(y)

    dom_df = pd.DataFrame(
        dict(optimized_OOS=optimized, non_optimized_OOS=non_optimized)
    )
    dom_df.index = y
    # visually, non_optimized curve above optimized curve indicates good
    # backtest with low overfitting.
    dom_df["SD2"] = dom_df.non_optimized_OOS - dom_df.optimized_OOS

    result = PBO(
        pbo_test,
        prob_oos_loss,
        lm,
        dom_df,
        Cs,
        J,
        J_bar,
        R,
        R_bar,
        R_rank,
        R_bar_rank,
        rn,
        rn_bar,
        w_bar,
        logits,
        R_n_star,
        R_bar_n_star,
    )

    if plot:
        plot_pbo(result, name_exp, hist=hist)

    return result


def pbo_core_calc(Cs, Ms, Ms_values, Ms_index, metric_func, verbose=False):
    # make sure chucks are concatenated in their original order
    order = [x for x, _ in Cs]
    sort_ind = np.argsort(order)

    Cs_values = np.array([v for _, v in Cs])
    if verbose:
        print("Cs index = {}, ".format(order), end="")
    J_x = np.concatenate(Cs_values[sort_ind, :])

    # find Cs_bar
    Cs_bar_index = list(sorted(Ms_index - set(order)))
    if verbose:
        print("Cs_bar_index = {}".format(Cs_bar_index))
    J_bar_x = np.concatenate(Ms_values[Cs_bar_index, :])

    R_x = metric_func(J_x)
    R_bar_x = metric_func(J_bar_x)

    R_rank_x = ss.rankdata(R_x)
    R_bar_rank_x = ss.rankdata(R_bar_x)

    rn_x = np.argmax(R_rank_x)
    rn_bar_x = R_bar_rank_x[rn_x]

    # formula in paper used N+1 as the denominator for w_bar. For good reason
    # to avoid 1.0 in w_bar which leads to inf in logits. Intuitively, just
    # because all of the sample.get_figure()s have outperformed one cannot be 100% sure.
    w_bar_x = float(rn_bar_x) / (len(R_bar_rank_x) + 1)  # / (N+1)
    logit_x = spec.logit(w_bar_x)

    core = PBOCore(
        J_x,
        J_bar_x,
        R_x,
        R_bar_x,
        R_rank_x,
        R_bar_rank_x,
        rn_x,
        rn_bar_x,
        w_bar_x,
        logit_x,
    )

    return core


def plot_pbo(pbo_result, name_exp, hist=False):
    if not os.path.isdir('./train_results/' + name_exp + '/images'):
        os.mkdir('./train_results/' + name_exp + '/images')

    lm = pbo_result.linear_model
    sns.set(rc={'figure.figsize': (10, 6)})

    # Plot 1
    r2 = lm.rvalue ** 2
    # adj_r2 = r2 - (1 - r2) / (len(pbo_result.R_n_star) - 2.0)
    line_label = (
            "slope: {:.4f}\n".format(lm.slope)
            + "p: {:.4E}\n".format(lm.pvalue)
            + "$R^2$: {:.4f}\n".format(r2)
            + "Prob. OOS < hold: {:.1%}".format(pbo_result.prob_oos_loss)
    )

    sns.set(font_scale=1.7)
    plot1_perfdeg = sns.regplot(
        x="SR_IS",
        y="SR_OOS",
        # sns.lmplot(x='SR_IS', y='SR_OOS',
        data=pd.DataFrame(
            dict(SR_IS=pbo_result.R_n_star, SR_OOS=pbo_result.R_bar_n_star)
        ),
        scatter_kws={"alpha": 0.3, "color": "g"},
        line_kws={
            "alpha": 0.8,
            "label": line_label,
            "linewidth": 1.0,
            "color": "r",
        }
    )
    plot1_perfdeg.set_title("Performance Degradation, IS vs. OOS")
    plot1_perfdeg.legend(loc="best")

    # Plot 2
    # TODO hist is turned off at the moment. Error occurs when S is set to
    # a relatively large number, such as 16.
    plot2_logitsfreq = sns.distplot(
        pbo_result.logits,
        rug=True,
        # bins=10,  # default might be more useful
        rug_kws={"color": "r", "alpha": 0.5},
        kde_kws={"color": "k", "lw": 2.0, "label": "KDE"},
        hist=hist,
        hist_kws={
            "histtype": "step",
            "linewidth": 2.0,
            "alpha": 0.7,
            "color": "g",
        },
    )
    plot2_logitsfreq.axvline(0, c="r", ls="--")
    #plot2_logitsfreq.set_title("Frequency Dist. of Rank Logits, PBO = " + str(0.8) + '%')
    plot2_logitsfreq.set_title("Frequency Dist. of Rank Logits, PBO = " + str(round(pbo_result[0] * 100, 2)) + '%')
    plot2_logitsfreq.set_xlabel("Logits")
    plot2_logitsfreq.set_ylabel("Frequency")


    # Plot 3
    plot3_stochdom = pbo_result.stochastic.plot(secondary_y="SD2")
    plot3_stochdom.right_ax.axhline(0, c="r")
    plot3_stochdom.set_title("Stochastic Dominance")
    plot3_stochdom.set_ylabel("Frequency")
    plot3_stochdom.set_xlabel("SR Optimized vs. Non-Optimized")
    plot3_stochdom.right_ax.set_ylabel("2nd Order Stoch. Dominance")


def psr_from_returns(returns, risk_free=0, target_sharpe=0):
    """
    PSR from return series.

    Parameters:
        returns:
            return series
        risk_free:
            risk free or benchmark rate for sharpe ratio calculation,
            default 0.
        target_sharpe:
            minimum sharpe ratio

    Returns:
        PSR probabilities.
    """
    T = len(returns)
    sharpe = sharpe_iid(returns, bench=risk_free, factor=1)
    skew = returns.skew()
    kurtosis = returns.kurtosis() + 3

    return psr(
        sharpe=sharpe,
        T=T,
        skew=skew,
        kurtosis=kurtosis,
        target_sharpe=target_sharpe,
    )


def psr(sharpe, T, skew, kurtosis, target_sharpe=0):
    """
    Probabilistic Sharpe Ratio.

    Parameters:
        sharpe:
            observed sharpe ratio, in same frequency as T.
        T:
            no. of observations, should match return / sharpe sampling period.
        skew:
            sharpe ratio skew
        kurtosis:
            sharpe ratio kurtosis
        target_sharpe:
            target sharpe ratio

    Returns:
        Cumulative probabilities for observed sharpe ratios under standard
        Normal distribution.
    """
    value = (
            (sharpe - target_sharpe)
            * np.sqrt(T - 1)
            / np.sqrt(1.0 - skew * sharpe + sharpe ** 2 * (kurtosis - 1) / 4.0)
    )
    # print(value)
    psr = ss.norm.cdf(value, 0, 1)
    return psr


def dsr(test_sharpe, sharpe_std, N, T, skew, kurtosis):
    """
    Deflated Sharpe Ratio statistic. DSR = PSR(SR_0).
    See paper for definition of SR_0. http://ssrn.com/abstract=2460551

    Parameters:
        test_sharpe :
            reported sharpe, to be tested.
        sharpe_std :
            standard deviation of sharpe ratios from N trials / configurations
        N :
            number of backtest configurations
        T :
            number of observations
        skew :
            skew of returns
        kurtosis :
            kurtosis of returns

    Returns:
        DSR statistic
    """
    # sharpe_std = np.std(sharpe_n, ddof=1)
    target_sharpe = sharpe_std * expected_max(N)

    dsr_stat = psr(test_sharpe, T, skew, kurtosis, target_sharpe)

    return dsr_stat


def dsr_from_returns(test_sharpe, returns_df, risk_free=0):
    """
    Calculate DSR based on a set of given returns_df.

    Parameters:
        test_sharpe :
            Reported sharpe, to be tested.
        returns_df :
            Log return series
        risk_free :
            Risk free return, default 0.
    Returns:
        DSR statistic
    """
    T, N = returns_df.shape
    sharpe = sharpe_iid(returns_df, bench=risk_free, factor=1)
    sharpe_std = np.std(sharpe, ddof=1)
    skew = returns_df.skew()
    kurtosis = returns_df.kurtosis() + 3

    out = dsr(
        test_sharpe,
        sharpe_std=sharpe_std,
        N=N,
        T=T,
        skew=skew,
        kurtosis=kurtosis,
    )

    return out


# def sharpe_iid(df, bench=0, factor=np.sqrt(255)):
#     excess = df - bench
#     if isinstance(df, pd.DataFrame):
#         # return factor * (df.mean() - bench) / df.std(ddof=1)
#         return factor * excess.mean() / excess.std(ddof=1)
#     else:
#         # numpy way
#         return np.mean(excess, axis=0) / np.std(excess,
#                                                 axis=0, ddof=1) * factor


def minTRL(sharpe, skew, kurtosis, target_sharpe=0, prob=0.95):
    """
    Minimum Track Record Length.

    Parameters:
        sharpe :
            observed sharpe ratio, in same frequency as observations.
        skew :
            sharpe ratio skew
        kurtosis :
            sharpe ratio kurtosis
        target_sharpe :
            target sharpe ratio
        prob :
            minimum probability for estimating TRL.

    Returns:
        minTRL, in terms of number of observations.
    """
    min_track = (
            1
            + (1 - skew * sharpe + sharpe ** 2 * (kurtosis - 1) / 4.0)
            * (ss.norm.ppf(prob) / (sharpe - target_sharpe)) ** 2
    )
    return min_track


def expected_max(N):
    """
    Expected maximum of IID random variance X_n ~ Z, n = 1,...,N,
    where Z is the CDF of the standard Normal distribution,
    E[MAX_n] = E[max{x_n}]. Computed for a large N.

    """
    if N < 5:
        raise AssertionError("Condition N >> 1 not satisfied.")
    return (1 - np.euler_gamma) * ss.norm.ppf(
        1 - 1.0 / N
    ) + np.euler_gamma * ss.norm.ppf(1 - np.exp(-1) / N)


def minBTL(N, sharpe_IS):
    """
    Minimum backtest length. minBTL should be considered a necessary,
    non-sufficient condition to avoid overfitting. See PBO for a more precise
    measure of backtest overfitting.

    Paramters:
        N :
            number of backtest configurations
        sharpe_IS :
            In-Sample observed Sharpe ratio

    Returns:
        btl :
            minimum back test length
        upper_bound :
            upper bound for minBTL
    """
    exp_max = expected_max(N)

    btl = (exp_max / sharpe_IS) ** 2

    upper_bound = 2 * np.log(N) / sharpe_IS ** 2

    return (btl, upper_bound)
