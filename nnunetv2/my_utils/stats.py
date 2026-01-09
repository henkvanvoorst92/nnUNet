import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def string_p_value(p_value):
    if p_value < 0.00001:
        p_value_str = "p<0.00001"
    elif p_value < 0.001:
        p_value_str = "p<0.001"
    elif p_value < 0.01:
        p_value_str = "p<0.01"
    elif p_value < 0.05:
        p_value_str = "p<0.05"
    else:
        p_value_str = f"p={p_value:.2g}"
    return p_value_str

def asterix_p_value(p_value: float) -> str:
    if p_value < 0.00001:
        return "****" #"p<0.00001"
    elif p_value < 0.001:
        return "***" #"p<0.001"
    elif p_value < 0.01:
        return "**" #"p<0.01"
    elif p_value < 0.05:
        return '*' #"p<0.05"
    else:
        return '' #f"p={p_value:.2g}"



def fit_diff_time_mixedlm(df, id_col="ID", time_col="time", diff_col="diff", reml=True):
    d = df[[id_col, time_col, diff_col]].dropna().copy()
    d[id_col] = d[id_col].astype("category")
    d[time_col] = d[time_col].astype("category")  # key: no linearity assumed

    m = smf.mixedlm(
        f"{diff_col} ~ C({time_col})",
        data=d,
        groups=d[id_col],
    )
    r = m.fit(reml=reml)
    return r

def get_average_over_time(result, time_col="time"):
    """
    Computes the unweighted average of the fixed-effect means over timepoints.
    Assumes model: diff ~ C(time)
    """

    # Fixed effects only
    fe = result.fe_params
    # Align covariance strictly to fixed effects
    cov = result.cov_params().loc[fe.index, fe.index]
    # Identify time dummy terms
    time_terms = [k for k in fe.index if k.startswith(f"C({time_col})")]
    # Number of timepoints
    K = 1 + len(time_terms)
    # Build contrast vector
    w = pd.Series(0.0, index=fe.index)
    # Intercept contributes to all timepoints
    w["Intercept"] = 1.0
    # Each dummy contributes to exactly one timepoint
    for term in time_terms:
        w[term] = 1.0 / K

    # Estimate
    est = float(w @ fe)
    # Standard error (now dimensions match)
    se = float(np.sqrt(w.values @ cov.values @ w.values))
    z = stats.norm.ppf(0.975)
    ci_low, ci_hi = est - z * se, est + z * se
    #ci = (est - z * se, est + z * se)
    p = 2 * (1 - stats.norm.cdf(abs(est / se)))
    return {
        "mean_diff": est,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_hi,
        "p_value": p,
        "n_timepoints": K,
    }

def mixedlm_on_diff(
    df: pd.DataFrame,
    diff_col: str,
    group_col: str,
    reml: bool = True,
):
    """
    Fit an intercept-only linear mixed-effects model on precomputed differences.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe containing differences.
    diff_col : str
        Column with paired differences (e.g. modelB - modelA).
    group_col : str
        Grouping column for random effects (e.g. patient_id).
    reml : bool
        Use REML estimation.

    Returns
    -------
    results : dict
        Mean difference, 95% CI, p-value, and fitted model.
    """

    df = df[[group_col, diff_col]].dropna().copy()

    model = smf.mixedlm(
        f"{diff_col} ~ 1",
        data=df,
        groups=df[group_col],
    )

    res = model.fit(reml=reml)

    est = res.params["Intercept"]
    ci_low, ci_high = res.conf_int().loc["Intercept"]
    pval = res.pvalues["Intercept"]

    return {
        "mean_diff": est,
        "ci_95": (ci_low, ci_high),
        "p_value": pval,
        "n_groups": df[group_col].nunique(),
        "n_missing": df[diff_col].isna().sum(),
        "n_obs": len(df),
        "model": res,
    }
