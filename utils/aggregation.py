import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import clean_numeric, minmax_normalize

def generic_aggregate_engineered(
    engineered_df,
    score_prefix,
    variables,
    risk_direction=None,
    impute_strategy="median",
    keep_inputs=True,
    keep_zscores=True
):
    risk_direction = risk_direction or {}

    available = [v for v in variables if v in engineered_df.columns]
    if len(available) == 0:
        raise ValueError(f"No aggregation variables found for {score_prefix}")

    out = pd.DataFrame(index=engineered_df.index)
    X = pd.DataFrame(index=engineered_df.index)

    for col in available:
        s = pd.to_numeric(engineered_df[col], errors="coerce")

        if impute_strategy == "median":
            s = s.fillna(s.median())
        elif impute_strategy == "mean":
            s = s.fillna(s.mean())
        elif impute_strategy == "mode":
            mode_val = s.mode(dropna=True)
            s = s.fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)
        elif impute_strategy == "zero":
            s = s.fillna(0)
        else:
            raise ValueError("impute_strategy must be median, mean, mode, or zero")

        direction = risk_direction.get(col, "positive")

        if direction == "negative":
            s = -s
        elif direction != "positive":
            raise ValueError(f"Invalid risk direction for {col}")

        X[col] = s

    if keep_inputs:
        out = pd.concat([out, X.add_suffix("_input")], axis=1)

    Z = pd.DataFrame(
        StandardScaler().fit_transform(X),
        index=X.index,
        columns=[f"{c}_z" for c in X.columns]
    )

    if keep_zscores:
        out = pd.concat([out, Z], axis=1)

    raw_score = Z.mean(axis=1)

    out[f"{score_prefix}Score"] = raw_score
    out[f"{score_prefix}Score_Norm"] = minmax_normalize(raw_score)

    out[f"{score_prefix}Category"] = pd.cut(
        out[f"{score_prefix}Score_Norm"],
        bins=[-float("inf"), 0.25, 0.50, 0.75, float("inf")],
        labels=["low", "mild", "moderate", "high"]
    )

    return out.copy()