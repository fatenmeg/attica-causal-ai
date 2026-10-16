"""
domain_analysis.py

Exploratory dependency inspection for biomedical domains.
"""

"""
domain_analysis.py

Exploratory dependency inspection for biomedical domains.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.covariance import GraphicalLassoCV


def inspect_domain_dependencies(
    df,
    cols,
    domain_name,
    pearson_threshold=0.30,
):
    """
    Inspect the internal dependency structure of one domain.

    The function reports:
    - missingness,
    - Pearson correlations,
    - cross-validated Graphical Lasso regularisation,
    - precision matrix,
    - partial correlations,
    - Graphical-Lasso-selected sparse dependencies.

    Graphical Lasso sparsity is determined automatically by
    GraphicalLassoCV. No additional partial-correlation threshold
    is applied.
    """

    # ---------------------------------------------------------
    # 1. Validate requested columns
    # ---------------------------------------------------------

    cols = [c for c in cols if c in df.columns]

    print("=" * 80)
    print(f"DOMAIN INSPECTION: {domain_name}")
    print("=" * 80)

    print(f"Variables found: {len(cols)}")
    print(cols)

    if len(cols) < 2:
        print("Not enough variables for dependency inspection.")
        return None

    X = df[cols].copy()

    # ---------------------------------------------------------
    # 2. Missingness
    # ---------------------------------------------------------

    missing = (
        X.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nMissingness (%):")
    print(missing.to_frame("Missing %"))

    # ---------------------------------------------------------
    # 3. Imputation and standardisation
    # ---------------------------------------------------------

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # ---------------------------------------------------------
    # 4. Pearson correlation
    # ---------------------------------------------------------

    corr = pd.DataFrame(
        np.corrcoef(X_scaled.T),
        index=cols,
        columns=cols,
    )

    strong_corr = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            value = corr.iloc[i, j]

            if abs(value) >= pearson_threshold:
                strong_corr.append(
                    (cols[i], cols[j], value)
                )

    strong_corr = sorted(
        strong_corr,
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    print(
        f"\nPearson correlations with "
        f"|r| >= {pearson_threshold:.2f}:"
    )

    if strong_corr:
        for a, b, value in strong_corr:
            print(
                f"{a:20s} -- {b:20s}: "
                f"r = {value:.3f}"
            )
    else:
        print("None.")

    # ---------------------------------------------------------
    # 5. Graphical Lasso with automatic regularisation
    # ---------------------------------------------------------

    glasso = GraphicalLassoCV(cv=5)
    glasso.fit(X_scaled)

    print(
        "\nGraphical Lasso selected alpha: "
        f"{glasso.alpha_:.6f}"
    )

    precision = pd.DataFrame(
        glasso.precision_,
        index=cols,
        columns=cols,
    )

    # ---------------------------------------------------------
    # 6. Convert precision matrix to partial correlations
    # ---------------------------------------------------------

    precision_array = glasso.precision_

    diagonal = np.sqrt(
        np.diag(precision_array)
    )

    partial_corr_array = (
        -precision_array
        / np.outer(diagonal, diagonal)
    )

    np.fill_diagonal(
        partial_corr_array,
        1.0,
    )

    partial_corr = pd.DataFrame(
        partial_corr_array,
        index=cols,
        columns=cols,
    )

    # ---------------------------------------------------------
    # 7. Sparse edges
    #
    # No manual partial-correlation threshold.
    # An edge is retained when Graphical Lasso gives a
    # non-zero off-diagonal precision entry.
    # ---------------------------------------------------------

    sparse_edges = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            precision_value = precision_array[i, j]

            # Numerical tolerance only; not a scientific threshold.
            if not np.isclose(
                precision_value,
                0.0,
                atol=1e-10,
            ):
                sparse_edges.append(
                    (
                        cols[i],
                        cols[j],
                        partial_corr_array[i, j],
                    )
                )

    sparse_edges = sorted(
        sparse_edges,
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    print(
        "\nGraphical Lasso sparse dependencies "
        "(automatically selected):"
    )

    if sparse_edges:
        for a, b, value in sparse_edges:
            print(
                f"{a:20s} -- {b:20s}: "
                f"partial r = {value:.4f}"
            )
    else:
        print("No sparse dependencies retained.")

    # ---------------------------------------------------------
    # 8. Return diagnostics
    # ---------------------------------------------------------

    return {
        "missing": missing,
        "correlation": corr,
        "strong_correlations": strong_corr,
        "selected_alpha": glasso.alpha_,
        "precision": precision,
        "partial_correlation": partial_corr,
        "sparse_edges": sparse_edges,
    }