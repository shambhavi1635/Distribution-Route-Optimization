"""Step 2 - outlier removal, encoding and normalisation (training-ready feature matrix)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["Product Name", "Origin Factory", "Region", "Ship Mode", "Division", "Country/Region"]
NUMERIC = ["Distance Miles", "Units", "Sales", "Unit Price", "Order Month"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Lead Time"


def _iqr_bad(s: pd.Series, k: float) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def remove_extreme_outliers(df: pd.DataFrame, k: float = 3.0, min_group: int = 20):
    """Drop extreme outliers (beyond Q1 - k*IQR / Q3 + k*IQR, k=3).

    * Lead Time is tested globally.
    * Sales and Units are tested *within each product* - the catalogue mixes ~$3 bars with
      ~$20 specialty items, so a global Sales test would wrongly discard a whole product.
      Products with fewer than `min_group` orders are not tested.
    """
    bad_lead = _iqr_bad(df["Lead Time"], k)
    bad_sales = pd.Series(False, index=df.index)
    bad_units = pd.Series(False, index=df.index)
    for _, idx in df.groupby("Product Name").groups.items():
        if len(idx) >= min_group:
            bad_sales.loc[idx] = _iqr_bad(df.loc[idx, "Sales"], k)
            bad_units.loc[idx] = _iqr_bad(df.loc[idx, "Units"], k)
    drop = bad_lead | bad_sales | bad_units
    report = {"rows_before": int(len(df)), "rows_after": int((~drop).sum()),
              "rows_removed": int(drop.sum()), "rule": f"{k} x IQR (Sales/Units per product)",
              "by_column": {"Lead Time": int(bad_lead.sum()), "Sales": int(bad_sales.sum()),
                            "Units": int(bad_units.sum())}}
    return df[~drop].reset_index(drop=True), report


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def prepare_dataset(validated: pd.DataFrame):
    """Outlier removal -> prepared frame + report."""
    prepared, report = remove_extreme_outliers(validated)
    return prepared, report


def export_feature_matrix(prepared: pd.DataFrame) -> pd.DataFrame:
    """Fit the preprocessor on the prepared data and return the dense feature matrix."""
    pre = build_preprocessor()
    X = pre.fit_transform(prepared[FEATURES])
    cols = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    out = pd.DataFrame(X, columns=cols)
    out[TARGET] = prepared[TARGET].values
    return out
