"""Step 1 - load, clean and validate the raw order data.

Important data finding (see README / research paper):
Ship Dates in the raw file sit 2-4 years after Order Dates, and the gap is a
constant per Order-ID-year cohort (~904, ~1269, ~1269 and ~1635 days for Same Day
orders). The raw gap therefore carries no operational meaning. We rebuild a
corrected lead time = raw gap - cohort offset, where the offset is the median gap
of the cohort's Same Day orders (nominally 0 days). The corrected values land on
the expected ~0 / 2 / 3 / 5 days for Same Day / First / Second / Standard Class.
"""
import numpy as np
import pandas as pd

from . import config as C
from .utils import haversine_miles

STRING_COLS = ["Order ID", "Ship Mode", "Country/Region", "City", "State/Province",
               "Postal Code", "Division", "Region", "Product ID", "Product Name"]


def load_raw(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.DATA_RAW, dtype={"Postal Code": str})
    df.columns = [c.strip() for c in df.columns]
    for c in STRING_COLS:
        df[c] = df[c].astype(str).str.strip()
    return df


def clean_and_validate(df: pd.DataFrame):
    """Return (validated_df, report_dict)."""
    report = {"rows_raw": int(len(df))}
    df = df.drop_duplicates(subset="Row ID").copy()
    report["duplicate_rows_removed"] = report["rows_raw"] - len(df)

    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")
    n = len(df)
    df = df.dropna(subset=["Order Date", "Ship Date"])
    report["unparseable_dates_removed"] = n - len(df)

    report["missing_values_total"] = int(df.isna().sum().sum())
    report["financial_identity_mismatches"] = int(
        ((df["Sales"] - df["Cost"] - df["Gross Profit"]).abs() > 0.02).sum())

    n = len(df)
    df = df[(df["Sales"] > 0) & (df["Units"] > 0) & (df["Cost"] >= 0)]
    report["non_positive_sales_units_removed"] = n - len(df)

    n = len(df)
    df = df[df["Ship Date"] >= df["Order Date"]]
    report["ship_before_order_removed"] = n - len(df)

    n = len(df)
    df = df[df["Product Name"].isin(C.PRODUCT_FACTORY)]
    report["unmapped_products_removed"] = n - len(df)

    missing_states = sorted(set(df["State/Province"]) - set(C.STATE_COORDS))
    if missing_states:
        raise ValueError(f"No coordinates configured for: {missing_states}")

    # --- corrected lead time --------------------------------------------------
    df["Order Cohort"] = df["Order ID"].str.extract(r"-(\d{4})-")[0].astype(int)
    df["Raw Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    cohort_min = df.groupby("Order Cohort")["Raw Lead Time"].min()
    offsets = (df[df["Ship Mode"] == "Same Day"].groupby("Order Cohort")["Raw Lead Time"].median()
               .reindex(cohort_min.index).fillna(cohort_min))
    df["Cohort Offset"] = df["Order Cohort"].map(offsets)
    df["Lead Time"] = (df["Raw Lead Time"] - df["Cohort Offset"]).clip(lower=0).astype(float)
    report["raw_lead_time_range_days"] = [int(df["Raw Lead Time"].min()), int(df["Raw Lead Time"].max())]
    report["cohort_offsets_days"] = {int(k): float(v) for k, v in offsets.items()}
    report["corrected_lead_time_range_days"] = [float(df["Lead Time"].min()), float(df["Lead Time"].max())]

    # --- enrichment ----------------------------------------------------------
    df["Origin Factory"] = df["Product Name"].map(C.PRODUCT_FACTORY)
    f_lat = df["Origin Factory"].map(lambda f: C.FACTORIES[f][0])
    f_lon = df["Origin Factory"].map(lambda f: C.FACTORIES[f][1])
    df["Dest Lat"] = df["State/Province"].map(lambda s: C.STATE_COORDS[s][0])
    df["Dest Lon"] = df["State/Province"].map(lambda s: C.STATE_COORDS[s][1])
    df["Distance Miles"] = haversine_miles(f_lat.values, f_lon.values,
                                           df["Dest Lat"].values, df["Dest Lon"].values)
    df["Unit Price"] = df["Sales"] / df["Units"]
    df["Margin Pct"] = df["Gross Profit"] / df["Sales"] * 100
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Year"] = df["Order Date"].dt.year

    report["rows_validated"] = int(len(df))
    return df.reset_index(drop=True), report
