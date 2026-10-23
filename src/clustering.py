"""Step 4 - route & product clustering; slow-route and congestion detection."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config as C


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["Origin Factory", "State/Province"])
         .agg(Orders=("Row ID", "count"), Units=("Units", "sum"), Sales=("Sales", "sum"),
              Gross_Profit=("Gross Profit", "sum"), Avg_Lead=("Lead Time", "mean"),
              Std_Lead=("Lead Time", "std"), P90_Lead=("Lead Time", lambda s: s.quantile(0.9)),
              Avg_Distance=("Distance Miles", "mean"))
         .reset_index())
    g["Std_Lead"] = g["Std_Lead"].fillna(0)
    g["Margin_Pct"] = g["Gross_Profit"] / g["Sales"] * 100
    g["Route"] = g["Origin Factory"] + " -> " + g["State/Province"]
    return g


def cluster_routes(rs: pd.DataFrame, k: int = 3, min_orders: int = 5) -> pd.DataFrame:
    """KMeans on (avg lead, lead std, distance, log volume). Clusters are labelled
    Fast / Typical / Slow by their mean lead time."""
    rs = rs.copy()
    sub = rs[rs["Orders"] >= min_orders].copy()
    X = StandardScaler().fit_transform(
        np.column_stack([sub["Avg_Lead"], sub["Std_Lead"], sub["Avg_Distance"], np.log1p(sub["Orders"])]))
    km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(X)
    sub["_c"] = km.labels_
    rank = sub.groupby("_c")["Avg_Lead"].mean().sort_values().index.tolist()
    names = ["Fast", "Typical", "Slow"] if k == 3 else [f"Tier {i + 1}" for i in range(k)]
    sub["Route Cluster"] = sub["_c"].map({c: names[i] for i, c in enumerate(rank)})
    out = rs.merge(sub[["Route", "Route Cluster"]], on="Route", how="left")
    out["Route Cluster"] = out["Route Cluster"].fillna("Low volume (<%d orders)" % min_orders)
    out["Consistently Slow"] = (out["Route Cluster"] == "Slow") & (out["Std_Lead"] <= out["Std_Lead"].median())
    return out


def congestion_table(df: pd.DataFrame, min_orders: int = 10) -> pd.DataFrame:
    """Region x product combinations with high volume AND above-typical lead time."""
    g = (df.groupby(["Region", "Product Name"])
         .agg(Orders=("Row ID", "count"), Units=("Units", "sum"), Avg_Lead=("Lead Time", "mean"),
              Std_Lead=("Lead Time", "std"), Avg_Distance=("Distance Miles", "mean"))
         .reset_index())
    g = g[g["Orders"] >= min_orders].copy()
    g["Volume Rank"] = g["Orders"].rank(pct=True)
    g["Lead Rank"] = g["Avg_Lead"].rank(pct=True)
    g["Congested"] = (g["Volume Rank"] >= 0.7) & (g["Lead Rank"] >= 0.7)
    return g.sort_values(["Congested", "Orders"], ascending=False).reset_index(drop=True)


def cluster_products(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    p = (df.groupby(["Product Name", "Origin Factory", "Division"])
         .agg(Orders=("Row ID", "count"), Sales=("Sales", "sum"), Gross_Profit=("Gross Profit", "sum"),
              Avg_Lead=("Lead Time", "mean"), Avg_Distance=("Distance Miles", "mean"))
         .reset_index())
    p["Margin_Pct"] = p["Gross_Profit"] / p["Sales"] * 100
    X = StandardScaler().fit_transform(np.column_stack(
        [np.log1p(p["Orders"]), p["Avg_Lead"], p["Avg_Distance"], p["Margin_Pct"]]))
    km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(X)
    p["_c"] = km.labels_
    rank = p.groupby("_c")["Orders"].mean().sort_values(ascending=False).index.tolist()
    names = ["Core volume", "Mid volume", "Long tail"] if k == 3 else [f"Group {i + 1}" for i in range(k)]
    p["Product Cluster"] = p["_c"].map({c: names[i] for i, c in enumerate(rank)})
    return p.drop(columns="_c").sort_values("Orders", ascending=False).reset_index(drop=True)
