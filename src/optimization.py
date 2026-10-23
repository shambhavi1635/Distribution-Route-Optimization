"""Step 5b - scoring, ranking, top-N recommendations and network KPIs.

Score for a (product, candidate factory) scenario
    speed   = 0.6 * lead-time-reduction  +  0.4 * risk-reduction (shorter routes)
    raw     = w * speed + (1 - w) * profit-impact          (w = speed priority, 0..1)
    score   = raw * (0.5 + 0.5 * confidence/100)           (low-evidence scenarios are damped)
Each component is clipped and scaled to [-1, 1]:
    lead-time reduction % : +/-50%      risk reduction (distance %) : +/-100%
    profit impact %       : +/-10% of gross profit
"""
import numpy as np
import pandas as pd

from . import config as C


def score_scenarios(sim: pd.DataFrame, speed_weight: float = 0.5, profit_alert_pct: float = None,
                    min_confidence: float = None) -> pd.DataFrame:
    s = sim.copy()
    profit_alert_pct = C.DEFAULT_ASSUMPTIONS["profit_alert_pct"] if profit_alert_pct is None else profit_alert_pct
    min_confidence = C.DEFAULT_ASSUMPTIONS["min_confidence"] if min_confidence is None else min_confidence
    if s.empty:
        return s
    lead = s["Lead Reduction %"].clip(-50, 50) / 50
    risk = s["Distance Reduction %"].clip(-100, 100) / 100
    profit = s["Profit Impact %"].clip(-10, 10) / 10
    speed = 0.6 * lead + 0.4 * risk
    raw = speed_weight * speed + (1 - speed_weight) * profit
    s["Score"] = raw * (0.5 + 0.5 * s["Confidence"] / 100)

    def flag(r):
        if r["Is Current"]:
            return "-"
        if r["Profit Impact %"] < profit_alert_pct or r["Confidence"] < min_confidence:
            return "High"
        if r["Confidence"] < 65 or r["Profit Stability %"] < 80 or r["Profit Impact %"] < 0:
            return "Medium"
        return "Low"

    s["Risk Level"] = s.apply(flag, axis=1)
    return s


def recommend(scored: pd.DataFrame, top_n: int = 10, include_high_risk: bool = True) -> pd.DataFrame:
    """Best alternative factory per product (score > 0 and faster), ranked; top-N overall."""
    alt = scored[~scored["Is Current"] & (scored["Score"] > 0) & (scored["Lead Reduction %"] > 0)]
    if not include_high_risk:
        alt = alt[alt["Risk Level"] != "High"]
    if alt.empty:
        return alt
    best = alt.sort_values("Score", ascending=False).groupby("Product", as_index=False).head(1)
    best = best.sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)
    best.insert(0, "Rank", np.arange(1, len(best) + 1))
    return best


def alternatives_per_product(scored: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Top-N ranked alternative factories *for each product* (the product-level view)."""
    alt = scored[~scored["Is Current"]].sort_values(["Product", "Score"], ascending=[True, False])
    out = alt.groupby("Product", as_index=False).head(top_n).copy()
    out["Rank in Product"] = out.groupby("Product").cumcount() + 1
    return out.reset_index(drop=True)


def network_kpis(scored: pd.DataFrame, recs: pd.DataFrame) -> dict:
    """The four KPIs from the brief, computed over the products/filters in `scored`."""
    cur = scored[scored["Is Current"]].set_index("Product")
    n_prod = len(cur)
    tot_orders = cur["Orders"].sum()
    base_lead_w = (cur["Current Lead (days)"] * cur["Orders"]).sum()

    if recs.empty:
        return {"Lead Time Reduction % (network)": 0.0, "Lead Time Reduction % (reassigned products)": 0.0,
                "Profit Impact Stability %": 100.0, "Scenario Confidence Score": 0.0,
                "Recommendation Coverage % (products)": 0.0, "Recommendation Coverage % (orders)": 0.0,
                "Products Recommended": 0, "Products Evaluated": int(n_prod),
                "Network Profit Impact ($)": 0.0}
    saved = (recs["Current Lead (days)"] - recs["Scenario Lead (days)"]) * recs["Orders"]
    w = recs["Orders"]
    return {
        "Lead Time Reduction % (network)": 100 * saved.sum() / base_lead_w,
        "Lead Time Reduction % (reassigned products)":
            100 * saved.sum() / (recs["Current Lead (days)"] * recs["Orders"]).sum(),
        "Profit Impact Stability %": float(np.average(recs["Profit Stability %"], weights=w)),
        "Scenario Confidence Score": float(np.average(recs["Confidence"], weights=w)),
        "Recommendation Coverage % (products)": 100 * len(recs) / n_prod,
        "Recommendation Coverage % (orders)": 100 * recs["Orders"].sum() / tot_orders,
        "Products Recommended": int(len(recs)),
        "Products Evaluated": int(n_prod),
        "Network Profit Impact ($)": float(recs["Profit Delta ($)"].sum()),
    }
