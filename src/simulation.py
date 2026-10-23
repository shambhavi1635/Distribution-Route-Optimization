"""Step 5a - scenario simulation engine.

For every product, simulate shipping the product's orders from each alternative
factory and estimate the change in:
  * lead time   = ML-predicted lead time  +  distance-driven transit component
  * distance    = great-circle miles factory -> destination state
  * profit      = change in freight cost (assumption-driven; no freight data exists)
  * confidence  = how much historical evidence supports the scenario

Predictions for every order x factory are computed once (vectorised) so that any
filter combination in the dashboard is an instant aggregation.
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES
from .utils import haversine_miles


class ScenarioSimulator:
    def __init__(self, orders: pd.DataFrame, bundle: dict):
        self.orders = orders.reset_index(drop=True)
        self.bundle = bundle
        self.factories = list(C.FACTORIES)
        self._precompute()

    # ------------------------------------------------------------------ #
    def _precompute(self):
        o, pipe = self.orders, self.bundle["pipeline"]
        pred, dist = {}, {}
        for f, (lat, lon) in C.FACTORIES.items():
            tmp = o[FEATURES].copy()
            d = haversine_miles(lat, lon, o["Dest Lat"].values, o["Dest Lon"].values)
            tmp["Origin Factory"] = f
            tmp["Distance Miles"] = d
            pred[f] = pipe.predict(tmp)
            dist[f] = d
        self.pred = pd.DataFrame(pred)
        self.dist = pd.DataFrame(dist)
        self.hist_divisions = o.groupby("Origin Factory")["Division"].agg(lambda s: set(s)).to_dict()
        self.t_mult = o["Ship Mode"].map(C.MODE_TRANSIT_MULT).values
        self.c_mult = o["Ship Mode"].map(C.MODE_COST_MULT).values

    # ------------------------------------------------------------------ #
    def simulate(self, products=None, regions=None, ship_modes=None, assumptions=None) -> pd.DataFrame:
        a = {**C.DEFAULT_ASSUMPTIONS, **(assumptions or {})}
        o = self.orders
        mask = np.ones(len(o), dtype=bool)
        if regions:
            mask &= o["Region"].isin(regions).values
        if ship_modes:
            mask &= o["Ship Mode"].isin(ship_modes).values
        products = products or list(C.PRODUCT_FACTORY)
        fit = float(np.clip(self.bundle.get("holdout_r2", 0.0), 0, 1))

        rows = []
        for p in products:
            m = mask & (o["Product Name"].values == p)
            n = int(m.sum())
            if n == 0:
                continue
            cur = C.PRODUCT_FACTORY[p]
            division = o.loc[m, "Division"].iloc[0]
            units = o["Units"].values[m]
            gp = o["Gross Profit"].values[m]
            sales = o["Sales"].values[m]
            tm, cm = self.t_mult[m], self.c_mult[m]

            pc, dc = self.pred[cur].values[m], self.dist[cur].values[m]
            eff_cur = pc + dc / 1000 * a["days_per_1000mi"] * tm
            freight_cur = dc / 1000 * a["freight_usd_per_unit_1000mi"] * units * cm
            support = min(1.0, n / 30)

            for f in self.factories:
                pf, df_ = self.pred[f].values[m], self.dist[f].values[m]
                eff_new = pf + df_ / 1000 * a["days_per_1000mi"] * tm
                freight_new = df_ / 1000 * a["freight_usd_per_unit_1000mi"] * units * cm
                d_profit_i = -(freight_new - freight_cur)
                analog = 1.0 if f == cur else (0.7 if division in self.hist_divisions.get(f, set()) else 0.4)
                conf = 100 * (0.4 * support + 0.3 * fit + 0.3 * analog)
                lead_cur, lead_new = float(eff_cur.mean()), float(eff_new.mean())
                dist_cur, dist_new = float(dc.mean()), float(df_.mean())
                rows.append({
                    "Product": p, "Division": division, "Current Factory": cur, "Candidate Factory": f,
                    "Is Current": f == cur, "Orders": n, "Units": int(units.sum()),
                    "Sales": float(sales.sum()), "Gross Profit": float(gp.sum()),
                    "Current Lead (days)": lead_cur, "Scenario Lead (days)": lead_new,
                    "ML Lead Delta (days)": float((pf - pc).mean()),
                    "Lead Delta (days)": lead_new - lead_cur,
                    "Lead Reduction %": 100 * (lead_cur - lead_new) / lead_cur if lead_cur > 0 else 0.0,
                    "Current Distance (mi)": dist_cur, "Scenario Distance (mi)": dist_new,
                    "Distance Reduction %": 100 * (dist_cur - dist_new) / dist_cur if dist_cur > 0 else 0.0,
                    "Profit Delta ($)": float(d_profit_i.sum()),
                    "Profit Impact %": 100 * float(d_profit_i.sum()) / float(gp.sum()) if gp.sum() else 0.0,
                    "Profit Stability %": 100 * float((d_profit_i >= -0.01 * gp).mean()),
                    "Confidence": conf,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def by_region(self, product: str, factory: str, ship_modes=None, assumptions=None) -> pd.DataFrame:
        """Lead time & distance per destination region for one product shipped from `factory`
        vs its current factory (used by the What-If tab)."""
        a = {**C.DEFAULT_ASSUMPTIONS, **(assumptions or {})}
        o = self.orders
        m = (o["Product Name"].values == product)
        if ship_modes:
            m &= o["Ship Mode"].isin(ship_modes).values
        cur = C.PRODUCT_FACTORY[product]
        sub = o.loc[m, ["Region"]].copy()
        tm = self.t_mult[m]
        for tag, f in (("Current", cur), ("Scenario", factory)):
            sub[f"{tag} Lead"] = self.pred[f].values[m] + self.dist[f].values[m] / 1000 * a["days_per_1000mi"] * tm
            sub[f"{tag} Distance"] = self.dist[f].values[m]
        g = sub.groupby("Region").mean().reset_index()
        g["Orders"] = sub.groupby("Region").size().values
        g["Lead Improvement (days)"] = g["Current Lead"] - g["Scenario Lead"]
        return g
