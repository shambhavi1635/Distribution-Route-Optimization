"""Nassau Candy - Factory Reallocation & Shipping Optimization Recommendation System.

Run with:   streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import clustering, config as C, data_prep, features, modeling
from src.optimization import alternatives_per_product, network_kpis, recommend, score_scenarios
from src.simulation import ScenarioSimulator

st.set_page_config(page_title="Nassau Candy | Factory Reallocation Optimizer", page_icon="🍬", layout="wide")

CURRENT_GREY, SCENARIO_BLUE = "#9aa3ad", "#2a6ebb"
RISK_COLORS = {"Low": "#2e8b57", "Medium": "#d68910", "High": "#c0392b"}
TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# helpers (compatible with old and new Streamlit APIs)
# ---------------------------------------------------------------------------
def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def fmt(df, cols_fmt):
    return df.style.format(cols_fmt)


# ---------------------------------------------------------------------------
# data / model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_prepared() -> pd.DataFrame:
    p = C.PROCESSED / "orders_prepared.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["Order Date", "Ship Date"])
    validated, _ = data_prep.clean_and_validate(data_prep.load_raw())
    prepared, _ = features.prepare_dataset(validated)
    return prepared


@st.cache_resource(show_spinner="Loading model and pre-computing factory scenarios...")
def get_simulator():
    df = load_prepared()
    bundle = modeling.load_bundle(df)
    return ScenarioSimulator(df, bundle), bundle


@st.cache_data(show_spinner=False)
def route_tables():
    df = load_prepared()
    routes = clustering.cluster_routes(clustering.route_summary(df))
    return routes, clustering.congestion_table(df), clustering.cluster_products(df), modeling.signal_diagnostics(df)


orders = load_prepared()
sim, bundle = get_simulator()

# ---------------------------------------------------------------------------
# sidebar - user capabilities
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")
prod_order = orders["Product Name"].value_counts().index.tolist()
product = st.sidebar.selectbox("Product selector", prod_order, help="Drives the simulator and what-if tabs.")
regions = st.sidebar.multiselect("Region selector", sorted(orders["Region"].unique()),
                                 help="Customer regions to include. Empty = all regions.")
modes = st.sidebar.multiselect("Ship mode filter", C.SHIP_MODES, help="Empty = all ship modes.")
priority = st.sidebar.slider("Optimization priority", 0, 100, 50, step=5,
                             help="0 = protect profit, 100 = maximise speed.")
st.sidebar.caption(f"Weighting: {100 - priority}% profit / {priority}% speed")
top_n = st.sidebar.slider("Top-N recommendations", 3, 15, 8)
include_high = st.sidebar.checkbox("Include high-risk reassignments in rankings", value=True)

with st.sidebar.expander("Assumptions (editable)"):
    st.caption("The dataset has no freight-cost or transit-time field, so these drive the "
               "distance-related part of the simulation.")
    days_per_1000 = st.number_input("Transit days per 1,000 miles (Standard Class)", 0.0, 5.0,
                                    float(C.DEFAULT_ASSUMPTIONS["days_per_1000mi"]), 0.1)
    freight = st.number_input("Freight USD per unit per 1,000 miles (Standard Class)", 0.0, 2.0,
                              float(C.DEFAULT_ASSUMPTIONS["freight_usd_per_unit_1000mi"]), 0.01)
    profit_alert = st.number_input("Profit alert threshold (% of gross profit)", -50.0, 0.0,
                                   float(C.DEFAULT_ASSUMPTIONS["profit_alert_pct"]), 0.5)
    min_conf = st.number_input("Minimum scenario confidence", 0.0, 100.0,
                               float(C.DEFAULT_ASSUMPTIONS["min_confidence"]), 5.0)

assump = {"days_per_1000mi": days_per_1000, "freight_usd_per_unit_1000mi": freight}

sim_df = sim.simulate(regions=regions or None, ship_modes=modes or None, assumptions=assump)
if sim_df.empty:
    st.warning("No orders match the selected region / ship-mode filters.")
    st.stop()
scored = score_scenarios(sim_df, speed_weight=priority / 100, profit_alert_pct=profit_alert, min_confidence=min_conf)
recs = recommend(scored, top_n=top_n, include_high_risk=include_high)
kpis = network_kpis(scored, recs)

# ---------------------------------------------------------------------------
# header + KPIs
# ---------------------------------------------------------------------------
st.title("Factory Reallocation & Shipping Optimization")
st.caption("Predict shipping outcomes under different factory configurations, recommend product "
           "reassignments, and balance shipping efficiency with profitability.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lead Time Reduction", f"{kpis['Lead Time Reduction % (network)']:.1f}%",
          help="Order-weighted reduction in end-to-end lead time across all evaluated orders if the "
               "recommended reassignments were applied (operational gain).")
k2.metric("Profit Impact Stability", f"{kpis['Profit Impact Stability %']:.0f}%",
          help="Share of orders whose profit does not fall by more than 1% of their gross profit "
               "under the recommended moves (financial safety).")
k3.metric("Scenario Confidence", f"{kpis['Scenario Confidence Score']:.0f} / 100",
          help="Evidence behind the scenarios: sample size, model fit and whether the factory has "
               "shipped this division before (reliability).")
k4.metric("Recommendation Coverage", f"{kpis['Recommendation Coverage % (products)']:.0f}%",
          delta=f"{kpis['Products Recommended']} of {kpis['Products Evaluated']} products",
          delta_color="off", help="Share of products with an actionable reassignment (scalability).")

tabs = st.tabs(["Factory Optimization Simulator", "What-If Scenario Analysis", "Recommendation Dashboard",
                "Risk & Impact Panel", "Model & Route Insights"])

# ===========================================================================
# 1. Factory Optimization Simulator
# ===========================================================================
with tabs[0]:
    st.subheader(f"{product}")
    ps = scored[scored["Product"] == product].copy()
    if ps.empty:
        st.info("This product has no orders under the current filters.")
    else:
        cur = ps["Current Factory"].iloc[0]
        st.write(f"Currently produced at **{cur}** - {int(ps['Orders'].iloc[0]):,} orders in scope.")
        ps["Label"] = np.where(ps["Is Current"], ps["Candidate Factory"] + " (current)", ps["Candidate Factory"])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(ps, x="Label", y="Scenario Lead (days)", color="Candidate Factory",
                         color_discrete_map=C.FACTORY_COLORS, template=TEMPLATE,
                         title="Predicted end-to-end lead time by factory (days)")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="days")
            fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
            show(fig)
        with c2:
            fig = px.bar(ps, x="Label", y="Scenario Distance (mi)", color="Candidate Factory",
                         color_discrete_map=C.FACTORY_COLORS, template=TEMPLATE,
                         title="Average shipping distance by factory (miles)")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="miles")
            fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
            show(fig)

        cols = ["Candidate Factory", "Is Current", "Scenario Lead (days)", "Lead Delta (days)", "Lead Reduction %",
                "Scenario Distance (mi)", "Distance Reduction %", "Profit Delta ($)", "Profit Impact %",
                "Confidence", "Risk Level", "Score"]
        show_df(fmt(ps.sort_values("Score", ascending=False)[cols], {
            "Scenario Lead (days)": "{:.2f}", "Lead Delta (days)": "{:+.3f}", "Lead Reduction %": "{:+.2f}",
            "Scenario Distance (mi)": "{:,.0f}", "Distance Reduction %": "{:+.1f}", "Profit Delta ($)": "{:+,.0f}",
            "Profit Impact %": "{:+.2f}", "Confidence": "{:.0f}", "Score": "{:+.3f}"}))

        # demand map
        m = (orders["Product Name"] == product)
        if regions:
            m &= orders["Region"].isin(regions)
        if modes:
            m &= orders["Ship Mode"].isin(modes)
        dem = (orders[m].groupby(["State/Province", "Dest Lat", "Dest Lon"]).size()
               .reset_index(name="Orders"))
        mp = go.Figure()
        mp.add_trace(go.Scattergeo(lat=dem["Dest Lat"], lon=dem["Dest Lon"], mode="markers",
                                   marker=dict(size=np.sqrt(dem["Orders"]) * 3 + 4, color="#9aa3ad",
                                               opacity=0.55, line=dict(width=1, color="white")),
                                   text=dem["State/Province"] + ": " + dem["Orders"].astype(str) + " orders",
                                   hoverinfo="text", name="Customer demand"))
        for f, (la, lo) in C.FACTORIES.items():
            mp.add_trace(go.Scattergeo(lat=[la], lon=[lo], mode="markers+text",
                                       marker=dict(size=14, symbol="diamond", color=C.FACTORY_COLORS[f],
                                                   line=dict(width=2, color="white")),
                                       text=[f + (" (current)" if f == cur else "")], textposition="top center",
                                       name=f, hoverinfo="text"))
        mp.update_geos(scope="north america", showland=True, landcolor="#f3f4f6", showcountries=True,
                       countrycolor="#cfd4da", showsubunits=True, subunitcolor="#e1e4e8",
                       lataxis_range=[24, 58], lonaxis_range=[-130, -55])
        mp.update_layout(template=TEMPLATE, height=480, margin=dict(l=0, r=0, t=40, b=0),
                         title="Factories and customer demand (bubble size = orders)",
                         legend=dict(orientation="h", y=-0.02))
        show(mp)

# ===========================================================================
# 2. What-If Scenario Analysis
# ===========================================================================
with tabs[1]:
    st.subheader("Current vs recommended assignment")
    ps = scored[(scored["Product"] == product) & (~scored["Is Current"])]
    if ps.empty:
        st.info("No scenario data for this product under the current filters.")
    else:
        best = ps.sort_values("Score", ascending=False)["Candidate Factory"].iloc[0]
        alts = ps["Candidate Factory"].tolist()
        scen = st.selectbox("Scenario factory (defaults to the top-ranked alternative)", alts,
                            index=alts.index(best))
        r = ps[ps["Candidate Factory"] == scen].iloc[0]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Lead time", f"{r['Scenario Lead (days)']:.2f} d", f"{r['Lead Delta (days)']:+.2f} d",
                  delta_color="inverse")
        a2.metric("Distance", f"{r['Scenario Distance (mi)']:,.0f} mi", f"{-r['Distance Reduction %']:+.1f}%",
                  delta_color="inverse")
        a3.metric("Profit impact", f"{r['Profit Delta ($)']:+,.0f} $", f"{r['Profit Impact %']:+.2f}% of GP")
        a4.metric("Confidence / risk", f"{r['Confidence']:.0f}", r["Risk Level"], delta_color="off")

        reg = sim.by_region(product, scen, modes or None, assump)
        reg = reg[reg["Region"].isin(regions)] if regions else reg
        long = pd.concat([
            reg[["Region", "Current Lead"]].rename(columns={"Current Lead": "Lead"}).assign(Assignment="Current"),
            reg[["Region", "Scenario Lead"]].rename(columns={"Scenario Lead": "Lead"}).assign(Assignment="Scenario"),
        ])
        fig = px.bar(long, x="Region", y="Lead", color="Assignment", barmode="group", template=TEMPLATE,
                     color_discrete_map={"Current": CURRENT_GREY, "Scenario": SCENARIO_BLUE},
                     title=f"Lead time by destination region: {C.PRODUCT_FACTORY[product]} vs {scen}")
        fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
        fig.update_layout(yaxis_title="days", xaxis_title=None)
        show(fig)
        show_df(fmt(reg[["Region", "Orders", "Current Lead", "Scenario Lead", "Lead Improvement (days)",
                         "Current Distance", "Scenario Distance"]],
                    {"Current Lead": "{:.2f}", "Scenario Lead": "{:.2f}", "Lead Improvement (days)": "{:+.3f}",
                     "Current Distance": "{:,.0f}", "Scenario Distance": "{:,.0f}"}))

    st.subheader("All products: current vs recommended factory")
    rec_map = recs.set_index("Product") if not recs.empty else pd.DataFrame()
    rows = []
    for p, g in scored.groupby("Product"):
        c = g[g["Is Current"]].iloc[0]
        if not recs.empty and p in rec_map.index:
            rr = rec_map.loc[p]
            rows.append({"Product": p, "Current Factory": c["Current Factory"],
                         "Recommended Factory": rr["Candidate Factory"],
                         "Lead Time Now (d)": c["Current Lead (days)"], "Lead Time After (d)": rr["Scenario Lead (days)"],
                         "Lead Reduction %": rr["Lead Reduction %"], "Profit Impact %": rr["Profit Impact %"],
                         "Risk": rr["Risk Level"]})
        else:
            rows.append({"Product": p, "Current Factory": c["Current Factory"], "Recommended Factory": "keep current",
                         "Lead Time Now (d)": c["Current Lead (days)"], "Lead Time After (d)": c["Current Lead (days)"],
                         "Lead Reduction %": 0.0, "Profit Impact %": 0.0, "Risk": "-"})
    cmp_df = pd.DataFrame(rows).sort_values("Lead Reduction %", ascending=False)
    show_df(fmt(cmp_df, {"Lead Time Now (d)": "{:.2f}", "Lead Time After (d)": "{:.2f}",
                         "Lead Reduction %": "{:+.2f}", "Profit Impact %": "{:+.2f}"}))

# ===========================================================================
# 3. Recommendation Dashboard
# ===========================================================================
with tabs[2]:
    st.subheader(f"Top {top_n} reassignment recommendations")
    st.caption(f"Ranked by score at {priority}% speed / {100 - priority}% profit priority. "
               "Score = priority-weighted blend of lead-time reduction, risk reduction (shorter routes) and "
               "profit impact, damped by scenario confidence.")
    if recs.empty:
        st.info("No reassignment improves on the current setup under these settings.")
    else:
        t = recs.copy()
        t["Move"] = t["Product"] + ":  " + t["Current Factory"] + "  ->  " + t["Candidate Factory"]
        show_df(fmt(t[["Rank", "Product", "Current Factory", "Candidate Factory", "Orders", "Lead Reduction %",
                       "Distance Reduction %", "Profit Impact %", "Profit Stability %", "Confidence",
                       "Risk Level", "Score"]],
                    {"Lead Reduction %": "{:+.2f}", "Distance Reduction %": "{:+.1f}", "Profit Impact %": "{:+.2f}",
                     "Profit Stability %": "{:.0f}", "Confidence": "{:.0f}", "Score": "{:.3f}"}))
        fig = px.bar(t.sort_values("Lead Reduction %"), x="Lead Reduction %", y="Move", orientation="h",
                     template=TEMPLATE, color_discrete_sequence=[SCENARIO_BLUE],
                     title="Expected lead-time reduction per recommended move (%)")
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig.update_layout(yaxis_title=None, height=max(320, 42 * len(t) + 120))
        show(fig)
        st.download_button("Download recommendations (CSV)", t.drop(columns="Move").to_csv(index=False),
                           file_name="recommendations.csv", mime="text/csv")
    with st.expander("Top-3 ranked alternatives for every product"):
        alt = alternatives_per_product(scored, 3)
        show_df(fmt(alt[["Product", "Rank in Product", "Candidate Factory", "Lead Reduction %",
                         "Distance Reduction %", "Profit Impact %", "Confidence", "Risk Level", "Score"]],
                    {"Lead Reduction %": "{:+.2f}", "Distance Reduction %": "{:+.1f}",
                     "Profit Impact %": "{:+.2f}", "Confidence": "{:.0f}", "Score": "{:+.3f}"}))

# ===========================================================================
# 4. Risk & Impact Panel
# ===========================================================================
with tabs[3]:
    st.subheader("Profit impact alerts and high-risk reassignments")
    alt_all = scored[~scored["Is Current"]].copy()

    def reason(r):
        out = []
        if r["Profit Impact %"] < profit_alert:
            out.append(f"profit {r['Profit Impact %']:+.1f}% of gross profit")
        if r["Confidence"] < min_conf:
            out.append(f"low confidence ({r['Confidence']:.0f}; {int(r['Orders'])} orders)")
        return "; ".join(out)

    alt_all["Why flagged"] = alt_all.apply(reason, axis=1)
    high = alt_all[alt_all["Risk Level"] == "High"]
    profit_hits = alt_all[alt_all["Profit Impact %"] < profit_alert]

    if profit_hits.empty:
        st.success(f"No scenario drops profit by more than {abs(profit_alert):.1f}% of gross profit.")
    else:
        st.error(f"{len(profit_hits)} scenario(s) breach the profit alert threshold ({profit_alert:.1f}%).")
    rec_high = recs[recs["Risk Level"] == "High"] if not recs.empty else recs
    if len(rec_high):
        st.warning(f"{len(rec_high)} of the ranked recommendations are high-risk "
                   f"(mostly thin order history): " + ", ".join(rec_high["Product"]) + ".")
    elif not recs.empty:
        st.success("None of the ranked recommendations is high-risk.")

    fig = px.scatter(alt_all, x="Lead Reduction %", y="Profit Impact %", size=np.sqrt(alt_all["Orders"]),
                     color="Risk Level", color_discrete_map=RISK_COLORS, template=TEMPLATE,
                     category_orders={"Risk Level": ["Low", "Medium", "High"]},
                     hover_data=["Product", "Candidate Factory", "Orders", "Confidence"],
                     title="Every alternative scenario: speed gain vs profit impact (bubble = orders)")
    fig.add_hline(y=0, line_color="#bbb")
    fig.add_vline(x=0, line_color="#bbb")
    show(fig)

    st.markdown("**High-risk scenarios**")
    if high.empty:
        st.write("None.")
    else:
        show_df(fmt(high.sort_values("Orders", ascending=False)[
            ["Product", "Candidate Factory", "Orders", "Lead Reduction %", "Profit Impact %", "Confidence",
             "Why flagged"]], {"Lead Reduction %": "{:+.2f}", "Profit Impact %": "{:+.2f}", "Confidence": "{:.0f}"}))
    st.caption("Risk levels: High = profit alert breached or confidence below the minimum; Medium = confidence < 65, "
               "profit stability < 80% or a profit decline; Low = otherwise.")

# ===========================================================================
# 5. Model & Route Insights
# ===========================================================================
with tabs[4]:
    routes, congestion, prod_clusters, diag = route_tables()
    st.subheader("Predictive model")
    st.write(f"Selected model: **{bundle['model_name']}** (holdout R2 = {bundle['holdout_r2']:.3f}). "
             "Chosen by accuracy and interpretability: the simplest model within 2% of the best RMSE.")
    mt = bundle["metrics"].copy()
    show_df(fmt(mt, {"CV RMSE": "{:.3f}", "Holdout RMSE": "{:.3f}", "Holdout MAE": "{:.3f}", "Holdout R2": "{:.3f}"}))
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(diag.sort_values("variance_explained_pct"), x="variance_explained_pct", y="driver",
                     orientation="h", template=TEMPLATE, color_discrete_sequence=[SCENARIO_BLUE],
                     title="Share of lead-time variance explained by each driver (%)")
        fig.update_layout(yaxis_title=None, xaxis_title="% of variance")
        show(fig)
    with c2:
        imp = bundle["importance"].sort_values("rmse_increase")
        fig = px.bar(imp, x="rmse_increase", y="feature", orientation="h", template=TEMPLATE,
                     color_discrete_sequence=[SCENARIO_BLUE],
                     title="Permutation importance (RMSE increase when shuffled)")
        fig.update_layout(yaxis_title=None)
        show(fig)
    st.info("Reading this honestly: in this dataset lead time is almost entirely a function of **ship mode**. "
            "Factory, region and distance explain well under 1% of its variance, so the model alone cannot "
            "rank factories. The simulator therefore adds an explicit, adjustable distance-driven transit "
            "and freight component (sidebar > Assumptions) on top of the model prediction.")

    st.subheader("Route clusters")
    rt = routes[routes["Route Cluster"].isin(["Fast", "Typical", "Slow"])]
    fig = px.scatter(rt, x="Avg_Distance", y="Avg_Lead", size="Orders", color="Route Cluster",
                     template=TEMPLATE, hover_name="Route",
                     color_discrete_map={"Fast": "#2a6ebb", "Typical": "#9aa3ad", "Slow": "#d9822b"},
                     category_orders={"Route Cluster": ["Fast", "Typical", "Slow"]},
                     title="Factory-to-state routes (routes with at least 5 orders)")
    fig.update_layout(xaxis_title="average distance (miles)", yaxis_title="average lead time (days)")
    show(fig)
    st.markdown("**Slowest routes**")
    slow = routes[routes["Route Cluster"] == "Slow"].sort_values("Avg_Lead", ascending=False)
    show_df(fmt(slow[["Route", "Orders", "Avg_Lead", "Std_Lead", "Avg_Distance", "Consistently Slow"]],
                {"Avg_Lead": "{:.2f}", "Std_Lead": "{:.2f}", "Avg_Distance": "{:,.0f}"}))
    st.markdown("**Congested region-product combinations** (top 30% by volume and by lead time)")
    cg = congestion[congestion["Congested"]]
    if cg.empty:
        st.write("None flagged.")
    else:
        show_df(fmt(cg[["Region", "Product Name", "Orders", "Avg_Lead", "Std_Lead", "Avg_Distance"]],
                    {"Avg_Lead": "{:.2f}", "Std_Lead": "{:.2f}", "Avg_Distance": "{:,.0f}"}))

st.divider()
st.caption("Lead time = corrected shipping lead time (raw gap minus per-cohort offset) + distance-driven transit "
           "assumption. Distances are great-circle miles from factory to approximate state/province centres. "
           "Freight cost and per-factory production cost are not in the data: profit impact reflects only "
           "freight changes under the assumptions in the sidebar.")
