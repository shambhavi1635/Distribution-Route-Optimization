"""Static figures (matplotlib) for the research paper / README."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

BLUE, GREY, ORANGE = "#2a6ebb", "#9aa3ad", "#d9822b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e8eb", "grid.linewidth": 0.8, "axes.axisbelow": True})


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_raw_vs_corrected(validated):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(validated["Raw Lead Time"], bins=60, color=GREY)
    ax[0].set(title="Raw Ship Date - Order Date (days)", xlabel="days", ylabel="orders")
    order = ["Same Day", "First Class", "Second Class", "Standard Class"]
    means = validated.groupby("Ship Mode")["Lead Time"].mean().reindex(order)
    ax[1].bar(order, means.values, color=BLUE)
    for i, v in enumerate(means.values):
        ax[1].text(i, v + 0.08, f"{v:.1f}", ha="center")
    ax[1].set(title="Corrected lead time by ship mode (days)", ylabel="days", ylim=(0, means.max() * 1.15))
    ax[1].tick_params(axis="x", rotation=15)
    return _save(fig, "01_raw_vs_corrected_lead_time.png")


def fig_variance(diag):
    d = diag.sort_values("variance_explained_pct")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(d["driver"], d["variance_explained_pct"], color=[BLUE if v > 5 else GREY for v in d["variance_explained_pct"]])
    ax.set(title="Share of lead-time variance explained (%)", xlabel="% of variance")
    return _save(fig, "02_variance_explained.png")


def fig_models(metrics):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(metrics["Model"], metrics["Holdout RMSE"], color=[BLUE if s else GREY for s in metrics["Selected"]])
    ax[0].set(title="Holdout RMSE (days, lower is better)", ylim=(0, metrics["Holdout RMSE"].max() * 1.25))
    ax[1].bar(metrics["Model"], metrics["Holdout R2"], color=[BLUE if s else GREY for s in metrics["Selected"]])
    ax[1].set(title="Holdout R2 (higher is better)", ylim=(0, 1))
    for a in ax:
        a.tick_params(axis="x", rotation=15)
    return _save(fig, "03_model_comparison.png")


def fig_routes(routes):
    rt = routes[routes["Route Cluster"].isin(["Fast", "Typical", "Slow"])]
    colors = {"Fast": BLUE, "Typical": GREY, "Slow": ORANGE}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in rt.groupby("Route Cluster"):
        ax.scatter(g["Avg_Distance"], g["Avg_Lead"], s=np.sqrt(g["Orders"]) * 6, alpha=0.7,
                   color=colors[name], label=name, edgecolor="white", linewidth=0.5)
    ax.set(title="Route clusters (bubble = orders)", xlabel="average distance (miles)",
           ylabel="average lead time (days)")
    ax.legend(title="Cluster")
    return _save(fig, "04_route_clusters.png")


def fig_recommendations(recs):
    r = recs.sort_values("Lead Reduction %")
    labels = r["Product"].str.replace("Wonka Bar - ", "").str.replace("Wonka Bar -", "") + ": " + \
        r["Current Factory"] + " -> " + r["Candidate Factory"]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(r) + 1.2))
    ax.barh(labels, r["Lead Reduction %"], color=BLUE)
    for i, v in enumerate(r["Lead Reduction %"]):
        ax.text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set(title="Recommended reassignments: expected lead-time reduction (%)", xlabel="%")
    return _save(fig, "05_recommendations.png")


def fig_factory_map(orders):
    dem = orders.groupby(["Dest Lat", "Dest Lon"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(dem["Dest Lon"], dem["Dest Lat"], s=np.sqrt(dem["n"]) * 8, color=GREY, alpha=0.55, label="Customer demand")
    for f, (la, lo) in C.FACTORIES.items():
        ax.scatter(lo, la, marker="D", s=110, color=C.FACTORY_COLORS[f], edgecolor="white", zorder=3)
        ax.annotate(f, (lo, la), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set(title="Factories and customer demand", xlabel="longitude", ylabel="latitude", xlim=(-128, -55), ylim=(27, 56))
    return _save(fig, "06_factory_map.png")


def make_all():
    prepared = pd.read_csv(C.PROCESSED / "orders_prepared.csv")
    validated = pd.read_csv(C.PROCESSED / "orders_validated.csv")
    paths = [
        fig_raw_vs_corrected(validated),
        fig_variance(pd.read_csv(C.PROCESSED / "signal_diagnostics.csv")),
        fig_models(pd.read_csv(C.PROCESSED / "model_comparison.csv")),
        fig_routes(pd.read_csv(C.PROCESSED / "route_clusters.csv")),
        fig_recommendations(pd.read_csv(C.PROCESSED / "recommendations.csv")),
        fig_factory_map(prepared),
    ]
    return paths


if __name__ == "__main__":
    for p in make_all():
        print(p)
