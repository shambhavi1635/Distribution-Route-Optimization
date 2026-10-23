"""Smoke test: confirms the artefacts exist and that the Streamlit app runs without exceptions.

    python check_dashboard.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    "data/raw/Nassau Candy Distributor.csv",
    "data/processed/orders_validated.csv",
    "data/processed/orders_prepared.csv",
    "data/processed/feature_matrix.csv",
    "data/processed/route_clusters.csv",
    "data/processed/scenario_simulation.csv",
    "data/processed/recommendations.csv",
    "models/best_model.joblib",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("Missing artefacts (run `python run_pipeline.py` first):")
        for m in missing:
            print("  -", m)
        return 1
    print("All artefacts present.")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    if at.exception:
        print("App raised exceptions:")
        for e in at.exception:
            print("  -", e.value)
        return 1
    print(f"App rendered: {len(at.tabs)} tabs, {len(at.metric)} KPI/metric tiles, no exceptions.")

    # exercise the sidebar controls once
    at.slider[0].set_value(90).run()
    at.multiselect[1].set_value(["Standard Class"]).run()
    if at.exception:
        print("App raised exceptions after changing controls:", [e.value for e in at.exception])
        return 1
    print("Sidebar controls (priority slider, ship-mode filter) OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
