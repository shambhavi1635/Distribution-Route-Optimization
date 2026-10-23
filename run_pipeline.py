"""Run the whole analysis end-to-end and write every artefact to data/processed, models/ and reports/.

    python run_pipeline.py
"""
import json

import pandas as pd

from src import config as C
from src import clustering, data_prep, features, modeling
from src.optimization import alternatives_per_product, network_kpis, recommend, score_scenarios
from src.simulation import ScenarioSimulator


def main():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    summary = {}

    # 1. clean & validate -------------------------------------------------------
    raw = data_prep.load_raw()
    validated, vrep = data_prep.clean_and_validate(raw)
    validated.to_csv(C.PROCESSED / "orders_validated.csv", index=False)
    summary["validation"] = vrep
    print(f"[1/5] validated rows: {len(validated):,}")

    # 2. features -----------------------------------------------------------------
    prepared, orep = features.prepare_dataset(validated)
    prepared.to_csv(C.PROCESSED / "orders_prepared.csv", index=False)
    features.export_feature_matrix(prepared).to_csv(C.PROCESSED / "feature_matrix.csv", index=False)
    summary["outliers"] = orep
    print(f"[2/5] prepared rows: {len(prepared):,} (removed {orep['rows_removed']})")

    # 3. models -------------------------------------------------------------------
    diag = modeling.signal_diagnostics(prepared)
    diag.to_csv(C.PROCESSED / "signal_diagnostics.csv", index=False)
    bundle = modeling.train_and_select(prepared, verbose=True)
    modeling.save_bundle(bundle)
    summary["model"] = {"selected": bundle["model_name"], "holdout_r2": bundle["holdout_r2"]}
    print(f"[3/5] selected model: {bundle['model_name']}  (holdout R2={bundle['holdout_r2']:.3f})")

    # 4. clustering -----------------------------------------------------------------
    routes = clustering.cluster_routes(clustering.route_summary(prepared))
    routes.to_csv(C.PROCESSED / "route_clusters.csv", index=False)
    clustering.congestion_table(prepared).to_csv(C.PROCESSED / "region_product_congestion.csv", index=False)
    clustering.cluster_products(prepared).to_csv(C.PROCESSED / "product_clusters.csv", index=False)
    print(f"[4/5] routes clustered: {len(routes)}")

    # 5. simulation & recommendations -----------------------------------------------
    sim = ScenarioSimulator(prepared, bundle)
    scored = score_scenarios(sim.simulate(), speed_weight=0.5)
    scored.to_csv(C.PROCESSED / "scenario_simulation.csv", index=False)
    recs = recommend(scored, top_n=15)
    recs.to_csv(C.PROCESSED / "recommendations.csv", index=False)
    alternatives_per_product(scored, 3).to_csv(C.PROCESSED / "alternatives_per_product.csv", index=False)
    kpis = network_kpis(scored, recs)
    summary["kpis"] = kpis
    with open(C.PROCESSED / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("[5/5] recommendations:", len(recs))

    # 6. static figures for the report ---------------------------------------------
    from src import reporting
    reporting.make_all()
    print("[6/6] figures written to reports/figures")
    print(json.dumps(kpis, indent=2, default=float))


if __name__ == "__main__":
    main()
