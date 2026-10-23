"""Regenerate and execute the stage notebooks (outputs embedded).

    python scripts/build_notebooks.py
Requires: nbformat, nbclient, ipykernel (see requirements.txt).
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
PRE = '''import sys
from pathlib import Path
ROOT = Path.cwd().resolve().parents[1]          # notebooks/<stage>/ -> project root
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np, matplotlib.pyplot as plt
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
from src import config as C'''

NOTEBOOKS = {
"data_cleaning_validation/01_data_cleaning_validation.ipynb": [
("md", "# 01 - Data cleaning & validation\nLoads the raw Nassau Candy orders, validates them and rebuilds a **corrected lead time**.\n\n**Finding:** raw `Ship Date - Order Date` is 900-1,640 days because each Order-ID-year cohort carries a constant offset. The offset is estimated from the cohort's Same Day orders (nominally 0 days) and removed."),
("code", PRE + "\nfrom src import data_prep"),
("code", "raw = data_prep.load_raw()\nprint(raw.shape)\nraw.head()"),
("code", "validated, report = data_prep.clean_and_validate(raw)\nreport"),
("code", "validated.groupby('Order Cohort')[['Raw Lead Time','Cohort Offset','Lead Time']].agg(['min','median','max']).round(1)"),
("code", "order = ['Same Day','First Class','Second Class','Standard Class']\nfig, ax = plt.subplots(1, 2, figsize=(10, 3.4))\nax[0].hist(validated['Raw Lead Time'], bins=60, color='#9aa3ad'); ax[0].set(title='Raw gap (days)')\nm = validated.groupby('Ship Mode')['Lead Time'].mean().reindex(order)\nax[1].bar(order, m.values, color='#2a6ebb'); ax[1].set(title='Corrected lead time by ship mode (days)')\nax[1].tick_params(axis='x', rotation=15); plt.show()\nm.round(2)"),
("code", "C.PROCESSED.mkdir(parents=True, exist_ok=True)\nvalidated.to_csv(C.PROCESSED / 'orders_validated.csv', index=False)\nvalidated[['Product Name','Origin Factory','State/Province','Distance Miles','Lead Time','Ship Mode']].head()"),
],
"feature_engineering/02_feature_engineering.ipynb": [
("md", "# 02 - Feature engineering\nExtreme-outlier removal (3 x IQR), categorical encoding (one-hot), numeric normalisation (z-score) and the training-ready feature matrix."),
("code", PRE + "\nfrom src import features"),
("code", "validated = pd.read_csv(C.PROCESSED / 'orders_validated.csv', parse_dates=['Order Date','Ship Date'])\nprepared, rep = features.prepare_dataset(validated)\nrep"),
("code", "print('Categorical:', features.CATEGORICAL); print('Numeric:', features.NUMERIC); print('Target:', features.TARGET)"),
("code", "X = features.export_feature_matrix(prepared)\nprint(X.shape)\nX.head()"),
("code", "prepared.to_csv(C.PROCESSED / 'orders_prepared.csv', index=False)\nX.to_csv(C.PROCESSED / 'feature_matrix.csv', index=False)"),
],
"predictive_modeling/03_predictive_modeling.ipynb": [
("md", "# 03 - Predictive modelling of lead time\nLinear Regression (baseline), Random Forest and Gradient Boosting; 80/20 split + 5-fold CV. Selection rule: lowest holdout RMSE, preferring the simpler model when within 2%.\n\nBefore modelling, a **signal diagnostic** shows how much of the lead-time variance each driver explains."),
("code", PRE + "\nfrom src import modeling"),
("code", "prepared = pd.read_csv(C.PROCESSED / 'orders_prepared.csv')\ndiag = modeling.signal_diagnostics(prepared)\ndiag.round(3)"),
("code", "diag.sort_values('variance_explained_pct').plot.barh(x='driver', y='variance_explained_pct', legend=False, color='#2a6ebb', figsize=(6,4), title='Variance explained (%)'); plt.show()"),
("code", "bundle = modeling.train_and_select(prepared)\nbundle['metrics'].round(3)"),
("code", "print('Selected:', bundle['model_name'], '| holdout R2 =', round(bundle['holdout_r2'], 3))\nbundle['importance'].round(4)"),
("md", "**Reading the result.** Ship mode explains ~two thirds of the variance; factory, region, product and distance each explain well under 1%. All three models therefore score almost identically (RMSE ~1.0 day), and the baseline linear model is selected. Because every product ships from exactly one factory historically, factory identity cannot be separated from product - so the simulator adds an explicit distance-driven transit/freight component (see notebook 05)."),
("code", "modeling.save_bundle(bundle)"),
],
"route_clustering/04_route_product_clustering.ipynb": [
("md", "# 04 - Route & product clustering\nKMeans on route performance (avg lead, lead variability, distance, volume) to find consistently slow routes and congested region-product combinations."),
("code", PRE + "\nfrom src import clustering"),
("code", "prepared = pd.read_csv(C.PROCESSED / 'orders_prepared.csv')\nroutes = clustering.cluster_routes(clustering.route_summary(prepared))\nroutes['Route Cluster'].value_counts()"),
("code", "routes[routes['Route Cluster']=='Slow'].sort_values('Avg_Lead', ascending=False).round(2)"),
("code", "colors = {'Fast':'#2a6ebb','Typical':'#9aa3ad','Slow':'#d9822b'}\nfig, ax = plt.subplots(figsize=(7,4))\nfor n, g in routes[routes['Route Cluster'].isin(colors)].groupby('Route Cluster'):\n    ax.scatter(g['Avg_Distance'], g['Avg_Lead'], s=np.sqrt(g['Orders'])*6, alpha=.7, color=colors[n], label=n)\nax.set(xlabel='avg distance (mi)', ylabel='avg lead (days)'); ax.legend(); plt.show()"),
("code", "cong = clustering.congestion_table(prepared)\ncong[cong['Congested']].round(2)"),
("code", "clustering.cluster_products(prepared).round(2)"),
("code", "routes.to_csv(C.PROCESSED / 'route_clusters.csv', index=False)\ncong.to_csv(C.PROCESSED / 'region_product_congestion.csv', index=False)\nclustering.cluster_products(prepared).to_csv(C.PROCESSED / 'product_clusters.csv', index=False)"),
],
"simulation_optimization/05_scenario_simulation_recommendations.ipynb": [
("md", "# 05 - Scenario simulation & recommendations\nFor every product, simulate shipping from each alternative factory, predict the new lead time, measure profit sensitivity, rank by lead-time reduction / risk reduction / profit impact, and generate top-N recommendations."),
("code", PRE + "\nfrom src import modeling\nfrom src.simulation import ScenarioSimulator\nfrom src.optimization import score_scenarios, recommend, alternatives_per_product, network_kpis"),
("code", "prepared = pd.read_csv(C.PROCESSED / 'orders_prepared.csv', parse_dates=['Order Date','Ship Date'])\nbundle = modeling.load_bundle(prepared)\nsim = ScenarioSimulator(prepared, bundle)\nscored = score_scenarios(sim.simulate(), speed_weight=0.5)\nscored[scored['Product']=='Wonka Bar - Milk Chocolate'][['Candidate Factory','Scenario Lead (days)','Lead Reduction %','Distance Reduction %','Profit Impact %','Confidence','Risk Level','Score']].round(2)"),
("code", "recs = recommend(scored, top_n=15)\nrecs[['Rank','Product','Current Factory','Candidate Factory','Orders','Lead Reduction %','Distance Reduction %','Profit Impact %','Profit Stability %','Confidence','Risk Level','Score']].round(2)"),
("code", "kpis = network_kpis(scored, recs)\npd.Series(kpis).round(2)"),
("md", "### Sensitivity to the speed-vs-profit slider"),
("code", "rows = []\nfor w in [0, .25, .5, .75, 1.0]:\n    s = score_scenarios(sim.simulate(), speed_weight=w); r = recommend(s, top_n=15); k = network_kpis(s, r)\n    rows.append({'speed weight': w, 'products recommended': k['Products Recommended'], 'network lead reduction %': round(k['Lead Time Reduction % (network)'], 2), 'network profit impact $': round(k['Network Profit Impact ($)'])})\npd.DataFrame(rows)"),
("code", "scored.to_csv(C.PROCESSED / 'scenario_simulation.csv', index=False)\nrecs.to_csv(C.PROCESSED / 'recommendations.csv', index=False)\nalternatives_per_product(scored, 3).to_csv(C.PROCESSED / 'alternatives_per_product.csv', index=False)"),
],
}

for rel, cells in NOTEBOOKS.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    path = ROOT / "notebooks" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbf.write(nb, path)
    print("built", path.relative_to(ROOT))
