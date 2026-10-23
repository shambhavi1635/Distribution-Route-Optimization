# Factory Reallocation & Shipping Optimization Recommendation System
### Nassau Candy Distributor - Unified Mentor Project 2

Decision-intelligence system that **predicts shipping lead time under different factory configurations, recommends which products should move to which factory, and balances speed against profit**. It goes beyond descriptive analytics (Project 1) to *what should we change?*

| Deliverable (from the brief) | Where it is |
|---|---|
| Research paper (EDA, insights, recommendations) | `reports/research_paper.md` / `.docx` |
| Streamlit dashboard (live analytics) | `app.py` |
| Executive summary for government stakeholders | `reports/executive_summary.md` / `.docx` |

---

## 1. Quick start

**Windows (PowerShell / VS Code terminal)**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # optional - artefacts are already included
streamlit run app.py
```

**macOS / Linux**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py          # optional - artefacts are already included
streamlit run app.py
```

`python check_dashboard.py` runs a headless smoke test of the whole app.

> The saved model (`models/best_model.joblib`) was trained with scikit-learn 1.9. If your version differs and it cannot be loaded, the app silently retrains it in a few seconds - no action needed.

---

## 2. Project structure (mirrors Project 1)

```
nassau_factory_optimization/
├── data/
│   ├── raw/Nassau Candy Distributor.csv
│   └── processed/
│       ├── orders_validated.csv          # cleaned + corrected lead time + distances
│       ├── orders_prepared.csv           # after outlier removal (modelling table)
│       ├── feature_matrix.csv            # encoded + normalised, training-ready
│       ├── signal_diagnostics.csv        # variance explained by each driver
│       ├── model_comparison.csv  feature_importance.csv
│       ├── route_clusters.csv  region_product_congestion.csv  product_clusters.csv
│       ├── scenario_simulation.csv       # every product x factory scenario
│       ├── recommendations.csv  alternatives_per_product.csv
│       └── summary.json
├── notebooks/
│   ├── data_cleaning_validation/01_data_cleaning_validation.ipynb
│   ├── feature_engineering/02_feature_engineering.ipynb
│   ├── predictive_modeling/03_predictive_modeling.ipynb
│   ├── route_clustering/04_route_product_clustering.ipynb
│   └── simulation_optimization/05_scenario_simulation_recommendations.ipynb
├── src/
│   ├── config.py        # factories, product->factory map, state coordinates, assumptions
│   ├── data_prep.py     # load, validate, corrected lead time, distances
│   ├── features.py      # outliers, encoding, scaling
│   ├── modeling.py      # Linear / Random Forest / Gradient Boosting, selection, importance
│   ├── clustering.py    # route & product clustering, congestion
│   ├── simulation.py    # scenario simulation engine
│   ├── optimization.py  # scoring, ranking, top-N, KPIs
│   └── reporting.py     # static figures
├── models/              # best_model.joblib, model_summary.json
├── reports/             # research paper, executive summary, figures/
├── scripts/build_notebooks.py
├── app.py               # Streamlit dashboard
├── run_pipeline.py      # end-to-end run
├── check_dashboard.py   # smoke test
├── requirements.txt  .gitignore  .streamlit/config.toml
└── README.md
```

---

## 3. How the brief maps to the code

| Brief | Implementation |
|---|---|
| Data preparation & encoding | `features.py`: 3xIQR extreme-outlier removal (Lead Time global; Sales/Units per product), z-score scaling, one-hot encoding, `feature_matrix.csv` |
| Predictive modelling | `modeling.py`: lead time from product, origin factory, destination region, ship mode (+ distance, units, sales, month); Linear (baseline), Random Forest, Gradient Boosting |
| Model evaluation | RMSE, MAE, R2 on a 20% holdout + 5-fold CV; best model = lowest RMSE, simpler model preferred if within 2% |
| Route & product clustering | `clustering.py`: KMeans on routes (factory -> state) -> Fast / Typical / Slow; congested region-product combinations; product clusters |
| Scenario simulation engine | `simulation.py`: every product x every factory -> new lead time, distance, profit sensitivity, confidence |
| Optimization & recommendation logic | `optimization.py`: ranks by lead-time reduction, risk reduction, profit impact; top-N recommendations |
| KPIs | Lead Time Reduction %, Profit Impact Stability, Scenario Confidence Score, Recommendation Coverage - shown at the top of the app |
| Streamlit modules | 4 required tabs (Simulator, What-If, Recommendations, Risk & Impact) + a Model & Route Insights tab |
| User capabilities | Product selector, region selector, ship-mode filter, speed-vs-profit slider (all in the sidebar) |

---

## 4. Three things you must know (and mention to your mentor)

**1. The raw lead time is not usable as-is.** `Ship Date - Order Date` is 904-1,642 days. The gap is a constant per Order-ID-year cohort (904 / 1,269 / 1,269 / 1,635 days for Same Day orders), i.e. the Ship Dates were shifted when the dataset was assembled. We estimate each cohort's offset from its Same Day orders (nominally 0 days) and subtract it. The corrected lead times land on 0.0 / 2.2 / 3.2 / 5.0 days for Same Day / First / Second / Standard Class - exactly what those services promise. This is an inference from the data, not documented, so state it as an assumption.

**2. Ship mode is the only real driver of lead time.** It explains ~66% of the variance; product, factory, region, division and distance each explain under 0.2%. So all three models score almost the same (RMSE ~1.0 day, R2 ~0.67) and the linear baseline is selected on interpretability. The model cannot, by itself, rank factories - and because each product has only ever shipped from one factory, factory and product cannot be separated in the data.

**3. Therefore the simulator adds an explicit, editable distance model.** End-to-end lead time = ML prediction + transit days per 1,000 miles (default 0.6, scaled by ship mode); profit impact = change in freight cost (default $0.08 per unit per 1,000 miles, scaled by ship mode). Neither is in the dataset - they are assumptions, editable in the sidebar (*Assumptions*). Production cost, capacity and factory-specific costs are also unknown and treated as equal across factories. Read the recommendations as *"what the geography implies"*, to be piloted, not as proven savings.

---

## 5. Headline results (default settings: 50/50 speed vs profit, top-15)

* **Consolidate production towards the Secret Factory** (central US). 12 of 15 products get a reassignment; the other 3 already ship from it. Covers ~98% of orders.
* Average shipping distance of the moved products falls ~26%; **network lead time -5.8%**; profit **+$1.2k (+1.3% of gross profit)** from lower freight.
* Confidence 72/100. All 5 core Wonka bars are medium/low risk. The tiny candy lines (3-10 orders) are flagged **High risk** purely for thin evidence - do not act on those.
* Moving chocolate to a factory that has never made it is an extrapolation (lower confidence by design) - **pilot one bar first**.

---

## 6. Re-running things

```bash
python run_pipeline.py                 # data -> features -> models -> clusters -> simulation -> figures
python scripts/build_notebooks.py      # rebuild + execute the 5 notebooks
python check_dashboard.py              # smoke-test the app
```

## 7. Tech
Python 3.10+, pandas, NumPy, scikit-learn, Plotly, Streamlit, matplotlib, joblib.
"# Distribution-Route-Optimization" 
