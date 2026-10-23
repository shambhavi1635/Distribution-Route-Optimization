"""Step 3 - predictive modelling of shipping lead time.

Models: Linear Regression (baseline), Random Forest, Gradient Boosting.
Selection rule ("accuracy + interpretability"): take the lowest holdout RMSE, but if a
simpler model is within `tolerance` (default 2%) of that RMSE, prefer the simpler one.
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import config as C
from .features import CATEGORICAL, FEATURES, NUMERIC, TARGET, build_preprocessor

MODEL_PATH = C.MODELS / "best_model.joblib"

# higher = more interpretable
INTERPRETABILITY = {"Linear Regression": 3, "Gradient Boosting": 2, "Random Forest": 1}


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=C.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=C.RANDOM_STATE),
    }


def _pipe(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def signal_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """How much of the variance in lead time does each candidate driver explain?
    eta^2 for categoricals (between-group SS / total SS); |Spearman rho| for numerics."""
    y = df[TARGET]
    sst = ((y - y.mean()) ** 2).sum()
    rows = []
    for c in CATEGORICAL:
        grp = df.groupby(c)[TARGET]
        ssb = (grp.count() * (grp.mean() - y.mean()) ** 2).sum()
        rows.append({"driver": c, "type": "categorical", "variance_explained_pct": 100 * ssb / sst})
    for c in NUMERIC:
        rho = df[[c, TARGET]].corr(method="spearman").iloc[0, 1]
        rows.append({"driver": c, "type": "numeric (rho^2)", "variance_explained_pct": 100 * rho ** 2})
    return pd.DataFrame(rows).sort_values("variance_explained_pct", ascending=False).reset_index(drop=True)


def train_and_select(prepared: pd.DataFrame, tolerance: float = 0.02, verbose: bool = False):
    X, y = prepared[FEATURES], prepared[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=C.RANDOM_STATE)
    cv = KFold(5, shuffle=True, random_state=C.RANDOM_STATE)

    rows, fitted = [], {}
    for name, model in get_models().items():
        pipe = _pipe(model)
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "CV RMSE": cv_rmse,
            "Holdout RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "Holdout MAE": float(mean_absolute_error(y_te, pred)),
            "Holdout R2": float(r2_score(y_te, pred)),
        })
        fitted[name] = pipe
        if verbose:
            print(rows[-1])
    metrics = pd.DataFrame(rows)

    best_rmse = metrics["Holdout RMSE"].min()
    ok = metrics[metrics["Holdout RMSE"] <= best_rmse * (1 + tolerance)]
    chosen = max(ok["Model"], key=lambda m: INTERPRETABILITY[m])
    metrics["Selected"] = metrics["Model"] == chosen

    # permutation importance on held-out data for the selected (train-only) model
    pi = permutation_importance(fitted[chosen], X_te, y_te, n_repeats=10,
                                scoring="neg_root_mean_squared_error", random_state=C.RANDOM_STATE)
    importance = (pd.DataFrame({"feature": FEATURES, "rmse_increase": pi.importances_mean,
                                "std": pi.importances_std})
                  .sort_values("rmse_increase", ascending=False).reset_index(drop=True))

    # refit the selected model on all prepared data for deployment
    final = _pipe(get_models()[chosen]).fit(X, y)
    r2 = float(metrics.loc[metrics["Model"] == chosen, "Holdout R2"].iloc[0])
    bundle = {
        "pipeline": final,
        "model_name": chosen,
        "holdout_r2": r2,
        "metrics": metrics,
        "importance": importance,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
    }
    return bundle


def save_bundle(bundle: dict):
    C.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    bundle["metrics"].to_csv(C.PROCESSED / "model_comparison.csv", index=False)
    bundle["importance"].to_csv(C.PROCESSED / "feature_importance.csv", index=False)
    with open(C.MODELS / "model_summary.json", "w") as f:
        json.dump({"selected_model": bundle["model_name"], "holdout_r2": bundle["holdout_r2"],
                   "sklearn_version": bundle["sklearn_version"]}, f, indent=2)


def load_bundle(prepared: pd.DataFrame = None):
    """Load the saved model; if it is missing or incompatible with the installed
    scikit-learn, retrain from `prepared` (when given)."""
    try:
        bundle = joblib.load(MODEL_PATH)
        bundle["pipeline"].predict(prepared[FEATURES].head(2)) if prepared is not None else None
        return bundle
    except Exception:
        if prepared is None:
            raise
        bundle = train_and_select(prepared)
        try:
            save_bundle(bundle)
        except Exception:
            pass
        return bundle
