from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import get_flattened_data
from metrics import compute_binary_metrics
from utils import save_json, set_seed



SEED = 123
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    set_seed(SEED)

    x_train, x_val, x_test, y_train, y_val, y_test = get_flattened_data()

    c_values = [0.01, 0.1, 1.0, 10.0]

    best_model = None
    best_c = None
    best_val_auc = -1
    validation_results = []

    for c in c_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=c,
                max_iter=1000,
                class_weight="balanced"
            )),
        ])

        model.fit(x_train, y_train)

        val_prob = model.predict_proba(x_val)[:, 1]
        val_pred = model.predict(x_val)
        val_metrics = compute_binary_metrics(y_val, val_pred, val_prob)
        val_metrics["log_loss"] = float(log_loss(y_val, val_prob))

        validation_results.append({"C": c, **val_metrics})

        print(f"C={c}, validation AUC={val_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_c = c
            best_model = model

    test_prob = best_model.predict_proba(x_test)[:, 1]
    test_pred = best_model.predict(x_test)
    test_metrics = compute_binary_metrics(y_test, test_pred, test_prob)
    test_metrics["log_loss"] = float(log_loss(y_test, test_prob))
    

    joblib.dump(best_model, OUTPUT_DIR / "baseline_logistic_regression.joblib")

    save_json({
        "model": "logistic_regression",
        "best_C": best_c,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
    }, OUTPUT_DIR / "baseline_metrics.json")

    print("Best C:", best_c)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()