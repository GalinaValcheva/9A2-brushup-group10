from src.metrics import compute_binary_metrics


def test_compute_binary_metrics_contains_expected_keys():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_prob = [0.1, 0.7, 0.8, 0.9]

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    expected_keys = {
        "accuracy",
        "f1",
        "auc",
        "sensitivity",
        "specificity",
        "confusion_matrix",
    }

    assert expected_keys.issubset(metrics.keys())
