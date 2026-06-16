from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def compute_binary_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
    }
