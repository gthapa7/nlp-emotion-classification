from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred) -> dict:
	accuracy = accuracy_score(y_true, y_pred)
	precision, recall, f1, _ = precision_recall_fscore_support(
		y_true,
		y_pred,
		average="macro",
		zero_division=0,
	)
	return {
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
	}


def save_metrics_row(metrics: dict, model_name: str, results_path: Path) -> None:
	results_path = Path(results_path)
	results_path.parent.mkdir(parents=True, exist_ok=True)

	row = {
		"model": model_name,
		"accuracy": round(metrics.get("accuracy", 0.0), 4),
		"precision": round(metrics.get("precision", 0.0), 4),
		"recall": round(metrics.get("recall", 0.0), 4),
		"f1": round(metrics.get("f1", 0.0), 4),
		"timestamp": datetime.now().isoformat(timespec="seconds"),
	}

	metrics_df = pd.DataFrame([row])
	if results_path.exists():
		metrics_df.to_csv(results_path, mode="a", header=False, index=False)
	else:
		metrics_df.to_csv(results_path, index=False)
