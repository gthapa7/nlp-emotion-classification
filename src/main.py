import argparse
import inspect
from pathlib import Path
import sys

import pandas as pd


IMPORTANT_MODELS = {
    "fc": "FC_BOW",
    "rnn": "RNN_LSTM",
    "transformer": "TRANSFORMER",
}


def get_model_module(name: str):
    if name == "fc":
        from src.models import fc

        return fc
    if name == "rnn":
        from src.models import rnn

        return rnn
    if name == "transformer":
        from src.models import transformer

        return transformer
    raise ValueError(f"Unknown model: {name}")


def print_best_metrics(metrics_path: Path) -> int:
    if not metrics_path.exists():
        print(f"No metrics found at {metrics_path}")
        return 1

    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df[metrics_df["model"].isin(IMPORTANT_MODELS.values())]
    if metrics_df.empty:
        print("No important models found in metrics.csv")
        return 1

    best_metrics = (
        metrics_df.sort_values("f1", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .sort_values("model")
    )

    print("Best run per important model:")
    print(best_metrics.to_string(index=False))
    return 0


def call_with_epochs(fn, data_dir: Path, results_path: Path, epochs: int | None) -> None:
    params = inspect.signature(fn).parameters
    if epochs is not None and "epochs" in params:
        fn(data_dir=data_dir, results_path=results_path, epochs=epochs)
    else:
        fn(data_dir=data_dir, results_path=results_path)


def run_train(
    model_name: str,
    data_dir: Path,
    results_path: Path,
    epochs: int | None,
) -> int:
    module = get_model_module(model_name)
    train_fn = getattr(module, "train", None)
    train_eval_fn = getattr(module, "train_and_evaluate", None)

    if callable(train_eval_fn):
        call_with_epochs(train_eval_fn, data_dir, results_path, epochs)
        return 0
    if callable(train_fn):
        call_with_epochs(train_fn, data_dir, results_path, epochs)
        return 0

    print(
        f"Model '{model_name}' has no train() or train_and_evaluate() in src/models/{model_name}.py"
    )
    return 1


def run_evaluate(
    model_name: str,
    data_dir: Path,
    results_path: Path,
    epochs: int | None,
) -> int:
    module = get_model_module(model_name)
    eval_fn = getattr(module, "evaluate", None)
    train_eval_fn = getattr(module, "train_and_evaluate", None)

    if callable(eval_fn):
        call_with_epochs(eval_fn, data_dir, results_path, epochs)
        return 0
    if callable(train_eval_fn):
        call_with_epochs(train_eval_fn, data_dir, results_path, epochs)
        return 0

    print(
        f"Model '{model_name}' has no evaluate() or train_and_evaluate() in src/models/{model_name}.py"
    )
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emotion classification pipeline")
    parser.add_argument(
        "--task",
        choices=["summary", "train", "evaluate"],
        default="summary",
        help="What to run",
    )
    parser.add_argument(
        "--model",
        choices=["fc", "rnn", "transformer"],
        default="transformer",
        help="Model to run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to the data folder",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results") / "metrics.csv",
        help="Path to metrics.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (if supported by model)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.task == "summary":
        return print_best_metrics(args.results_path)

    if args.task == "train":
        return run_train(args.model, args.data_dir, args.results_path, args.epochs)

    if args.task == "evaluate":
        return run_evaluate(args.model, args.data_dir, args.results_path, args.epochs)

    print(f"Unknown task: {args.task}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
