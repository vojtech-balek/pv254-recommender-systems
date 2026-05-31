import argparse
from pathlib import Path
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import polars as pl

from data_processing.cf_prep import prepare_cf_data
from systems.baseline_naive import BaselineRecommender
from systems.content_based_baseline import ContentBasedRecommender as ContentBasedBaseline
from systems.content_based_improved import ContentBasedRecommender as ContentBasedImproved
from systems.cf_recommender import CF_Recommender


METRICS = ["hit_rate", "precision", "recall"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_cf_data(train_path, test_path, output_train_path, output_test_path, redo):
    if redo or (not output_train_path.exists()) or (not output_test_path.exists()):
        prepare_cf_data(
            train_data_path=str(train_path),
            test_data_path=str(test_path),
            output_train_path=str(output_train_path),
            output_test_path=str(output_test_path),
        )

    train_df = pl.read_parquet(output_train_path)
    test_df = pl.read_parquet(output_test_path)
    return train_df, test_df


def run_baseline(books_path, train_path, test_path, top_k, min_ratings):
    model = BaselineRecommender(top_n=top_k, min_ratings=min_ratings)
    model.fit(str(books_path))
    metrics = model.evaluate(str(train_path), str(test_path), top_k=top_k)
    return metrics


def run_content_based_baseline(top_k, max_users):
    model = ContentBasedBaseline(max_users=max_users)
    model.build_tf_idf()
    profiles = model.build_user_profiles()
    metrics = model.evaluate(profiles, top_k=top_k)
    return metrics


def run_content_based_improved(top_k, max_users):
    params = {
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.9,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "eps": 0.7,
        "min_samples": 5,
    }
    vectorizer_kwargs = {k: v for k, v in params.items() if k in {
        "max_features", "min_df", "max_df", "ngram_range", "sublinear_tf"
    }}
    dbscan_kwargs = {k: v for k, v in params.items() if k in {"eps", "min_samples"}}
    dbscan_kwargs["metric"] = "cosine"
    model = ContentBasedImproved(
        max_users=max_users,
        vectorizer_kwargs=vectorizer_kwargs,
        dbscan_kwargs=dbscan_kwargs,
    )
    model.build_tf_idf()
    profiles = model.build_user_profiles()
    metrics = model.evaluate(profiles, top_k=top_k)
    return metrics


def run_cf(train_df, test_df, top_k, embedding_dim, epochs, objective, use_bias):
    model = CF_Recommender(
        top_n=top_k,
        embedding_dim=embedding_dim,
        objective=objective,
        use_bias=use_bias,
    )
    model.fit(train_df, epochs=epochs)
    metrics = model.evaluate(test_df, top_k=top_k)
    return metrics


def plot_metrics(results, output_path):
    model_names = [name for name in results.keys() if name != "baseline"]
    baseline_metrics = results.get("baseline", {})
    all_values = []
    for metrics in results.values():
        all_values.extend([metrics.get(m, 0.0) for m in METRICS])
    max_value = max(all_values) if all_values else 0.0
    y_max = min(1.0, max(0.1, max_value * 1.25))

    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    model_labels = [name.replace("_", " ").title() for name in model_names]
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(model_names)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        values = [results[name].get(metric, 0.0) for name in model_names]

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        bars = ax.bar(
            model_labels,
            values,
            color=[color_map[name] for name in model_names],
            edgecolor="#1f1f1f",
            linewidth=0.6,
        )

        baseline_value = baseline_metrics.get(metric)
        if baseline_value is not None:
            ax.axhline(
                baseline_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Naïve Baseline",
                alpha=0.9,
            )

        ax.set_title(f"{metric.replace('_', ' ').title()} (Higher is Better)")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", frameon=False)

        fig.tight_layout()

        metric_path = output_path.with_name(
            f"{output_path.stem}_{metric}{output_path.suffix}"
        )
        fig.savefig(metric_path, dpi=160)
        print(f"Saved plot to {metric_path}")
        plt.close(fig)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description="Compare recommenders and plot performance.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--min-ratings", type=int, default=1000)
    parser.add_argument("--cf-epochs", type=int, default=5)
    parser.add_argument("--cf-embedding-dim", type=int, default=8)
    parser.add_argument("--cf-objective", type=str, default="read_bool", choices=["rating", "read_bool"])
    parser.add_argument("--cf-use-bias", action="store_true")
    parser.add_argument("--redo-cf-prep", action="store_true")
    parser.add_argument("--output", type=str, default="evaluation_results/model_comparison.png")
    args = parser.parse_args()

    root = _repo_root()
    processed_dir = root / "processed-data"

    books_path = processed_dir / "cleaned_books_fantasy_paranormal.json"
    train_path = processed_dir / "train_interactions_fantasy_paranormal.json"
    test_path = processed_dir / "test_interactions_fantasy_paranormal.json"

    cf_train_path = processed_dir / "cf_data_train.parquet"
    cf_test_path = processed_dir / "cf_data_test.parquet"

    results = {}

    print("\n=== Baseline (most read) ===")
    results["baseline"] = run_baseline(
        books_path,
        train_path,
        test_path,
        top_k=args.top_k,
        min_ratings=args.min_ratings,
    )

    print("\n=== Content-Based Baseline ===")
    results["content_based_baseline"] = run_content_based_baseline(
        top_k=args.top_k,
        max_users=args.max_users,
    )

    print("\n=== Content-Based Improved ===")
    results["content_based_improved"] = run_content_based_improved(
        top_k=args.top_k,
        max_users=args.max_users,
    )

    print("\n=== Collaborative Filtering ===")
    train_df, test_df = _prepare_cf_data(
        train_path,
        test_path,
        cf_train_path,
        cf_test_path,
        redo=args.redo_cf_prep,
    )
    results["cf_recommender"] = run_cf(
        train_df,
        test_df,
        top_k=args.top_k,
        embedding_dim=args.cf_embedding_dim,
        epochs=args.cf_epochs,
        objective=args.cf_objective,
        use_bias=args.cf_use_bias,
    )

    plot_metrics(results, root / args.output)

    print("\nSummary")
    for name, metrics in results.items():
        metric_line = ", ".join(f"{m}={metrics.get(m, 0.0):.4f}" for m in METRICS)
        print(f"{name}: {metric_line}")


if __name__ == "__main__":
    main()
