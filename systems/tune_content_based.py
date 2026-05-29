import argparse
import csv
import itertools
from pathlib import Path

from content_based_improved import ContentBasedRecommender


DEFAULT_GRID = {
    "max_features": [5000, 10000],
    "min_df": [2, 5],
    "max_df": [0.9, 0.95],
    "ngram_range": [(1, 1), (1, 2)],
    "sublinear_tf": [True, False],
    "eps": [0.5, 0.7, 0.9],
    "min_samples": [3, 5],
}


def iter_grid(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def split_params(params):
    vectorizer_keys = {"max_features", "min_df", "max_df", "ngram_range", "sublinear_tf"}
    vectorizer_kwargs = {k: v for k, v in params.items() if k in vectorizer_keys}
    dbscan_kwargs = {k: v for k, v in params.items() if k not in vectorizer_keys}
    dbscan_kwargs["metric"] = "cosine"
    return vectorizer_kwargs, dbscan_kwargs


def main():
    parser = argparse.ArgumentParser(description="Grid search for content-based recommender.")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-combos", type=int, default=20, help="Limit the number of grid combos.")
    parser.add_argument("--score-metric", type=str, default="precision", choices=["precision", "hit_rate", "recall"])
    parser.add_argument("--output", type=str, default="tuning_results.csv")
    parser.add_argument("--top-n-results", type=int, default=5, help="Print the top N configs by score.")
    args = parser.parse_args()

    output_path = Path(args.output)
    combos = list(iter_grid(DEFAULT_GRID))
    if args.max_combos is not None:
        combos = combos[: args.max_combos]

    best = {"score": -1.0, "params": None}
    results = []

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "score_metric",
            "score",
            "hit_rate",
            "precision",
            "recall",
            "total_users",
            "max_features",
            "min_df",
            "max_df",
            "ngram_range",
            "sublinear_tf",
            "eps",
            "min_samples",
        ])

        for idx, params in enumerate(combos, start=1):
            vectorizer_kwargs, dbscan_kwargs = split_params(params)
            print(f"Running combo {idx}/{len(combos)}: {params}")

            recommender = ContentBasedRecommender(
                max_users=args.max_users,
                vectorizer_kwargs=vectorizer_kwargs,
                dbscan_kwargs=dbscan_kwargs,
            )
            recommender.build_tf_idf()
            profiles = recommender.build_user_profiles()
            metrics = recommender.evaluate(profiles, top_k=args.top_k)

            score = metrics.get(args.score_metric, 0.0)
            if score > best["score"]:
                best = {"score": score, "params": params}

            results.append({"score": score, "metrics": metrics, "params": params})

            writer.writerow([
                args.score_metric,
                f"{score:.6f}",
                f"{metrics['hit_rate']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall']:.6f}",
                metrics["total_users"],
                params["max_features"],
                params["min_df"],
                params["max_df"],
                params["ngram_range"],
                params["sublinear_tf"],
                params["eps"],
                params["min_samples"],
            ])

    print("Best params:")
    print(best)

    if args.top_n_results > 0:
        print(f"Top {args.top_n_results} configs by {args.score_metric}:")
        top_results = sorted(results, key=lambda r: r["score"], reverse=True)[: args.top_n_results]
        for i, row in enumerate(top_results, start=1):
            print(f"{i}. score={row['score']:.6f} params={row['params']}")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

