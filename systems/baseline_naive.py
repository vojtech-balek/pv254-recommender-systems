import polars as pl
import json
from pathlib import Path

class BaselineRecommender:
    """
    A naive baseline recommender that suggests the most popular and
    highest-rated books to everyone. This does not personalize
    recommendations per user.
    """
    def __init__(self, top_n=10, min_ratings=100):
        self.top_n = top_n
        self.min_ratings = min_ratings
        self.popular_books = None

    @staticmethod
    def _norm_book_id(x):
        return "" if x is None else str(x)

    def fit(self, books_path):
        books_df = pl.scan_ndjson(books_path)


        valid_books = books_df.filter(
            pl.col("ratings_count").cast(pl.Int32, strict=False) >= self.min_ratings
        )

        self.popular_books = valid_books.sort(
            by=["ratings_count"],
            descending=[True]
        ).head(self.top_n).collect()

    def _get_recommendation_ids(self, top_n=None, exclude_ids=None):
        if self.popular_books is None:
            raise ValueError("Model must be fitted before making recommendations.")

        id_col = "work_id" if "work_id" in self.popular_books.columns else "book_id"
        if id_col not in self.popular_books.columns:
            raise ValueError("No work_id or book_id column available for evaluation.")

        exclude_ids = exclude_ids or set()
        ids = [self._norm_book_id(x) for x in self.popular_books[id_col].to_list()]
        filtered = [x for x in ids if x not in exclude_ids]

        return filtered[: (top_n or self.top_n)]

    def recommend(self, user_id=None):
        """
        Returns the top_n global recommendations.
        The user_id parameter is ignored since this is a global baseline.
        """
        if self.popular_books is None:
            raise ValueError("Model must be fitted before making recommendations.")

        cols_to_show = ["book_id", "title_without_series", "average_rating", "ratings_count"]
        available_cols = [c for c in cols_to_show if c in self.popular_books.columns]

        return self.popular_books.select(available_cols)

    def evaluate(self, train_path, test_path, top_k=10, export_path=None):
        print(f"Evaluating on test set (top_{top_k})...")

        train_df = pl.scan_ndjson(train_path).with_columns(pl.col("work_id").cast(pl.Utf8))
        test_df = pl.scan_ndjson(test_path).with_columns(pl.col("work_id").cast(pl.Utf8))

        grouped_train = train_df.group_by("user_id").agg(pl.col("work_id")).collect()
        grouped_test = test_df.group_by("user_id").agg(pl.col("work_id")).collect()

        train_user_books = {
            row["user_id"]: set(map(self._norm_book_id, row["work_id"]))
            for row in grouped_train.iter_rows(named=True)
        }
        test_user_books = {
            row["user_id"]: set(map(self._norm_book_id, row["work_id"]))
            for row in grouped_test.iter_rows(named=True)
        }

        hits = 0
        precision_sum = 0.0
        recall_sum = 0.0
        total = 0

        export_data = {"good_recommendations": [], "bad_recommendations": []}

        for user_id, true_books in test_user_books.items():
            recommended = self._get_recommendation_ids(
                top_n=top_k,
                exclude_ids=train_user_books.get(user_id, set())
            )

            num_hits = len(set(recommended).intersection(true_books)) if recommended else 0

            record = {
                "user_id": user_id,
                "recommended": recommended,
                "true_books": list(true_books),
                "num_hits": num_hits
            }

            if num_hits > 0:
                hits += 1
                if len(export_data["good_recommendations"]) < 10:
                    export_data["good_recommendations"].append(record)
            else:
                if len(export_data["bad_recommendations"]) < 10:
                    export_data["bad_recommendations"].append(record)

            precision_sum += num_hits / top_k
            recall_sum += num_hits / len(true_books) if len(true_books) > 0 else 0.0
            total += 1

        if total > 0:
            print(f"Hit Rate@{top_k}: {hits / total:.4f}")
            print(f"Precision@{top_k}: {precision_sum / total:.4f}")
            print(f"Recall@{top_k}: {recall_sum / total:.4f}")
        else:
            print("No users to evaluate.")

        if export_path and total > 0:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4)
            print(f"Exported some evaluations to {export_path}")

        return {
            "hit_rate": hits / total if total > 0 else 0.0,
            "precision": precision_sum / total if total > 0 else 0.0,
            "recall": recall_sum / total if total > 0 else 0.0,
            "total_users": total,
        }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    DATA_PATH = str(base_dir / "../processed-data/cleaned_books_fantasy_paranormal.json")
    TRAIN_PATH = str(base_dir / "../processed-data/train_interactions_fantasy_paranormal.json")
    TEST_PATH = str(base_dir / "../processed-data/test_interactions_fantasy_paranormal.json")

    baseline = BaselineRecommender(top_n=10, min_ratings=1000)

    print("Training Naive Baseline...")
    baseline.fit(DATA_PATH)

    print("\n--- Naive Baseline Recommendations for ANY user ---")
    print(baseline.recommend(user_id="eeb2537723b8382a4fd8d891d4a403a0"))

    baseline.evaluate(TRAIN_PATH, TEST_PATH, top_k=10, export_path="evaluation_examples.json")
