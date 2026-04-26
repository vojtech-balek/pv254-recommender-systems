import polars as pl

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

    def fit(self, books_path):
        books_df = pl.scan_ndjson(books_path)


        valid_books = books_df.filter(
            pl.col("ratings_count").cast(pl.Int32, strict=False) >= self.min_ratings
        )

        self.popular_books = valid_books.sort(
            by=["average_rating", "ratings_count"],
            descending=[True, True]
        ).head(self.top_n).collect()

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

if __name__ == "__main__":
    DATA_PATH = "../processed-data/cleaned_books_fantasy_paranormal.json"
    baseline = BaselineRecommender(top_n=10, min_ratings=1000)

    print("Training Naive Baseline...")
    baseline.fit(DATA_PATH)

    print("\n--- Naive Baseline Recommendations for ANY user ---")
    print(baseline.recommend(user_id="eeb2537723b8382a4fd8d891d4a403a0"))
