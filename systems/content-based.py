import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path

class ContentBasedRecommender:
    """
    A content-based recommender system
    """
    def __init__(self, top_n=10, min_ratings=10, max_users=None):
        self.top_n = top_n
        self.min_ratings = min_ratings
        self.max_users = max_users
        self.book_tf_idf = None
        self.book_id_to_idx = None
        self.book_info = {}
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.book_ids = None

        # Resolve paths relative to this file so running from repo root works.
        base_dir = Path(__file__).resolve().parent
        self.books_df = pl.scan_ndjson(str(base_dir / '../processed-data/processed_books_texts.json'))
        self.train_df = pl.scan_ndjson(str(base_dir / '../processed-data/train_interactions_fantasy_paranormal.json'))
        self.test_df = pl.scan_ndjson(str(base_dir / '../processed-data/test_interactions_fantasy_paranormal.json'))

    @staticmethod
    def _norm_book_id(x):
        # Use string consistently across the whole pipeline to avoid int/str mismatches
        return "" if x is None else str(x)

    def build_tf_idf(self):
        pdf = self.books_df.select(["work_id", "combined_text", "title", "author_names", "description"]).collect().to_pandas()
        pdf = pdf.dropna(subset=["combined_text"])

        # Normalize id type: treat book_id (work_id) as string everywhere
        pdf["work_id"] = pdf["work_id"].astype(str)
        pdf = pdf.set_index("work_id")

        self.book_ids = pdf.index.values
        self.book_id_to_idx = {self._norm_book_id(book_id): idx for idx, book_id in enumerate(self.book_ids)}

        # Save book metadata for evaluation exports (use string keys)
        meta = pdf[["title", "author_names", "description"]].to_dict('index')
        self.book_info = {self._norm_book_id(k): v for k, v in meta.items()}

        self.book_tf_idf = self.vectorizer.fit_transform(pdf["combined_text"])
        print(f"TF-IDF matrix built with shape: {self.book_tf_idf.shape}")

    def build_user_profiles(self):
        train_df = self.train_df.with_columns(pl.col("work_id").cast(pl.Utf8))
        grouped_train = train_df.group_by('user_id').agg([pl.col('work_id'), pl.col('rating')])

        # avoids OOM
        if self.max_users is not None:
            grouped_train = grouped_train.head(self.max_users)

        users = grouped_train.collect()
        user_profiles = {}
        for row in users.iter_rows(named=True):
            user_id = row['user_id']
            books = row['work_id']
            ratings = row['rating']

            valid_indices = []
            valid_ratings = []
            for b, r in zip(books, ratings):
                b = self._norm_book_id(b)
                if b in self.book_id_to_idx:
                    valid_indices.append(self.book_id_to_idx[b])
                    valid_ratings.append(r)

            if valid_indices:
                user_books_tfidf = self.book_tf_idf[valid_indices]
                user_profile_vector = self._aggregate(user_books_tfidf, valid_ratings)
                user_profiles[user_id] = user_profile_vector

        print("Successfully built user profiles!")
        return user_profiles

    @staticmethod
    def _aggregate(user_books_tfidf, ratings):

        ratings_array = np.array(ratings).reshape(-1, 1) / 5.0
        weighted_books = user_books_tfidf.multiply(ratings_array)
        user_profile_matrix = weighted_books.mean(axis=0)
        return np.asarray(user_profile_matrix).flatten()

    def recommend(self, user_id, user_profiles, train_user_books, top_n=5):
        if user_id not in user_profiles:
            return []

        user_profile = user_profiles[user_id].reshape(1, -1)
        scores = cosine_similarity(user_profile, self.book_tf_idf).flatten()

        # Sort descending
        top_indices = scores.argsort()[::-1]

        read_books = train_user_books.get(user_id, set())
        recommended = []

        for idx in top_indices:
            book_id = self._norm_book_id(self.book_ids[idx])
            if book_id not in read_books:
                recommended.append(book_id)
                if len(recommended) == top_n:
                    break

        return recommended

    def evaluate(self, user_profiles, top_k=1, export_path=None):
        print(f"Evaluating on test set (top_{top_k})...")

        test_df = self.test_df.with_columns(pl.col("work_id").cast(pl.Utf8))
        grouped_test = test_df.group_by('user_id').agg(pl.col('work_id')).collect()
        test_user_books = {row['user_id']: set(map(self._norm_book_id, row['work_id'])) for row in grouped_test.iter_rows(named=True)}

        train_df = self.train_df.with_columns(pl.col("work_id").cast(pl.Utf8))
        grouped_train = train_df.group_by('user_id').agg(pl.col('work_id')).collect()
        train_user_books = {row['user_id']: set(map(self._norm_book_id, row['work_id'])) for row in grouped_train.iter_rows(named=True)}

        hits = 0
        precision_sum = 0.0
        recall_sum = 0.0
        total = 0

        export_data = {"good_recommendations": [], "bad_recommendations": []}

        for user_id, true_books in test_user_books.items():
            if user_id not in user_profiles:
                continue

            recommended = self.recommend(user_id, user_profiles, train_user_books, top_n=top_k)

            if not recommended:
                continue

            num_hits = len(set(recommended).intersection(true_books))

            def _get_metadata(bid: str):
                bid = self._norm_book_id(bid)
                info = self.book_info.get(bid, {})

                authors = info.get("author_names", [])
                if hasattr(authors, "tolist"):
                    authors = authors.tolist()
                elif not isinstance(authors, list):
                    authors = list(authors) if authors else []

                return {
                    "work_id": bid,
                    "title": str(info.get("title", "Unknown")),
                    "author_names": authors,
                    "description": str(info.get("description", ""))
                }

            record = {
                "user_id": user_id,
                "profile_books": [_get_metadata(b) for b in train_user_books.get(user_id, set())],
                "recommended": [_get_metadata(b) for b in recommended],
                "true_books": [_get_metadata(b) for b in true_books],
                "num_hits": num_hits
            }

            if num_hits > 0:
                hits += 1
                if len(export_data["good_recommendations"]) < 10:
                    export_data["good_recommendations"].append(record)
            else:
                if len(export_data["bad_recommendations"]) < 10:
                    export_data["bad_recommendations"].append(record)

            precision_sum += num_hits / len(recommended)
            recall_sum += num_hits / len(true_books)
            total += 1

        if total > 0:
            print(f"Hit Rate@{top_k}: {hits / total:.4f}")
            print(f"Precision@{top_k}: {precision_sum / total:.4f}")
            print(f"Recall@{top_k}: {recall_sum / total:.4f}")
        else:
            print("No users to evaluate.")

        if export_path and total > 0:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4)
            print(f"Exported some evaluations to {export_path}")


if __name__ == "__main__":
    recommender = ContentBasedRecommender(max_users=1000)
    recommender.build_tf_idf()
    user_profiles = recommender.build_user_profiles()
    recommender.evaluate(user_profiles, top_k=10, export_path="evaluation_examples.json")
