import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        self.vectorizer = TfidfVectorizer(max_features=15000)
        self.book_ids = None
        self.books_df = pl.scan_ndjson('../processed-data/processed_books_texts.json')
        self.train_df = pl.scan_ndjson('../processed-data/train_interactions_fantasy_paranormal.json')
        self.test_df = pl.scan_ndjson('../processed-data/test_interactions_fantasy_paranormal.json')

    def build_tf_idf(self):
        pdf = self.books_df.select(["book_id", "combined_text"]).collect().to_pandas()
        pdf = pdf.dropna(subset=["combined_text"])
        pdf = pdf.set_index("book_id")

        self.book_ids = pdf.index.values
        self.book_id_to_idx = {str(book_id): idx for idx, book_id in enumerate(self.book_ids)}
        self.book_tf_idf = self.vectorizer.fit_transform(pdf["combined_text"])
        print(f"TF-IDF matrix built with shape: {self.book_tf_idf.shape}")

    def build_user_profiles(self):
        grouped_train = self.train_df.group_by('user_id').agg(pl.col('book_id'))

        # Limit the number of users to avoid OOM
        if self.max_users is not None:
            grouped_train = grouped_train.head(self.max_users)

        users = grouped_train.collect()
        user_profiles = {}
        for row in users.iter_rows(named=True):
            user_id = row[ 'user_id']
            books = row['book_id']
            valid_indices = [self.book_id_to_idx[b] for b in books if b in self.book_id_to_idx]
            if valid_indices:
                user_books_tfidf = self.book_tf_idf[valid_indices]
                user_profile_vector = self._aggregate(user_books_tfidf)
                user_profiles[user_id] = user_profile_vector


        print("Successfully built user profiles!")
        return user_profiles

    @staticmethod
    def _aggregate(user_books_tfidf):
        user_profile_matrix = user_books_tfidf.mean(axis=0)
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
            book_id = str(self.book_ids[idx])
            if book_id not in read_books:
                recommended.append(book_id)
                if len(recommended) == top_n:
                    break

        return recommended

    def evaluate(self, user_profiles, top_k=5):
        print("Evaluating on test set...")
        grouped_test = self.test_df.group_by('user_id').agg(pl.col('book_id')).collect()
        test_user_books = {row['user_id']: set(map(str, row['book_id'])) for row in grouped_test.iter_rows(named=True)}

        grouped_train = self.train_df.group_by('user_id').agg(pl.col('book_id')).collect()
        train_user_books = {row['user_id']: set(map(str, row['book_id'])) for row in grouped_train.iter_rows(named=True)}

        precisions = []
        recalls = []

        for user_id, true_books in test_user_books.items():
            if user_id not in user_profiles:
                continue

            recommended = self.recommend(user_id, user_profiles, train_user_books, top_n=top_k)

            if not recommended:
                continue

            hits = len(set(recommended).intersection(true_books))
            precisions.append(hits / len(recommended))
            recalls.append(hits / len(true_books))

        print(f"Precision@{top_k}: {np.mean(precisions):.4f}")
        print(f"Recall@{top_k}: {np.mean(recalls):.4f}")


if __name__ == "__main__":
    # We set max_users to limit RAM usage
    recommender = ContentBasedRecommender(max_users=1000)
    recommender.build_tf_idf()
    user_profiles = recommender.build_user_profiles()
    recommender.evaluate(user_profiles, top_k=5)
