import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
class ContentBasedRecommender:
    """
    A content-based recommender system
    """
    def __init__(self, top_n=10, min_ratings=10):
        self.top_n = top_n
        self.min_ratings = min_ratings
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
        return user_profiles

    def _aggregate(self, user_books_tfidf):
        user_profile_matrix = user_books_tfidf.mean(axis=0)
        return np.asarray(user_profile_matrix).flatten()

    def recommend(self, user_id=None):
        pass


if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.build_tf_idf()
    recommender.train()
