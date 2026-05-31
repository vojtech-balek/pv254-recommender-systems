import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
import json
from pathlib import Path



class ContentBasedRecommender:
    """
    A content-based recommender system
    """
    def __init__(self, top_n=10, min_ratings=10, max_users=None, vectorizer_kwargs=None, dbscan_kwargs=None):
        self.top_n = top_n
        self.min_ratings = min_ratings
        self.max_users = max_users
        self.book_tf_idf = None
        self.book_id_to_idx = None
        self.book_info = {}
        vectorizer_kwargs = vectorizer_kwargs or {}
        if "max_features" not in vectorizer_kwargs:
            vectorizer_kwargs["max_features"] = 10000
        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        self.dbscan_kwargs = dbscan_kwargs or {"eps": 0.7, "min_samples": 5, "metric": "cosine"}
        self.book_ids = None

        base_dir = Path(__file__).resolve().parent
        self.books_df = pl.scan_ndjson(str(base_dir / '../processed-data/processed_books_texts.json'))
        self.train_df = pl.scan_ndjson(str(base_dir / '../processed-data/train_interactions_fantasy_paranormal.json'))
        self.test_df = pl.scan_ndjson(str(base_dir / '../processed-data/test_interactions_fantasy_paranormal.json'))

    @staticmethod
    def _norm_book_id(x):
        return "" if x is None else str(x)

    def build_tf_idf(self):
        pdf = self.books_df.select(["work_id", "combined_text", "title", "author_names", "description"]).collect().to_pandas()
        pdf = pdf.dropna(subset=["combined_text"])

        pdf["work_id"] = pdf["work_id"].astype(str)
        pdf = pdf.sort_values("work_id")
        pdf = pdf.set_index("work_id")

        self.book_ids = pdf.index.values
        self.book_id_to_idx = {self._norm_book_id(book_id): idx for idx, book_id in enumerate(self.book_ids)}

        meta = pdf[["title", "author_names", "description"]].to_dict('index')
        self.book_info = {self._norm_book_id(k): v for k, v in meta.items()}

        self.book_tf_idf = self.vectorizer.fit_transform(pdf["combined_text"])
        print(f"TF-IDF matrix built with shape: {self.book_tf_idf.shape}")

    def build_user_profiles(self):
        train_df = self.train_df.with_columns(pl.col("work_id").cast(pl.Utf8))
        grouped_train = train_df.group_by('user_id').agg([pl.col('work_id'), pl.col('rating')])
        
        grouped_train = grouped_train.sort('user_id')

        if self.max_users is not None:
            grouped_train = grouped_train.head(self.max_users)

        users = grouped_train.collect()
        user_profiles = {}
        self.user_book_clusters = {}
        for row in users.iter_rows(named=True):
            user_id = row['user_id']
            books = row['work_id']
            ratings = row['rating']

            valid_indices = []
            valid_ratings = []
            valid_books = []
            for b, r in zip(books, ratings):
                if r >= 3:
                    b = self._norm_book_id(b)
                    if b in self.book_id_to_idx:
                        valid_indices.append(self.book_id_to_idx[b])
                        valid_ratings.append(r)
                        valid_books.append(b)

            if valid_indices:
                user_books_tfidf = self.book_tf_idf[valid_indices]
                cluster_profiles, labels = self._aggregate(user_books_tfidf, valid_ratings)
                user_profiles[user_id] = cluster_profiles
                self.user_book_clusters[user_id] = {b: int(lbl) for b, lbl in zip(valid_books, labels)}

        print("Successfully built user profiles!")
        return user_profiles

    def _aggregate(self, user_books_tfidf, ratings):
        clustering = DBSCAN(**self.dbscan_kwargs).fit(user_books_tfidf)
        labels = clustering.labels_
        unique_labels = set(labels)

        cluster_profiles = []
        ratings_array = np.array(ratings).reshape(-1, 1) / 5.0

        for k in unique_labels:
            class_member_mask = (labels == k)
            cluster_tfidf = user_books_tfidf[class_member_mask]
            cluster_ratings = ratings_array[class_member_mask]

            weighted_cluster_books = cluster_tfidf.multiply(cluster_ratings)
            cluster_profile = weighted_cluster_books.mean(axis=0)
            cluster_profiles.append(np.asarray(cluster_profile).flatten())

        if not cluster_profiles:
            weighted_books = user_books_tfidf.multiply(ratings_array)
            fallback_profile = weighted_books.mean(axis=0)
            cluster_profiles.append(np.asarray(fallback_profile).flatten())
            labels = np.zeros(len(ratings))

        return cluster_profiles, labels

    def recommend(self, user_id, user_profiles, train_user_books, top_n=5):
        if user_id not in user_profiles:
            return []

        cluster_profiles = user_profiles[user_id]
        
        all_cluster_scores = []
        for profile in cluster_profiles:
            profile = profile.reshape(1, -1)
            scores = cosine_similarity(profile, self.book_tf_idf).flatten()
            all_cluster_scores.append(scores)
            
        if not all_cluster_scores:
            return []

        read_books = train_user_books.get(user_id, set())
        recommended = []
        
        cluster_sorted_indices = [scores.argsort()[::-1] for scores in all_cluster_scores]
        pointers = [0] * len(cluster_sorted_indices)
        
        while len(recommended) < top_n:
            added_in_round = False
            for i, sorted_indices in enumerate(cluster_sorted_indices):
                if len(recommended) >= top_n:
                    break
                    
                while pointers[i] < len(sorted_indices):
                    idx = sorted_indices[pointers[i]]
                    pointers[i] += 1
                    book_id = self._norm_book_id(self.book_ids[idx])
                    
                    if book_id not in read_books and book_id not in recommended:
                        recommended.append(book_id)
                        added_in_round = True
                        break
                        
            if not added_in_round:
                break

        return recommended

    def evaluate(self, user_profiles, top_k=1, export_path=None):
        print(f"Evaluating on test set (top_{top_k})...")

        test_df = self.test_df.with_columns(pl.col("work_id").cast(pl.Utf8))
        grouped_test = test_df.group_by('user_id').agg(pl.col('work_id')).sort('user_id').collect()
        test_user_books = {row['user_id']: set(map(self._norm_book_id, row['work_id'])) for row in grouped_test.iter_rows(named=True)}

        train_df = self.train_df.with_columns(pl.col("work_id").cast(pl.Utf8))
        grouped_train = train_df.group_by('user_id').agg(pl.col('work_id')).sort('user_id').collect()
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

            num_hits = len(set(recommended).intersection(true_books)) if recommended else 0

            def _get_metadata(bid: str):
                bid = self._norm_book_id(bid)
                info = self.book_info.get(bid, {})

                authors = info.get("author_names", [])
                if hasattr(authors, "tolist"):
                    authors = authors.tolist()
                elif not isinstance(authors, list):
                    authors = list(authors) if authors else []

                meta = {
                    "work_id": bid,
                    "title": str(info.get("title", "Unknown")),
                    "author_names": authors,
                    "description": str(info.get("description", ""))
                }

                if hasattr(self, 'user_book_clusters') and user_id in self.user_book_clusters:
                    if bid in self.user_book_clusters[user_id]:
                        meta["cluster"] = self.user_book_clusters[user_id][bid]

                return meta

            record = {
                "user_id": user_id,
                "profile_books": [_get_metadata(b) for b in sorted(train_user_books.get(user_id, set()))],
                "recommended": [_get_metadata(b) for b in recommended],
                "true_books": [_get_metadata(b) for b in sorted(true_books)],
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
            hit_rate = hits / total
            precision = precision_sum / total
            recall = recall_sum / total
            print(f"Hit Rate@{top_k}: {hit_rate:.4f}")
            print(f"Precision@{top_k}: {precision:.4f}")
            print(f"Recall@{top_k}: {recall:.4f}")
        else:
            hit_rate = 0.0
            precision = 0.0
            recall = 0.0
            print("No users to evaluate.")

        if export_path and total > 0:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4)
            print(f"Exported some evaluations to {export_path}")

        return {"hit_rate": hit_rate, "precision": precision, "recall": recall, "total_users": total}


if __name__ == "__main__":
    recommender = ContentBasedRecommender(max_users=1000, vectorizer_kwargs={'max_features': 5000, 'min_df': 2, 'max_df': 0.9, 'ngram_range': (1, 2), 'sublinear_tf': True})
    recommender.build_tf_idf()
    user_profiles = recommender.build_user_profiles()
    recommender.evaluate(user_profiles, top_k=10, export_path="evaluation_examples.json")
