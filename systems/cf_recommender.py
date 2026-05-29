import json
from pathlib import Path
import random

import torch
from tqdm import tqdm

import polars as pl
import torch.nn.functional as F

import numpy as np

class CF_Recommender():
    """
    A collaborative filtering recommender that suggests books (items) based on
    user interactions and ratings.
    """

    def __init__(self, top_n=10, embedding_dim=64, objective="rating", use_bias=False):
        self.embedding_dim = embedding_dim
        self.top_n = top_n
        self.objective = objective
        self.use_bias = use_bias

        self.train_df = None
        self.user_rated_items = {}

        self.enc2user = {}
        self.enc2item = {}

        self.W_user = None
        self.W_item = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"Initialized CF_Recommender with top_n={top_n}, embedding_dim={embedding_dim}, use_bias={use_bias}, device={self.device}"
        )

    @staticmethod
    def _norm_book_id(x):
        return "" if x is None else str(x)

    def _predict(self, user_ids, item_ids):
        """Pointwise predictions for (user, item) pairs."""
        scores = (self.W_user[user_ids] * self.W_item[item_ids]).sum(dim=1)
        if self.use_bias:
            scores = scores + self.user_bias[user_ids] + self.item_bias[item_ids] + self.global_mean
        return scores

    def fit(
        self,
        train_df,
        epochs=10,
        batch_size=8192,
        lr=1e-3,
        lambda_reg=1e-4,
        models_dir=None,
    ):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        print(f"Using device: {self.device}")

        self.train_df = train_df

        user_ids = train_df["user_id_enc"].to_numpy()
        item_ids = train_df["work_id_enc"].to_numpy()
        ratings = train_df["rating"].to_numpy() / 5.0

        self.enc2user = dict(train_df.select("user_id_enc", "user_id").unique().iter_rows())
        self.enc2item = dict(train_df.select("work_id_enc", "work_id").unique().iter_rows())

        user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)
        ratings = torch.tensor(ratings, dtype=torch.float32, device=self.device)

        num_users = int(user_ids.max().item()) + 1
        num_items = int(item_ids.max().item()) + 1
        n = len(ratings)

        self.W_user = (
            torch.randn(num_users, self.embedding_dim, device=self.device) * 0.1
        ).requires_grad_(True)
        self.W_item = (
            torch.randn(num_items, self.embedding_dim, device=self.device) * 0.1
        ).requires_grad_(True)

        params = [self.W_user, self.W_item]

        if self.use_bias:
            self.user_bias = torch.zeros(num_users, device=self.device, requires_grad=True)
            self.item_bias = torch.zeros(num_items, device=self.device, requires_grad=True)
            self.global_mean = ratings.mean().item()
            params += [self.user_bias, self.item_bias]

        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*( n // batch_size + 1))

        for epoch in range(epochs):

            idx_schedule = torch.randperm(n, device=self.device)
            total_loss, n_batches = 0.0, 0

            progress_bar = tqdm(
                range(0, n, batch_size), desc=f"Epoch {epoch+1}/{epochs}"
            )
            for start in progress_bar:
                idx = idx_schedule[start : start + batch_size]
                u = user_ids[idx]
                i = item_ids[idx]
                r = ratings[idx]
                
                if self.objective == "read_bool":
                    pos_mask = r >= 0.6
                    if pos_mask.any():
                        pos_u, pos_i = u[pos_mask], i[pos_mask]
                        pos_scores = torch.sigmoid(self._predict(pos_u, pos_i))
                        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
                        
                        neg_items = torch.randint(0, self.W_item.shape[0], pos_u.shape, device=self.device)
                        neg_scores = torch.sigmoid(self._predict(pos_u, neg_items))
                        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
                        
                        loss = (pos_loss + neg_loss) / 2
                    else:
                        continue
                else:
                    scores = self._predict(u, i)
                    loss = ((scores - r) ** 2).mean()

                loss += lambda_reg * (
                    self.W_user[u].norm(dim=1).pow(2).mean()
                    + self.W_item[i].norm(dim=1).pow(2).mean()
                )
                if self.use_bias:
                    loss += lambda_reg * (
                        self.user_bias[u].norm().pow(2).mean()
                        + self.item_bias[i].norm().pow(2).mean()
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                if n_batches % 20 == 0:
                    progress_bar.set_postfix(loss=f"{total_loss / n_batches:.4f}")                

            scheduler.step()

            print(f"Epoch {epoch+1}: Avg Loss = {total_loss / n_batches:.4f}")
        
        self.user_rated_items = {
            row["user_id_enc"]: set(row["work_id_enc"])
            for row in train_df.group_by("user_id_enc")
            .agg(pl.col("work_id_enc"))
            .to_dicts()
        }

    def recommend(self, user_id, item_ids):
        scores = self._predict(user_id, item_ids)
        exclude = self.user_rated_items.get(user_id, set())
        if exclude:
            for idx, item in enumerate(item_ids.tolist()):
                if item in exclude:
                    scores[idx] = float('-inf')
        top_indices = torch.topk(scores, self.top_n).indices
        return [item_ids[i] for i in top_indices.tolist()]

    def _mask_rated(self, scores, user_ids_enc):
        for i, user_id_enc in enumerate(user_ids_enc.tolist()):
            exclude = self.user_rated_items.get(user_id_enc, set())
            if exclude:
                scores[i, list(exclude)] = float('-inf')
        return scores

    def recommend_batch(self, user_ids_enc, item_ids_enc):
        user_ids_enc = torch.as_tensor(user_ids_enc, device=self.device)
        item_ids_enc = torch.as_tensor(item_ids_enc, device=self.device)

        user_embeds = self.W_user[user_ids_enc]
        item_embeds = self.W_item[item_ids_enc]

        scores = user_embeds @ item_embeds.T
        if self.use_bias:
            scores = scores + self.user_bias[user_ids_enc].unsqueeze(1)
            scores = scores + self.item_bias[item_ids_enc].unsqueeze(0)
            scores = scores + self.global_mean

        scores = self._mask_rated(scores, user_ids_enc)

        top_indices = torch.topk(scores, self.top_n, dim=1).indices
        return item_ids_enc[top_indices].cpu().tolist()
    
    def _build_export_record(self, user_id_enc, true_enc, recommended_enc, num_hits, book_info):
        def _get_metadata(book_id):
            book_id = str(book_id)
            info = book_info.get(book_id, {})

            authors = info.get("author_names", [])
            if hasattr(authors, "tolist"):
                authors = authors.tolist()
            elif not isinstance(authors, list):
                authors = list(authors) if authors else []

            return {
                "work_id": book_id,
                "title": info.get("title", "Unknown"),
                "author_names": authors,
                "description": info.get("description", ""),
            }

        return {
            "user_id": self.enc2user.get(user_id_enc, user_id_enc),
            "true_books": [_get_metadata(self.enc2item.get(e, e)) for e in true_enc],
            "recommended": [_get_metadata(self.enc2item.get(e, e)) for e in recommended_enc],
            "num_hits": num_hits,
        }

    def evaluate(self, test_df, top_k=1, export_path=None, book_info=None):
        print(f"Evaluating on test set (top_{top_k})...")

        users_enc = torch.tensor(test_df["user_id_enc"].to_list(), device=self.device)
        items_enc = torch.tensor(test_df["work_id_enc"].to_list(), device=self.device)
        ratings = torch.tensor(test_df["rating"].to_list(), dtype=torch.float32, device=self.device) / 5.0

        with torch.no_grad():
            preds = self._predict(users_enc, items_enc)
            preds = torch.clamp(preds, 0.0, 1.0)
            rmse = torch.sqrt(torch.mean((preds - ratings) ** 2)).item()

        grouped = test_df.group_by("user_id_enc").agg(pl.col("work_id_enc")).to_dicts()
        test_user_books_enc = {row["user_id_enc"]: set(row["work_id_enc"]) for row in grouped}

        hits = 0
        precision_sum = 0.0
        recall_sum = 0.0
        total = 0
        
        export_data = {"good_recommendations": [], "bad_recommendations": []}

        batch_size = 8192
        item_ids_enc = torch.arange(self.W_item.shape[0], device=self.device)      

        for batch_start in tqdm(range(0, len(test_user_books_enc), batch_size), desc="Evaluating batches"):
            batch_user_ids_enc = list(test_user_books_enc.keys())[batch_start:batch_start + batch_size]
            batch_recommendations_enc = self.recommend_batch(batch_user_ids_enc, item_ids_enc)

            for user_id_enc, recommended_enc in zip(batch_user_ids_enc, batch_recommendations_enc):
                true_enc = test_user_books_enc[user_id_enc]
                num_hits = len(set(recommended_enc) & true_enc)

                hits += int(num_hits > 0)
                precision_sum += num_hits / top_k
                recall_sum += num_hits / len(true_enc) if true_enc else 0.0
                total += 1

                if export_path and book_info:
                    bucket = "good_recommendations" if num_hits > 0 else "bad_recommendations"
                    if len(export_data[bucket]) < 10:
                        export_data[bucket].append(self._build_export_record(
                            user_id_enc, true_enc, recommended_enc, num_hits, book_info
                        ))

        if total == 0:
            print("No users to evaluate.")
            return None

        if export_path:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4)
            print(f"Exported some evaluations to {export_path}")

        metrics = {
            "rmse_normalized": rmse,
            "rmse": rmse * 5,
            "hit_rate": hits / total,
            "precision": precision_sum / total,
            "recall": recall_sum / total,
        }
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return metrics
    
    def showcase_user(self, user_id_enc, test_df, book_info=None):
        def _fmt(enc_id):
            work_id = str(self.enc2item.get(enc_id, enc_id))
            if book_info and work_id in book_info:
                info = book_info[work_id]
                return {"work_id": work_id, "title": info.get("title", "?"), "authors": info.get("author_names", [])}
            return {"work_id": work_id, "title": "?", "authors": []}

        user_train = self.train_df.filter(pl.col("user_id_enc") == user_id_enc)
        top_rated = user_train.sort("rating", descending=True).head(5)

        item_ids = torch.arange(self.W_item.shape[0], device=self.device)
        recommended = self.recommend(user_id_enc, item_ids)

        user_test = test_df.filter(pl.col("user_id_enc") == user_id_enc)
        test_items = set(user_test["work_id_enc"].to_list())
        hits = set(recommended) & test_items

        result = {
            "user_id": self.enc2user.get(user_id_enc, user_id_enc),
            "top_rated_train": [
                {"rating": row["rating"], **_fmt(row["work_id_enc"])}
                for row in top_rated.iter_rows(named=True)
            ],
            "recommended": [
                {**_fmt(r), "hit": r in test_items}
                for r in recommended
            ],
            "test_books": [_fmt(t) for t in list(test_items)[:10]],
            "hits": len(hits),
        }

        print(f"\n=== User {self.enc2user.get(user_id_enc, user_id_enc)} ===")
        print(f"Top trained: {[f'{r['rating']}★ {_fmt(r['work_id_enc'])['title']}' for r in top_rated.iter_rows(named=True)]}")
        print(f"Recommended: {[_fmt(r)['title'] + (' ✓' if r in test_items else '') for r in recommended]}")
        print(f"Hits: {len(hits)}/{self.top_n}")

        return result
        
