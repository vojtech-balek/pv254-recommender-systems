from pathlib import Path

import torch
from tqdm import tqdm

class CF_Recommender:
    """
    A collaborative filtering recommender that suggests books based on
    user interactions and ratings.
    """

    def __init__(self, embedding_dim=64, top_n=10):
        self.embedding_dim = embedding_dim
        self.top_n = top_n

        self.W_user = None
        self.W_book = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Initialized CF_Recommender with embedding_dim={embedding_dim}, top_n={top_n}, device={self.device}')


    def load(self, models_dir):
        models_dir = Path(models_dir)
        model = torch.load(models_dir / "cf_model.pt", map_location=self.device)

        self.W_user = model["W_user"]
        self.W_book = model["W_book"]

        print(f"Loaded model from {models_dir / 'cf_model.pt'}")

    def save(self, models_dir):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "W_user": self.W_user,
            "W_book": self.W_book
        }, models_dir / "cf_model.pt")
        print(f"Saved model to {models_dir / 'cf_model.pt'}")

    def fit(self, user_ids, item_ids, ratings, epochs=10, batch_size=8192, lr=1e-3, lambda_reg=1e-4, models_dir=None):
        print(f'Using device: {self.device}')

        user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)
        ratings = torch.tensor(ratings, dtype=torch.float32, device=self.device)

        num_users = int(user_ids.max().item()) + 1
        num_items = int(item_ids.max().item()) + 1
        n = len(ratings)

        self.W_user = (torch.randn(num_users, self.embedding_dim, device=self.device) * 0.1).requires_grad_(True)
        self.W_book = (torch.randn(num_items, self.embedding_dim, device=self.device) * 0.1).requires_grad_(True)

        optimizer = torch.optim.Adam([self.W_user, self.W_book], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        for epoch in range(epochs):

            idx_schedule = torch.randperm(n, device=self.device)
            total_loss, n_batches = 0.0, 0

            progress_bar = tqdm(range(0, n, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
            for start in progress_bar:
                idx = idx_schedule[start:start + batch_size]
                u = user_ids[idx]
                i = item_ids[idx]
                r = ratings[idx]

                user_embeds = self.W_user[u]
                item_embeds = self.W_book[i]

                scores = (user_embeds * item_embeds).sum(dim=1)
                loss = ((scores - r) ** 2).mean() + lambda_reg * (
                    user_embeds.norm(dim=1).pow(2).mean() + item_embeds.norm(dim=1).pow(2).mean()
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

        self.save(models_dir)

    def score(self, user_id, item_id):
        user_embed = self.W_user[user_id]
        item_embed = self.W_book[item_id]
        return (user_embed * item_embed).sum()
    
    def recommend(self, user_id, item_ids):
        scores = (self.W_book[item_ids] * self.W_user[user_id]).sum(dim=1)
        top_indices = torch.topk(scores, self.top_n).indices

        return [item_ids[i] for i in top_indices.tolist()]
