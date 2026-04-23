import numpy as np
import torch


def evaluate_recommender(recommender, train_df, test_df):
    device = recommender.device

    # --- Evaluate: RMSE on test ---
    test_user_ids = test_df["user_id"].to_numpy()
    test_book_ids = test_df["book_id"].to_numpy()
    test_ratings = test_df["rating"].to_numpy() / 5.0

    test_u = torch.tensor(test_user_ids, dtype=torch.long, device=device)
    test_b = torch.tensor(test_book_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        predicted = (
            recommender.W_user[test_u] * recommender.W_book[test_b]
        ).sum(dim=1).cpu().numpy()

    mse = np.mean((predicted - test_ratings) ** 2)
    rmse = np.sqrt(mse)
    rmse_native = rmse * 5

    print(f"\nTest RMSE (normalized [0,1] scale): {rmse:.4f}")
    print(f"Test RMSE (native 1-5 scale):       {rmse_native:.4f}\n")

    # Baseline: global mean predictor
    global_mean = train_df["rating"].mean() / 5.0
    baseline_rmse = np.sqrt(np.mean((global_mean - test_ratings) ** 2))
    print(f"Baseline RMSE (normalized [0,1] scale): {baseline_rmse:.4f}")
    print(f"Baseline RMSE (native 1-5 scale):       {baseline_rmse * 5:.4f}")
