import pathlib

import polars as pl

from systems.cf import CF_Recommender
from data_processing.cf_prep import prepare_cf_data
from eval.eval import evaluate_recommender

LOAD_EXISTING_MODEL = False

if __name__ == "__main__":

    train_path = "./processed-data/cf_data_train.parquet"
    test_path = "./processed-data/cf_data_test.parquet"

    if (not pathlib.Path(train_path).exists()) or (not pathlib.Path(test_path).exists()):
        print("Data files not found, preparing data...")
        prepare_cf_data(
            input_path="./processed-data/cleaned_interactions_fantasy_paranormal.json",
            output_train_path=train_path,
            output_test_path=test_path
        )

    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)

    user_ids = train_df["user_id"].to_numpy()
    book_ids = train_df["book_id"].to_numpy()
    ratings = train_df["rating"].to_numpy() / 5.0

    cf_recommender = CF_Recommender(embedding_dim=64, top_n=10)

    if pathlib.Path("./models/cf_model.pt").exists() and LOAD_EXISTING_MODEL:
        print("Loading existing model...")
        cf_recommender.load("./models/")
    else:
        print("No existing model found, training new model...")
        cf_recommender.fit(user_ids, book_ids, ratings, epochs=5, lr=1e-3, lambda_reg=1e-4, models_dir="./models")

    evaluate_recommender(cf_recommender, train_df, test_df)
