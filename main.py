import pathlib

import polars as pl

from systems.cf_recommender import CF_Recommender
from data_processing.cf_prep import prepare_cf_data

import torch

LOAD_EXISTING_MODEL = False
REDO_DATA_PREP = True

def prepare_data(output_train_path="./processed-data/cf_data_train.parquet", output_test_path="./processed-data/cf_data_test.parquet"):
    if REDO_DATA_PREP or (not pathlib.Path(output_train_path).exists()) or (
        not pathlib.Path(output_test_path).exists()
    ):
        print("Data files not found, preparing data...")
        prepare_cf_data(
            train_data_path = "./processed-data/train_interactions_fantasy_paranormal.json",
            test_data_path = "./processed-data/test_interactions_fantasy_paranormal.json",
            output_train_path=output_train_path,
            output_test_path=output_test_path,
        )

    train_df = pl.read_parquet(output_train_path)
    test_df = pl.read_parquet(output_test_path)

    return train_df, test_df

def train_model(cf_recommender, train_df=None, models_dir="./models"):
    print("Training new model...")
    cf_recommender.fit(
        train_df,
        epochs=10,
        lr=4e-3,
        lambda_reg=2e-3,
        models_dir=models_dir,
    )

def evaluate_global_mean_baseline(train_df, test_df):
    print("Evaluating Global Mean RMSE Baseline...")

    train_mean = train_df["rating"].mean()
    ratings = torch.tensor(test_df["rating"].to_list(), dtype=torch.float32)

    rmse = torch.sqrt(torch.mean((ratings - train_mean) ** 2)).item()

    print(f"Train Mean: {train_mean:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return {"rmse": rmse}

def define_book_info_dict():
    print("Defining book info dictionary...")
    rows = (
        pl.scan_ndjson("./processed-data/processed_books_texts.json")
        .select(["work_id", "title", "author_names", "description"])
        .collect()
        .to_dicts()
    )
    for row in rows:
        row["author_names"] = list(row.get("author_names") or [])
        
    book_info = {str(row["work_id"]): row for row in rows}
    print(f"Defined book info for {len(book_info)} books.")
    return book_info

if __name__ == "__main__":

    train_df, test_df = prepare_data()
    cf_recommender = CF_Recommender(top_n=10, embedding_dim=8, objective="read_bool", use_bias=False)
    train_model(cf_recommender, train_df)

    book_info = define_book_info_dict()
    cf_recommender.evaluate(test_df, top_k=10, export_path="./evaluation_results/cf_evaluation.json", book_info=book_info)
    evaluate_global_mean_baseline(train_df, test_df)
