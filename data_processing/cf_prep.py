import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def prepare_cf_data(train_data_path, test_data_path, output_train_path, output_test_path):
    print("Preparing collaborative filtering data...")
    print(f"Reading raw data from {train_data_path} and {test_data_path}...")
    train_raw = pl.read_ndjson(train_data_path).to_pandas()
    test_raw = pl.read_ndjson(test_data_path).to_pandas()

    user_enc = LabelEncoder()
    book_enc = LabelEncoder()

    train_raw["user_id_enc"] = user_enc.fit_transform(train_raw["user_id"])
    train_raw["work_id_enc"] = book_enc.fit_transform(train_raw["work_id"])

    # filter test set to only include users and books seen in training set, then encode
    test_raw = test_raw[
        test_raw["user_id"].isin(user_enc.classes_) &
        test_raw["work_id"].isin(book_enc.classes_)
    ]
    test_raw["user_id_enc"] = user_enc.transform(test_raw["user_id"])
    test_raw["work_id_enc"] = book_enc.transform(test_raw["work_id"])

    train_df = pl.from_pandas(train_raw).select(["user_id_enc", "work_id_enc", "rating", "user_id", "work_id"])
    test_df = pl.from_pandas(test_raw).select(["user_id_enc", "work_id_enc", "rating", "user_id", "work_id"])

    n_users = train_df["user_id_enc"].n_unique()
    n_works = train_df["work_id_enc"].n_unique()
    
    print(f"Train — Users: {n_users}, Works(unique books): {n_works}, Interactions: {len(train_df)}")
    print(f"Test — Interactions: {len(test_df)}\n")

    train_df.write_parquet(output_train_path)
    test_df.write_parquet(output_test_path)
