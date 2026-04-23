import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def prepare_cf_data(input_path, output_train_path, output_test_path, test_size=0.2, random_state=42):
    interactions_df = pl.read_ndjson(input_path)

    train_raw, test_raw = train_test_split(interactions_df.to_pandas(), test_size=test_size, random_state=random_state)

    user_enc = LabelEncoder()
    book_enc = LabelEncoder()

    train_raw["user_id"] = user_enc.fit_transform(train_raw["user_id"])
    train_raw["book_id"] = book_enc.fit_transform(train_raw["book_id"])

    test_raw = test_raw[
        test_raw["user_id"].isin(user_enc.classes_) &
        test_raw["book_id"].isin(book_enc.classes_)
    ]
    test_raw["user_id"] = user_enc.transform(test_raw["user_id"])
    test_raw["book_id"] = book_enc.transform(test_raw["book_id"])

    train_df = pl.from_pandas(train_raw).select(["user_id", "book_id", "rating"])
    test_df = pl.from_pandas(test_raw).select(["user_id", "book_id", "rating"])

    n_users = train_df["user_id"].n_unique()
    n_books = train_df["book_id"].n_unique()
    
    print(f"Train — Users: {n_users}, Books: {n_books}, Interactions: {len(train_df)}")
    print(f"Test — Interactions: {len(test_df)}")

    train_df.write_parquet(output_train_path)
    test_df.write_parquet(output_test_path)
