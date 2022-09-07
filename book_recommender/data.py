import pandas as pd

def get_books_data():
    #return pd.read_csv("gs://data_for_api_again/books_info_books_cleaned.csv")
    return pd.read_parquet("book_recommender/data/books_info_books_cleaned.parquet")
