import pandas as pd

def get_books_data():
    return pd.read_csv("gs://book_recommender/books_info/books_cleaned.csv")
