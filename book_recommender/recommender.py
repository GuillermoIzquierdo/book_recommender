## Imports

# Basics:
import pandas as pd
import numpy as np

 # Sklearn:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Language processing

import re
from sklearn.feature_extraction.text import TfidfVectorizer

from google.cloud import bigquery

def genre_recomendation(df, book_title, n_features = 25, n_books=5):
    '''
    Recommends books based on a book_title.
    Expects a df with at least two columns:
        "clean_genres" — string of words without punctuation
        "book_title" – string of words (revise)
    Optional keywords:
        "n_features" – amount of features used to make the recommendation.
        "n_books" – quantity of books that will be recommended.
    '''
    if book_title in df['book_title'].values:
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(df['clean_genres'])
        count_df = pd.DataFrame(count_matrix.toarray(), index=df.index.tolist())
        svd = TruncatedSVD(n_components=25)
        latent_df = svd.fit_transform(count_df)
        latent_df = pd.DataFrame(latent_df[:,0:n_features], index=df['book_title'].tolist())
        v = np.array(latent_df.loc[book_title]).reshape(1, -1)
        sim = cosine_similarity(latent_df, v).reshape(-1)
        dictDf = {'sim':sim}
        recommendation_df = pd.DataFrame(dictDf, index = latent_df.index)
        r = recommendation_df.sort_values('sim', ascending = False).head(n_books + 1).index
        return r[1:]
    else:
        return ''

def get_titles():
    client = bigquery.Client()
    query='''
    SELECT * FROM `lewagon-bootcamp-356013.book_recommender.books_info`
    '''
    data = client.query(query).to_dataframe()
    return data

def desc_recommendator(query, df, min_df = 0.2, n_indices = -10):
    '''_'''
    preprocessed = re.sub('[^a-zA-Z0-9]', '', query.lower())
    vectorizer = TfidfVectorizer(min_df = min_df)
    vectorized_text = vectorizer.fit_transform(df['mod_desc'])
    query_vec = vectorizer.transform([preprocessed])
    similarity = cosine_similarity(query_vec, vectorized_text).flatten()
    indices = np.argpartition(similarity, n_indices)[n_indices:]
    results = df.iloc[indices]
    return results

def find_users(liked_books : list):
    client = bigquery.Client()
    query = f'''
    SELECT DISTINCT(user_id) FROM `lewagon-bootcamp-356013.book_recommender.interactions2`
    WHERE book_id IN {tuple(liked_books)}
    AND rating >= 4
    '''
    overlap_users = client.query(query).to_dataframe()['user_id'].tolist()
    return overlap_users

def books_users(overlap_users):
    client = bigquery.Client()
    query=f'''
    SELECT user_id, book_id, rating FROM `lewagon-bootcamp-356013.book_recommender.interactions2`
    WHERE user_id IN {tuple(overlap_users)}
    '''
    book_recs = client.query(query).to_dataframe()
    return book_recs

def get_recs(book_recs, liked_books):
    '''books_titles needs to be in the environment already assigned'''
    all_recs = book_recs['book_id'].value_counts().to_frame().reset_index()
    all_recs.columns = ["book_id", "book_count"]
    books_titles = get_titles()
    all_recs_titles = all_recs.merge(books_titles, how="inner", on="book_id")
    all_recs_titles["score"] = all_recs_titles["book_count"] * (all_recs_titles["book_count"] / all_recs_titles["ratings"])
    popular_recs = all_recs_titles[all_recs_titles["book_count"] > 200].sort_values("score", ascending=False)
    final_recs = popular_recs[~popular_recs["book_id"].isin(liked_books)].head(10)
    return final_recs

def recommender(liked_books: list):
    overlap_users=find_users(liked_books)
    book_recs=books_users(overlap_users)
    recs=get_recs(book_recs, liked_books)
    return recs
