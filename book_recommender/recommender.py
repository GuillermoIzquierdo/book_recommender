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

# Google Cloud

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


def find_users(liked_books, table):

    client = bigquery.Client()

    query = f'''
    SELECT user_id FROM `lewagon-bootcamp-356013.book_recommender.interactions`
    WHERE book_id IN {tuple(liked_books)}
    '''

    df = client.query(query).to_dataframe()

    return df
