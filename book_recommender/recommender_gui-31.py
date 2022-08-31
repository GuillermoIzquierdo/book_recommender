## Imports

# Basics:
import pandas as pd
import numpy as np

 # Sklearn:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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
