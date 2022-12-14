from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def make_clickable(val):
    return '<a target="_blank" href="{}">Goodreads</a>'.format(val, val)

def show_image(val):
    return '<a href="{}"><img src="{}" width=50></img></a>'.format(val, val)

def search(query,df):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df["mod_title"])
    processed = re.sub("[^a-zA-Z0-9 ]", " ", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = df.iloc[indices]
    results = results.sort_values("ratings", ascending=False)
    #new line
    results.reset_index(inplace = True)

    #return results.head(5).style.format({'url': make_clickable, 'cover_image': show_image})
    return results.head(5)

"""Example of a query: search("harry potter and the prisoner of azkaban", vectorizer lalalalalalal)"""
""""""

def title_to_id(book_titles : list, df):
    book_ids = []
    for book_title in book_titles:
        book_id = df[df['title']==book_title].sort_values('ratings', ascending = False).head(1)['book_id'].values[0]
        book_ids.append(book_id)
    return book_ids
