from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(df["mod_title"])


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def make_clickable(val):
    return '<a target="_blank" href="{}">Goodreads</a>'.format(val, val)

def show_image(val):
    return '<a href="{}"><img src="{}" width=50></img></a>'.format(val, val)

def search(query,vectorizer):
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = df.iloc[indices]
    results = results.sort_values("ratings", ascending=False)

    return results.head(5).style.format({'url': make_clickable, 'cover_image': show_image})

"""Example of a query: search("harry potter and the prisoner of azkaban", vectorizer lalalalalalal)"""
""""""
