from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

from book_recommender.data import get_books_data
from book_recommender.search_engine_titles import search
from book_recommender.recommender import recommender
from book_recommender.search_engine_titles import title_to_id


from sklearn.feature_extraction.text import TfidfVectorizer


#from book_recommender.wherever is the model import the model "load_model"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # all origins
    allow_credentials=True,
    allow_methods=["*"], #all methods
    allow_headers=["*"], # all headers
)

app.state.data = get_books_data()


# defining a root / endpoint


@app.get("/search")
def predict(keywords: str):


    results = search(keywords, app.state.data)

    return {'book 1': {'title': str(results['title'][0]), 'image': str(results['cover_img'][0]), 'book_id': str(results['book_id'][0]), 'url': str(results['url'][0]), 'description': str(results['description'][0])},
            'book 2': {'title': str(results['title'][1]), 'image': str(results['cover_img'][1]), 'book_id': str(results['book_id'][1]), 'url': str(results['url'][1]), 'description': str(results['description'][1])},
            'book 3': {'title': str(results['title'][2]), 'image': str(results['cover_img'][2]), 'book_id': str(results['book_id'][2]), 'url': str(results['url'][2]), 'description': str(results['description'][2])},
            'book 4': {'title': str(results['title'][3]), 'image': str(results['cover_img'][3]), 'book_id': str(results['book_id'][3]), 'url': str(results['url'][3]), 'description': str(results['description'][3])},
            'book 5': {'title': str(results['title'][4]), 'image': str(results['cover_img'][4]), 'book_id': str(results['book_id'][4]), 'url': str(results['url'][4]), 'description': str(results['description'][4])}}


@app.post("/recommendation")
def recommendation(book_ids: str):
    lst_books = book_ids.split(',')
    recommendations = recommender(lst_books).to_json()
    return recommendations





@app.get("/")
def root():
    return {"greeting": "Hello"}
