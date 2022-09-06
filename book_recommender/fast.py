from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

from book_recommender.data import get_books_data
from book_recommender.search_engine_titles import search
from book_recommender.recommender import desc_recommendator

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


@app.get("/predict")
def predict(keywords: str):

   # breakpoint()
    results = search(keywords, app.state.data)

    return {'book 1': {'title': results['title'][0], 'image': results['cover_img'][0]},
            'book 2': {'title': results['title'][1], 'image': results['cover_img'][1]},
            'book 3': {'title': results['title'][2], 'image': results['cover_img'][2]},
            'book 4': {'title': results['title'][3], 'image': results['cover_img'][3]},
            'book 5': {'title': results['title'][4], 'image': results['cover_img'][4]},


            }


@app.get("/")
def root():
    return {"greeting": "Hello"}
