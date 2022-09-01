from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

#from book_recommender.wherever is the model import the model "load_model"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # all origins
    allow_credentials=True,
    allow_methods=["*"], #all methods
    allow_headers=["*"], # all headers
)

#app.state.model = load_model()

# defining a root / endpoint


@app.get("/predict")
def predict(keywords: str):
    return {'keywords': keywords}


@app.get("/")
def root():
    return {"greeting": "Hello"}
