FROM python:3.8.12-buster
COPY book_recommender book_recommender
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY books_info_books_cleaned.parquet books_info_books_cleaned.parquet
RUN pip install --upgrade pip
RUN pip install .
CMD uvicorn book_recommender.fast:app --host 0.0.0.0 --port $PORT
