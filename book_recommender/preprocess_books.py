import pandas as pd
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




"""[mod_desc] and [mod_title] are new columns that we add to the dataframe that contains the preprocess and cleaned data of ["title"] and ["description"]
respectively"""

def cleaning(sentence):

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    tokenized_sentence = word_tokenize(sentence) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    tokenized_sentence_cleaned = [ ## remove stopwords
        w for w in tokenized_sentence if not w in stop_words
    ]
    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)

    return cleaned_sentence


def preprocessing_book_title(df):
    df=df.dropna()
    df["mod_title"] = df["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["mod_title"] = df["mod_title"].str.replace("\s+", " ", regex=True)
    df["mod_title"]=df["mod_title"].str.lower()
    df = df[df["mod_title"].str.len() > 0]
    return df

def preprocessing_book_desc(df):
        df=df.dropna()
        df["mod_desc"] = df["description"].str.lower()
        df["mod_desc"]  = df["mod_desc"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
        df["mod_desc"]=df["description"].apply(cleaning)
        return df

def clean_genres(df):
    '''Removes characters – []'
    Expects a df with a column called "genres".
    Returns the df with a new column named "clean_genres"
    '''
    if 'clean_genres' in df.columns:
        df['clean_genres'] = df['genres'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace("'", '')
        )
        return df
    else:
        return '"clean_genres" not found in columns'
