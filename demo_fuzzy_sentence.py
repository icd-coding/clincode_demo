'''
@author: taridzo, 2021
Fuzzy string matching based on sentence similarity.
'''
import sys
from thefuzz import fuzz
import pandas as pd
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from tqdm import tqdm

tqdm.pandas()
#import spacy

# spacy.prefer_gpu()
# nlp = spacy.load('sv_core_news_lg')
# nltk.download('stopwords')
stop_words = set(stopwords.words('swedish'))
stemmer = SnowballStemmer("swedish")
punct = string.punctuation

SIMILARITY_THRESHOLD = 10


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punct))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])


def get_similarity(sentence, icd_description):
    return fuzz.partial_ratio(str(sentence).lower(), str(icd_description).lower())


def get_top_n(letter, choices):
    assigned_codes = pd.DataFrame(columns=['description', 'code', 'block', 'similarity'])
    sentences = sent_tokenize(letter)
    for sentence in sentences:
        sentence = remove_stopwords(sentence)
        sentence = remove_punctuation(sentence)
        choices['similarity'] = choices.description.apply(lambda x: get_similarity(sentence, x))
        choices = choices.loc[choices['similarity'] >= SIMILARITY_THRESHOLD]
        choices = choices.sort_values(['code', 'similarity'], ascending=False).groupby('code').head(1)
        assigned_codes = pd.concat([assigned_codes, choices], ignore_index=True)

    assigned_codes = assigned_codes.sort_values(['code', 'similarity'], ascending=False).groupby('code').head(1)

    if assigned_codes.empty:
        print("EMPTY: returning popular codes...")
        return ['K567', 'K573', 'K358', 'K590', 'K800']
    else:
        assigned_codes['similarity'] = assigned_codes['similarity'].astype(int)
        resx = assigned_codes.nlargest(5, 'similarity')
        return resx.code.to_list()


def get_fuzzy_sentence_top_n(txt, df):
    df.description = df.description.str.lower()
    return get_top_n(str.lower(txt), df)
