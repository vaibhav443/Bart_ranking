from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import unidecode
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import operator
import json
import requests

nltk.download('stopwords')
stop = stopwords.words('english')


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def basic_preprocessing_ST(Df):
    """

    :param Df: data
    :return: preprocessed data for semantic match
    """
    df = pd.DataFrame(Df, copy=True)
    df['description'] = df['description'].apply(remove_accented_chars)
    df['title'] = df['title'].apply(remove_accented_chars)

    # removing digits and converting into lower case
    df['description'] = df['description'].str.lower().str.replace('\d+', '')
    df['title'] = df['title'].str.lower().str.replace('\d+', '')

    # removing extrawhite spacing
    df['description'] = df['description'].str.replace('\s\s+', ' ')
    df['title'] = df['title'].str.replace('\s\s+', ' ')

    return df


def basic_preprocessing_keywords_ST(keywords):
    """

    :param keywords: keyword list
    :return: preprocessed keywords list for semantic match
    """
    for i in range(len(keywords)):
        keywords[i] = remove_accented_chars(keywords[i])
        keywords[i] = keywords[i].lower()
        keywords[i] = re.sub(r'\d+', '', keywords[i])
        keywords[i] = re.sub(r'\s\s+', ' ', keywords[i])

    return keywords


# from transformers import pipeline
# classifier = pipeline("zero-shot-classification",
#                         model="facebook/bart-large-mnli")
def bart_scores(data, keyword, lang):
    """

    :param data: preprocessed data
    :param keyword: preprocessed keyword
    :param lang: language (en,fr etc...)
    :return: dict of bart_scores for keyword with respect to each product
    """
    title_text = [data['title'][i] for i in data.index]
    description_text = [data['description'][i] for i in data.index]
    dict_bart_title = {i: bart_service(title_text[i], lang, keyword)['scores'][0] for i in range(len(title_text))}
    dict_bart_descrip = {i: bart_service(description_text[i], lang, keyword)['scores'][0] for i in
                         range(len(description_text))}
    dict_bart = {}
    for i in range(data.shape[0]):
        dict_bart[i] = (dict_bart_title[i]) * 60 + (dict_bart_descrip[i] * 40)

    return dict_bart


def bart_result(Dict_bart, N_prod, Df):
    """

    :param Dict_bart: dict of bart scores
    :param N_prod: No. of products to be ranked
    :param Df: product dataset
    :return: top n ranked products
    """
    sorted_dict = dict(sorted(Dict_bart.items(), key=operator.itemgetter(1), reverse=True))
    dict_items = sorted_dict.items()
    top_n_prods = list(dict_items)[:N_prod]
    results1 = pd.DataFrame(data=top_n_prods, columns=['index', 'score'])
    title1 = []
    description1 = []
    brand = []
    for i in results1.index:
        title1.append(Df['title'][results1['index'][i]])
        description1.append(Df['description'][results1['index'][i]])
        brand.append(Df['brand'][results1['index'][i]])

    results1['title'] = title1
    results1['description'] = description1
    results1['brand'] = brand

    return results1


def bart_service(text, lang, keyword):
    """

    :param text: piece of text
    :param lang: language(en,fr etc...)
    :param keyword: keyword for which bart score is to be calculated
    :return: scores for keyword with respect to text
    """
    bart_endpoint = 'http://35.180.247.177:8001/bart'

    params = {
        'text': text,
        'keyword': [keyword],
        'lang': lang
    }
    params = json.dumps(params)

    query_emb = requests.post(bart_endpoint, data=params)
    if query_emb.ok:
        query_emb = query_emb.json()
    else:
        raise ValueError(query_emb.text)

    return query_emb
