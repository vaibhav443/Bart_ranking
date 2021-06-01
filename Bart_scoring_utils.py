from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import spacy
import unidecode
import string
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import operator

nltk.download('stopwords')
stop = stopwords.words('english')


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def basic_preprocessing_exact(Df):
    # removing accented characters
    """

    :param Df: data for exact match
    :return: preprossed data for exact match
    """
    df = pd.DataFrame(Df, copy=True)
    df['description_eng'] = df['description_eng'].apply(remove_accented_chars)
    df['title_eng'] = df['title_eng'].apply(remove_accented_chars)

    # removing digits and converting into lower case
    df['description_eng'] = df['description_eng'].str.lower().str.replace('\d+', '')
    df['title_eng'] = df['title_eng'].str.lower().str.replace('\d+', '')

    # removing extrawhite spacing
    df['description_eng'] = df['description_eng'].str.replace('\s\s+', ' ')
    df['title_eng'] = df['title_eng'].str.replace('\s\s+', ' ')

    # removing punctuations
    df['description_eng'] = df['description_eng'].str.replace('[^\w\s]', '')
    df['title_eng'] = df['title_eng'].str.replace('[^\w\s]', '')

    # removing stopwords
    df['description_eng'] = df['description_eng'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['title_eng'] = df['title_eng'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df


def basic_preprocessing_ST(Df):
    """

    :param Df: data
    :return: preprossed data for semantic match
    """
    df = pd.DataFrame(Df, copy=True)
    df['description_eng'] = df['description_eng'].apply(remove_accented_chars)
    df['title_eng'] = df['title_eng'].apply(remove_accented_chars)

    df['description_eng'] = df['description_eng'].str.lower().str.replace('\d+', '')
    df['title_eng'] = df['title_eng'].str.lower().str.replace('\d+', '')

    df['description_eng'] = df['description_eng'].str.replace('\s\s+', ' ')
    df['title_eng'] = df['title_eng'].str.replace('\s\s+', ' ')

    return df

def basic_preprocessing_keyword_exact(keywords):

    """

    :param keywords: keywords list
    :return: preprossed keywords for exact match
    """
    for i in range(len(keywords)):
        keywords[i] = remove_accented_chars(keywords[i])
        keywords[i] = keywords[i].lower()
        keywords[i] = re.sub(r'\d+', '', keywords[i])
        keywords[i] = re.sub(r'\s\s+', ' ', keywords[i])
        keywords[i] = re.sub(r'[^\w\s]', '', keywords[i])
        keywords[i] = ' '.join([word for word in keywords[i].split() if word not in (stop)])
    return keywords


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


from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
def bart_scores(data, keyword):
    title_text = []
    description_text = []
    for i in data.index:
        title_text.append(data['title_eng'][i])
        description_text.append(data['description_eng'][i])
    dict_bart_title = {}
    i = 0
    for text in title_text:
        dict_bart_title[i] = classifier(text, keyword)['scores'][0]
        i = i + 1
    j = 0
    dict_bart_descrip = {}
    for text in description_text:
        dict_bart_descrip[j] = classifier(text, keyword)['scores'][0]
        j = j + 1

    dict_bart = {}
    for i in range(data.shape[0]):
        dict_bart[i] = (dict_bart_title[i]) * 60 + (dict_bart_descrip[i] * 40)

    return dict_bart


def bart_result(Dict_bart, N_prod, Df):

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