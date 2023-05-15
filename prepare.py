#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
import unicodedata
import nltk

#################################################

def remove_non_matching(text):
    regex = re.compile(r'[^\w\s\']')
    return regex.sub('', text)
    
def remove_non_matching_df(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: remove_non_matching(x))
    return df

def basic_clean_df(df):
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df= normalize_df(df)
    df=remove_non_matching_df(df)
    return df

def tokenize(text):
    tokens = nltk.tokenize.ToktokTokenizer()
    tokenized = tokens.tokenize(text, return_str=True)
    return tokenized

def tokenize_df(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: tokenize(x))
    return df

def stem(text):
    stemmer = nltk.porter.PorterStemmer()
    stem = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(stem)
    return text

def stem_df(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: stem(x))
    return df

def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in text.split()]
    text = ' '.join(lemmatized)
    return text

def lematize_df(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: lemmatize(x))
    return df

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords
    text = [word for word in text.split()]
    text = ' '.join([word for word in text 
                         if word not in stopwords.words('english')])
    return text


def remove_df_stopwords(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: remove_stopwords(x))
    return df

def remove_specials(text):
    regex = re.compile(r"[^a-z'\s]")
    return regex.sub('', text)
    
def remove_specials_df(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: remove_specials(x))
    return df

def normalize_text(text):
    return unicodedata.normalize('NFC', text)

def normalize_df(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: normalize_text(x))
    return df

def remove_words(text, words):
    for word in words:
        text = text.replace(word, '')
    return text

def remove_words_df(df):
    words = ['python', 'py', 'java', 'javascript', 'js', 'script']
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: remove_words(x, words))
    return df

#######################################################

def basic_clean_df(df):
    df = remove_words_df(df)
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = remove_df_stopwords(df)
    df = normalize_df(df)
    df = remove_non_matching_df(df)
    df = tokenize_df(df)
    df = lematize_df(df)
    df = normalize_df(df)
    df = remove_specials_df(df)
    df = df.drop(columns = 'repo')
    return df

########################################################
def readme_length(col):
    """ This function takes in each README.md file and returns the word count for each file"""
    length = []
    for x in df[col]:
        read_len= len(x)
        length.append(read_len)
    return length
#  Use this to add to df
# df['readme_length']= readme_length('readme_contents')
