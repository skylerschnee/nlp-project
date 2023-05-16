import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns

from prepare import basic_clean_df, readme_length

import nltk
import unicodedata

from wordcloud import WordCloud




def clean(text):
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


def get_word_freqs(df):
    python_words = clean(' '.join(df[df.language=='python']['readme_contents']))
    java_words = clean(' '.join(df[df.language=='java']['readme_contents']))
    java_script_words = clean(' '.join(df[df.language=='javascript']['readme_contents']))
    all_words = clean(' '.join(df['readme_contents']))
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    java_script_freq = pd.Series(java_script_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    return python_freq, java_freq, java_script_freq, all_freq, all_words, python_words, java_words, java_script_words



def top_twenty_words_vis(python_freq, java_freq, java_script_freq, all_freq):
    word_counts = pd.concat([python_freq,java_freq, java_script_freq, all_freq], axis=1
         ).fillna(0
                 ).astype(int)
    word_counts.columns = ['python','java', 'javascript','all']
    word_counts.head()
    word_counts.sort_values('all', ascending=False
                       )[['python','java','javascript']].head(20).plot.barh()
    
    
def get_top_twenty_all_wordgram(all_words):
    img = WordCloud(background_color='White',
         ).generate(' '.join(all_words))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most common python words')
    plt.show()
    
    

def top_twenty_bigrams(java_words, python_words, java_script_words):
    dif_words = [python_words, java_script_words, java_words]
    titles = ['python', 'js', 'java']
    for i in range(3):
        pd.Series(nltk.bigrams(dif_words[i])).value_counts().head(20).plot.barh()
        plt.title(f'top 20 bigrams {titles[i]}')
        plt.show()
        
        
def get_kruskal_wallis_test(df):
    '''
    get_kruskal_wallis_test takes in a pandas dataframe and outputs t-stat
    and p-value of a Kruskal-Wallis Test.
    '''
    python_readme_lengths = df[df.language == 'python']['readme_length']
    java_readme_lengths = df[df.language == 'java']['readme_length']
    javascript_readme_lengths= df[df.language == 'javascript']['readme_length']
    
    # Perform Kruskal-Wallis test
    statistic, p_value = stats.kruskal(
        python_readme_lengths, java_readme_lengths, javascript_readme_lengths)

    # Output the results
    print("Kruskal-Wallis Test")
    print(f"Test statistic: {statistic}")
    print(f"P-value: {p_value}")
    
    
    