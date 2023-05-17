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
from scipy.stats import stats

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB



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
    '''
    Calculate word frequencies for different programming languages in the given DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing programming language and README content.

    Returns:
    tuple: A tuple containing the following:
        - python_freq (Series): Word frequencies for Python language.
        - java_freq (Series): Word frequencies for Java language.
        - java_script_freq (Series): Word frequencies for JavaScript language.
        - all_freq (Series): Word frequencies for all programming languages combined.
        - all_words (str): Concatenated clean text of all README contents.
        - python_words (str): Concatenated clean text of Python README contents.
        - java_words (str): Concatenated clean text of Java README contents.
        - java_script_words (str): Concatenated clean text of JavaScript README contents.
    '''
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
    '''
    Create a horizontal bar plot to visualize the top 20 most frequently used words for each programming language.

    Parameters:
    python_freq (Series): Word frequencies for Python language.
    java_freq (Series): Word frequencies for Java language.
    java_script_freq (Series): Word frequencies for JavaScript language.
    all_freq (Series): Word frequencies for all programming languages combined.

    Returns:
    None
    '''
    word_counts = pd.concat([python_freq,java_freq, java_script_freq, all_freq], axis=1
         ).fillna(0
                 ).astype(int)
    word_counts.columns = ['python','java', 'javascript','all']
    word_counts.head()
    word_counts.sort_values('all', ascending=False
                       )[['python','java','javascript']].head(20).plot.barh()
    
    
def get_top_twenty_all_wordgram(all_words):
    '''
    returns a wordgram for all words
    '''
    img = WordCloud(background_color='White',
         ).generate(' '.join(all_words))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most common words')
    plt.show()
    
    

def top_twenty_bigrams(java_words, python_words, java_script_words):
    '''
    returns the top twenty bigram for all of the languages
    '''
    dif_words = [python_words, java_script_words, java_words]
    titles = ['python', 'js', 'java']
    for i in range(3):
        pd.Series(nltk.bigrams(dif_words[i])).value_counts().head(20).plot.barh()
        plt.title(f'top 20 bigrams {titles[i]}')
        plt.show()
        
def metrics_gala(train, val):
    '''
    used to build a dictionary
    '''
        results = {}

def get_models(train, val, test, t=0):
    '''
    Train and evaluate multiple classifiers using different models and vectorization techniques.

    Parameters:
    train (DataFrame): Training data containing 'readme_contents' and 'language' columns.
    val (DataFrame): Validation data containing 'readme_contents' and 'language' columns.
    test (DataFrame): Test data containing 'readme_contents' and 'language' columns.
    t (int, optional): Flag to determine if test accuracy should be printed. Default is 0.

    Returns:
    DataFrame: Results of model training and evaluation.
    '''
    results = {}
    x_train = train['readme_contents']
    y_train = train.language
    
    x_val = val['readme_contents']
    y_val = val.language

    x_test = test['readme_contents']
    y_test = test.language
    tv = TfidfVectorizer()
    dt = DecisionTreeClassifier(max_depth=10)
    rf= RandomForestClassifier(max_depth= 10)
    knn = KNeighborsClassifier(n_neighbors= 8)
    
    baseline_acc = round((train.language == 'python').mean(),2)
    results['baseline'] = {'train_acc':baseline_acc}


    #TfidVectorizer on DT
    train_tv= tv.fit_transform(x_train)
    dt.fit(train_tv, y_train)
    val_tv= tv.transform(x_val)
    dt_train=round(dt.score(train_tv, y_train),2)
    dt_val=round(dt.score(val_tv, y_val),2)

    results['DecisionTree']={'train_acc': dt_train,
                            ' val_acc':dt_val,
                            'difference': dt_train-dt_val}

    #TfidVectorizer on RF
    train_tv= tv.fit_transform(x_train)
    rf.fit(train_tv, y_train)
    val_tv= tv.transform(x_val)
    rf_train=round(rf.score(train_tv, y_train),2)
    rf_val=round(rf.score(val_tv, y_val),2)
    results['RandomForest']={'train_acc': round(rf.score(train_tv, y_train),2),
                            ' val_acc':round(rf.score(val_tv, y_val),2),
                            'difference': rf_train-rf_val}

    #TfidVectorizer on KNN
    train_tv= tv.fit_transform(x_train)
    knn.fit(train_tv, y_train)
    val_tv= tv.transform(x_val)
    knn_train=round(knn.score(train_tv, y_train),2)
    knn_val=round(knn.score(val_tv, y_val),2)
    results['KNearestNeighbor']={'train_acc':knn_train,
                                ' val_acc':knn_val,
                                'difference': knn_train-knn_val}

    tfidf = TfidfVectorizer()
    nb = MultinomialNB()
    X_bow = tfidf.fit_transform(x_train)
    nb.fit(X_bow, y_train)
    nb_train = nb.score(X_bow, y_train)
    x_val_bow = tfidf.transform(x_val)
    nb_val = nb.score(x_val_bow, y_val)
    results['Naive Bayes']={'train_acc': round(nb_train,2),
                                ' val_acc':round(nb_val,2),
                                'difference': nb_train-nb_val}
     
    if t == 0:
        return pd.DataFrame(results).T

    else:
        x_test_bow = tfidf.transform(x_test)
        knn.score(x_test_bow, y_test)
        print('Accuracy of Naive_bayes classifier on test set: {:.2f}'
             .format(knn.score(x_test_bow, y_test)))
    
    

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
    

    
def get_wordgrams(python_words, java_words, java_script_words):
    
    '''
    returns word cloud for all of the languages
    '''
    # Generate word clouds
    python_cloud = WordCloud(background_color='white', height=400, width=400).generate(' '.join(python_words))
    java_cloud = WordCloud(background_color='white', height=400, width=400).generate(' '.join(java_words))
    javascript_cloud = WordCloud(background_color='white', height=400, width=400).generate(' '.join(java_script_words))

    # Create the figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the word clouds
    axs[0].imshow(python_cloud, interpolation='bilinear')
    axs[1].imshow(java_cloud, interpolation='bilinear')
    axs[2].imshow(javascript_cloud, interpolation='bilinear')

    # Set titles for the subplots
    axs[0].set_title('Python', fontsize=14, fontweight='bold')
    axs[1].set_title('Java', fontsize=14, fontweight='bold')
    axs[2].set_title('JavaScript', fontsize=14, fontweight='bold')

    # Remove the axis ticks and labels
    for ax in axs:
        ax.axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.05)

    # Display the plot
    plt.show()


