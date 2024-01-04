#################################
# This file contains all functions which are used in the Exploratory Data Analysis part of eda_kg.ipynb
#################################


# load packages
import numpy as np
import pandas as pd
import re
import string

from wordcloud import WordCloud

# text analysis nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize



def cut_punctuation(data):
    """
    Remove puncutation function with nltk set 
    """
    for puncts in string.punctuation:
        data = data.replace(puncts, '')
    return data

def clean_df(df, stopwords):
    """
    Remove punctuation and stopwords and gender specific stopwords and "harry, said"
    """
    gender_specific_words = ['he', 'she', 'herself', 'himself', 'his', 'her']
    stopwords_1 = [''.join(item for item in x if item not in string.punctuation) for x in stopwords] 
    stopwords_2 = [word for word in stopwords_1 if word not in gender_specific_words]
    words_remove_harry = ['harry', 'said']
    
    # generate new dataframe columns
    df['words']=df['Text'].str.lower().apply(cut_punctuation).apply(word_tokenize) 
    df['counts'] = df['words'].str.len() 

    df['clean_words']=df['words'].apply(lambda x: [word for word in x if word not in stopwords_1]) 
    df['clean_words_gender'] = df['words'].apply(lambda x: [word for word in x if word not in stopwords_2]) 
    df['words_remove_harry'] = df['clean_words'].apply(lambda x: [word for word in x if word not in words_remove_harry])
    return df.reset_index().drop(['index'], axis = 1)

def create_sentence(df):
    """
    Tokenize the dataframe to create sentence
    """
    sentence_df = df[['Book','Chapter','Text']].reset_index().drop(["index"], axis=1)
    sentence_df = sentence_df.join(sentence_df.Text.apply(sent_tokenize).rename('sentences'))
    return sentence_df
def get_sentence_df(sentence_df):
    """
    Generate each sentenize token into its own row in a new dataframe (clean sentence)
    """
    sentence_df = sentence_df.sentences.apply(pd.Series).merge(sentence_df, left_index = True, right_index = True).drop(["Text"], axis = 1).drop(["sentences"], axis = 1).melt(id_vars = ['Book', 'Chapter'], value_name = "sentence")
    sentence_df = sentence_df.drop("variable", axis = 1).dropna()

    sentence_df = sentence_df.sort_values(by=['Book', 'Chapter']).reset_index().drop(['index'], axis = 1)
    sentence_df['sentence'] = sentence_df.sentence.apply(cut_punctuation).apply(lambda x: x.lower()) 
    return sentence_df



def get_WordCloud(data, color):
    """
    Build wordcloud from frequencies with maximal 100 words
    """
    data = data.set_index('word').to_dict()['count']
    word_cloud = WordCloud(width=600, height=300, max_words=100,colormap= color,margin=10,max_font_size=300,min_font_size = 1, background_color='black').generate_from_frequencies(data)
    return word_cloud


def clean_data_for_vectorization(txt): 
    """
    Removal of specific parameters and white spaces for TF-IDF Vectorization
    """
    token = re.sub('[^a-zA-z\s]', '', txt) 
    token = re.sub('_', '', token) 
    token = re.sub('\s+', ' ', token) 
    token = token.strip() 
    if token != '': 
            return token.lower() 


def get_gender_words(get_words):
    """
    Generate word rank slope chart after (Emil Hvitfeldt) for a set of male and female words
    """
    gender_words = pd.DataFrame({
        'men': ["he", "his", "men", "himself"],
        'women': [ "she", "her", "women", "herself"]
    })

    gender_words['male_rank_log10'] = np.log10(gender_words['men'].map(lambda x: get_words.index(x) + 1))
    gender_words['female_rank_log10'] = np.log10(gender_words['women'].map(lambda x: get_words.index(x) + 1))

    # Calculate rank difference log10
    gender_words['rank_diff_log10'] = gender_words['male_rank_log10'] - gender_words['female_rank_log10']

    rank_diff_log10 = gender_words['rank_diff_log10']
    gender_words = gender_words.melt(id_vars=['men', 'women'], value_vars=['male_rank_log10', 'female_rank_log10'],
                                var_name='index', value_name='rank')
    gender_words['label'] = np.where(gender_words['index'] == 'male_rank_log10',gender_words['men'], gender_words['women'])


    # Recode index values
    gender_words['index'] = gender_words['index'].replace({'male_rank_log10': 'male', 'female_rank_log10': 'female'})

    gender_words['rank_diff_log10'] = pd.Series(np.array([rank_diff_log10]*2).reshape(1,-1)[0])


    limits = np.max(np.abs(gender_words['rank_diff_log10'])) * np.array([-1, 1])
    return gender_words


def get_character_words(get_words):
    """
    Generate word rank slope chart after (Emil Hvitfeldt) for different characters and sentiment analysis
    """
    # Calculate male and female rank log10
    character_words = pd.DataFrame({
        'main_pos_character': ["harry", "good", "positive", "friend"],
        'main_neg_character': [ "voldemort", "bad", "negative", "enemy"]
    })

    character_words['pos_rank_log10'] = np.log10(character_words['main_pos_character'].map(lambda x: get_words.index(x) + 1))
    character_words['neg_rank_log10'] = np.log10(character_words['main_neg_character'].map(lambda x: get_words.index(x) + 1))

    # Calculate rank difference log10
    character_words['rank_diff_log10'] = character_words['pos_rank_log10'] - character_words['neg_rank_log10']

    rank_diff_log10 = character_words['rank_diff_log10']
    character_words = character_words.melt(id_vars=['main_pos_character', 'main_neg_character'], value_vars=['pos_rank_log10', 'neg_rank_log10'],
                                var_name='index', value_name='rank')
    character_words['label'] = np.where(character_words['index'] == 'pos_rank_log10',character_words['main_pos_character'], character_words['main_neg_character'])


    # Recode index values
    character_words['index'] = character_words['index'].replace({'pos_rank_log10': 'pos', 'neg_rank_log10': 'neg'})

    character_words['rank_diff_log10'] = pd.Series(np.array([rank_diff_log10]*2).reshape(1,-1)[0])


    limits = np.max(np.abs(character_words['rank_diff_log10'])) * np.array([-1, 1])
    return character_words 


def recombine_text(x):
    """
    Function to recombine text with join
    """
    comb = ' '.join(x)
    return comb