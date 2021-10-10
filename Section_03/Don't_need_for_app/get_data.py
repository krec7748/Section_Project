import pandas as pd
from bs4 import BeautifulSoup
import contractions
import re, string, unicodedata
import inflect
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

df = pd.read_csv("/Users/doukkim/Section_03/Section_Project/Section_03/tweet_emotions.csv")
#csv file source URI: https://www.kaggle.com/pashupatigupta/emotion-detection-from-text

#IBM_Natural_Language_Understanding API >>> Extracting emotions from text
from get_emotion import get_emotion

sadness_list, joy_list, fear_list, disgust_list, anger_list = get_emotion(df["content"])

df["sadness"] = sadness_list
df["joy"] = joy_list
df["fear"] = fear_list
df["disgust"] = disgust_list
df["anger"] = anger_list

df.reset_index(inplace = True, drop = True)

df.to_csv("/Users/doukkim/Section_03/Section_Project/Section_03/emotions.csv")