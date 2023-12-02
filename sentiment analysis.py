#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[2]:


tweets = pd.read_csv("D:/Data Science/Practice/Twitter US Airline Sentiment Analysis/tweets.csv") #10980 rows 12 cols


# In[3]:


tweets.head()


# In[4]:


print(tweets['negativereason_gold'].nunique())
print(tweets['negativereason_gold'].value_counts(),"\n")


# ## Cleaning

# In[5]:


drop_cols = ['airline_sentiment_gold','name','tweet_id', 'retweet_count','tweet_created','user_timezone','tweet_coord','tweet_location']
tweets.drop(drop_cols, axis = 1, inplace=True)


# In[6]:


tweets.head(3)


# In[7]:


# Stop Words
stops = stopwords.words('english')
stops += list(punctuation)
stops += ['flight','airline','flights','AA']


# In[8]:


abbreviations = {'ppl': 'people','cust':'customer','serv':'service','mins':'minutes','hrs':'hours','svc': 'service',
           'u':'you','pls':'please'}

tweet_index = tweets[~tweets.negativereason_gold.isna()].index

for index, row in tweets.iterrows():
    tweet = row.text
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) #remove links
    tweet = re.sub('@[^\s]+','',tweet) #remove usernames
    tweet = re.sub('[\s]+', ' ', tweet) #remove additional whitespaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #replace '#abc' with abc
    tweet = tweet.strip('\'"') #trim tweet
    words = []
    for word in tweet.split():
#         if not has Numbers(word):
        if word.lower() not in stops:
            if word in list(abbreviations.keys()):
                words.append(abbreviations[word])
            else:
                words.append(word.lower())   
    tweet = " ".join(words)
    tweet = " %s %s" % (tweet, row.airline)
    row.text = tweet
    if index in tweet_index:
        row.text = " %s %s" % (row.text, row.negativereason_gold)

del tweets['negativereason_gold']


# In[9]:


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

for index, row in tweets.iterrows():
    row.text = deEmojify(row.text)


# In[10]:


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

for index, row in tweets.iterrows():
    words = row.text.split()
    new_words = []
    for word in words:
        if not hasNumbers(word):
            new_words.append(word)
    row.text = " ".join(new_words)


# In[11]:


def convert_Sentiment(sentiment):
    if  sentiment == "positive":
        return 2
    elif sentiment == "neutral":
        return 1
    elif sentiment == "negative":
        return 0

tweets.airline_sentiment = tweets.airline_sentiment.apply(lambda x : convert_Sentiment(x))


# In[12]:


tweets.head()


# ## Creating vocab and data formatting

# In[13]:


v = TfidfVectorizer(analyzer='word', max_features=3150, max_df = 0.8, ngram_range=(1,1))
x= v.fit_transform(tweets.text)
y= tweets['airline_sentiment']


# In[14]:


X_train , X_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[15]:


smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)


# In[16]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
pred1 = clf.predict(X_test)
accuracy_score(pred1, y_test)


# In[17]:


clf = SVC()
clf.fit(X_train, y_train)
pred2 = clf.predict(X_test)
accuracy_score(pred2, y_test)


# In[18]:


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred3 = clf.predict(X_test)
accuracy_score(pred3, y_test)


# In[19]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred4 = clf.predict(X_test)
accuracy_score(pred4, y_test)


# In[20]:


clf = MultinomialNB()
clf.fit(X_train, y_train)
pred5 = clf.predict(X_test)
accuracy_score(pred5, y_test)


# In[21]:


clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
pred6 = clf.predict(X_test)
accuracy_score(pred6, y_test)


# In[22]:


clf = XGBClassifier()
clf.fit(X_train, y_train)
pred7 = clf.predict(X_test)
accuracy_score(pred7, y_test)


# In[24]:


cr = classification_report(y_test, pred1)
print("Classification Report:\n----------------------\n", cr)


# In[ ]:




