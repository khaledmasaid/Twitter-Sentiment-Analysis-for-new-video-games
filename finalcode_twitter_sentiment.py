import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from nltk.corpus import stopwords
import string
import numpy as np
from sklearn import manifold
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem import WordNetLemmatizer 
import string
import re
import itertools
#pip install pyspark
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
# pip install lda
import lda
# pip install gensim
import gensim
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA


nba = pd.read_csv(r"nba2k21.csv")
fifa = pd.read_csv(r"fifa21.csv")
cyber = pd.read_csv(r"cyberpunk.csv")
coldwar = pd.read_csv(r"cold war.csv")
valhalla = pd.read_csv(r"valhalla.csv")

#Remove duplicated columns
nba = nba.drop_duplicates(subset=['text'])
fifa = fifa.drop_duplicates(subset=['text'])
cyber = cyber.drop_duplicates(subset=['text'])
coldwar = coldwar.drop_duplicates(subset=['text'])
valhalla = valhalla.drop_duplicates(subset=['text'])

#--------------------------------------------------------------------------------------------------------

#Remove puntuation and stopwords
#Add RT to stopwords? Beceasue it shows up a lot and there is no need for RT
#Possibly drop duplicate from 'user' column?

def fix_abbreviation(data_str):
    data_str = data_str.lower()
    data_str = re.sub(r'\bthats\b', 'that is', data_str)
    data_str = re.sub(r'\bive\b', 'i have', data_str)
    data_str = re.sub(r'\bim\b', 'i am', data_str)
    data_str = re.sub(r'\bya\b', 'yeah', data_str)
    data_str = re.sub(r'\bcant\b', 'can not', data_str)
    data_str = re.sub(r'\bdont\b', 'do not', data_str)
    data_str = re.sub(r'\bwont\b', 'will not', data_str)
    data_str = re.sub(r'\bid\b', 'i would', data_str)
    data_str = re.sub(r'wtf', 'what the fuck', data_str)
    data_str = re.sub(r'\bwth\b', 'what the hell', data_str)
    data_str = re.sub(r'\br\b', 'are', data_str)
    data_str = re.sub(r'\bbro\b', 'brother', data_str)
    data_str = re.sub(r'\bu\b', 'you', data_str)
    data_str = re.sub(r'\bgotta\b', 'got to', data_str)
    data_str = re.sub(r'\bk\b', 'OK', data_str)
    data_str = re.sub(r'\bda\b', 'the', data_str)
    data_str = re.sub(r'\bno+\b', 'no', data_str)
    data_str = re.sub(r'\bcoo+\b', 'cool', data_str)
    # leave this step to the next function
    #data_str = re.sub(r'rt\b', '', data_str)
    data_str = data_str.strip()
    return data_str
fix_abbreviation_udf = udf(fix_abbreviation, StringType())

nba['Filteredtext-V1'] = nba['text'].map(fix_abbreviation)
fifa['Filteredtext-V1'] = fifa['text'].map(fix_abbreviation)
cyber['Filteredtext-V1'] = cyber['text'].map(fix_abbreviation)
coldwar['Filteredtext-V1'] = coldwar['text'].map(fix_abbreviation)
valhalla['Filteredtext-V1'] = valhalla['text'].map(fix_abbreviation)


# filter the Replied message and the web links

def remove_replied_and_url(text):
    # remove the text that described the condition of replying to someone 
    if 'rt @' in text:
        if len(text.split(": ")) < 2:
            text = text.split(": ")[0]
        else:
            text = text.split(": ")[1]
    # remove the unnecessay URLs
    if "https" in text:
        text = text.split("https")[0]
    return text

nba['Filteredtext-V2'] = nba['Filteredtext-V1'].map(remove_replied_and_url)
fifa['Filteredtext-V2'] = fifa['Filteredtext-V1'].map(remove_replied_and_url)
cyber['Filteredtext-V2'] = cyber['Filteredtext-V1'].map(remove_replied_and_url)
coldwar['Filteredtext-V2'] = coldwar['Filteredtext-V1'].map(remove_replied_and_url)
valhalla['Filteredtext-V2'] = valhalla['Filteredtext-V1'].map(remove_replied_and_url)


nba = nba.drop_duplicates(subset=['Filteredtext-V2'], keep="last")
fifa = fifa.drop_duplicates(subset=['Filteredtext-V2'], keep="last")
cyber = cyber.drop_duplicates(subset=['Filteredtext-V2'], keep="last")
coldwar = coldwar.drop_duplicates(subset=['Filteredtext-V2'], keep="last")
valhalla = valhalla.drop_duplicates(subset=['Filteredtext-V2'], keep="last")


# Lemmatization
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

nba["Filteredtext-V3"] = nba["Filteredtext-V2"].apply(lambda text: lemmatize_words(text))
fifa["Filteredtext-V3"] = fifa["Filteredtext-V2"].apply(lambda text: lemmatize_words(text))
cyber["Filteredtext-V3"] = cyber["Filteredtext-V2"].apply(lambda text: lemmatize_words(text))
coldwar["Filteredtext-V3"] = coldwar["Filteredtext-V2"].apply(lambda text: lemmatize_words(text))
valhalla["Filteredtext-V3"] = valhalla["Filteredtext-V2"].apply(lambda text: lemmatize_words(text))

### Remove frequent words that are not important
def frequencycount(x):
  cnt = Counter()
  for text in x["Filteredtext-V3"].values:
    for word in text.split():
        cnt[word] += 1
  return cnt.most_common(10)

a = frequencycount(nba)
b = frequencycount(fifa)
c = frequencycount(cyber)
d = frequencycount(coldwar)
e = frequencycount(valhalla)


FREQWORDS1 = set([w for (w, wc) in a])
FREQWORDS2 = set([w for (w, wc) in b])
FREQWORDS3 = set([w for (w, wc) in c])
FREQWORDS4 = set([w for (w, wc) in d])
FREQWORDS5 = set([w for (w, wc) in e])

l =[FREQWORDS1,FREQWORDS2,FREQWORDS3,FREQWORDS4,FREQWORDS5]
def remove_freqwords1(text):
  """custom function to remove the frequent words"""
  return " ".join([word for word in str(text).split() if word not in FREQWORDS1])
def remove_freqwords2(text):
  """custom function to remove the frequent words"""
  return " ".join([word for word in str(text).split() if word not in FREQWORDS2])
def remove_freqwords3(text):
  """custom function to remove the frequent words"""
  return " ".join([word for word in str(text).split() if word not in FREQWORDS3])
def remove_freqwords4(text):
  """custom function to remove the frequent words"""
  return " ".join([word for word in str(text).split() if word not in FREQWORDS4])
def remove_freqwords5(text):
  """custom function to remove the frequent words"""
  return " ".join([word for word in str(text).split() if word not in FREQWORDS5])

nba["Filteredtext-V4"] = nba["Filteredtext-V3"].apply(lambda text: remove_freqwords1(text))
fifa['Filteredtext-V4'] = fifa['Filteredtext-V3'].apply(lambda text: remove_freqwords2(text))
cyber['Filteredtext-V4'] = cyber['Filteredtext-V3'].apply(lambda text: remove_freqwords3(text))
coldwar['Filteredtext-V4'] = coldwar['Filteredtext-V3'].apply(lambda text: remove_freqwords4(text))
valhalla['Filteredtext-V4'] = valhalla['Filteredtext-V3'].apply(lambda text: remove_freqwords5(text))


# tokenize the text
stop_words = set( stopwords.words('english') + list(punctuation))

def filtercomments(text):
  tokenize = word_tokenize(text)
  remove_stopwords = [i.lower() for i in tokenize if i not in stop_words]
  return list(set(remove_stopwords))
  
nba['Filteredtext-V5'] = nba['Filteredtext-V4'].map(filtercomments)
fifa['Filteredtext-V5'] = fifa['Filteredtext-V4'].map(filtercomments)
cyber['Filteredtext-V5'] = cyber['Filteredtext-V4'].map(filtercomments)
coldwar['Filteredtext-V5'] = coldwar['Filteredtext-V4'].map(filtercomments)
valhalla['Filteredtext-V5'] = valhalla['Filteredtext-V4'].map(filtercomments)


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# Topic modelling with K-means clustering
#------------------------------------------------------------------------------------------------------

#Term frequency - Inverse Document Frequency TF-IDF
tf_mod = CountVectorizer(
    max_df=0.99,
    max_features=500,
    min_df=0.01,
    tokenizer= word_tokenize,
    ngram_range=(1,1))

tf_matrix = tf_mod.fit_transform(valhalla['Filteredtext-V2']) 
tf_matrix2 = tf_mod.fit_transform(nba['Filteredtext-V2']) 
tf_matrix3 = tf_mod.fit_transform(fifa['Filteredtext-V2']) 
tf_matrix4 = tf_mod.fit_transform(coldwar['Filteredtext-V2']) 
tf_matrix5 = tf_mod.fit_transform(cyber['Filteredtext-V2']) 
#------------------------------------------------------------------------------------------------------


kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(tf_matrix)

kmeans_results = valhalla
clusters = kmeans_model.labels_.tolist()
kmeans_results['cluster'] = clusters

cluster_size = kmeans_results['cluster'].value_counts().to_frame()
cluster_size


pca = KernelPCA(n_components=2)
tfidf_matrix_np=tf_matrix.toarray()
X = pca.fit_transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:,1]

pca_df = pd.DataFrame(dict(x = xs, y = ys, Cluster = clusters ))
plt.subplots(figsize=(16,9))
sns.scatterplot('x', 'y', data=pca_df, hue='Cluster')


#------------------------------------------------------------------------------------------------------
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(tf_matrix2)

kmeans_results = nba
clusters = kmeans_model.labels_.tolist()
kmeans_results['cluster'] = clusters


cluster_size = kmeans_results['cluster'].value_counts().to_frame()
cluster_size


pca = KernelPCA(n_components=2)
tfidf_matrix_np=tf_matrix2.toarray()
X = pca.fit_transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:,1]

pca_df = pd.DataFrame(dict(x = xs, y = ys, Cluster = clusters ))
plt.subplots(figsize=(16,9))
sns.scatterplot('x', 'y', data=pca_df, hue='Cluster')

#------------------------------------------------------------------------------------------------------
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(tf_matrix3)

kmeans_results = fifa
clusters = kmeans_model.labels_.tolist()
kmeans_results['cluster'] = clusters


cluster_size = kmeans_results['cluster'].value_counts().to_frame()
cluster_size


pca = KernelPCA(n_components=2)
tfidf_matrix_np=tf_matrix3.toarray()
X = pca.fit_transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:,1]

pca_df = pd.DataFrame(dict(x = xs, y = ys, Cluster = clusters ))
plt.subplots(figsize=(16,9))
sns.scatterplot('x', 'y', data=pca_df, hue='Cluster')
#------------------------------------------------------------------------------------------------------
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(tf_matrix4)

kmeans_results = coldwar
clusters = kmeans_model.labels_.tolist()
kmeans_results['cluster'] = clusters


cluster_size = kmeans_results['cluster'].value_counts().to_frame()
cluster_size


pca = KernelPCA(n_components=2)
tfidf_matrix_np=tf_matrix4.toarray()
X = pca.fit_transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:,1]

pca_df = pd.DataFrame(dict(x = xs, y = ys, Cluster = clusters ))
plt.subplots(figsize=(16,9))
sns.scatterplot('x', 'y', data=pca_df, hue='Cluster')
#------------------------------------------------------------------------------------------------------
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(tf_matrix5)

kmeans_results = cyber
clusters = kmeans_model.labels_.tolist()
kmeans_results['cluster'] = clusters


cluster_size = kmeans_results['cluster'].value_counts().to_frame()
cluster_size


pca = KernelPCA(n_components=2)
tfidf_matrix_np=tf_matrix5.toarray()
X = pca.fit_transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:,1]

pca_df = pd.DataFrame(dict(x = xs, y = ys, Cluster = clusters ))
plt.subplots(figsize=(16,9))
sns.scatterplot('x', 'y', data=pca_df, hue='Cluster')

#------------------------------------------------------------------------------------------------------

# sentiment analysis

analyzer = SentimentIntensityAnalyzer()

def VADER_score(text):
    va = analyzer.polarity_scores(text)['compound']
    return va

def VADER_cate(score):
    # negative < -0.05 <= neutral <= 0.05 < positive
    if score > 0.05:
        cate = "POS"
    elif score < -0.05:
        cate = "NEG"
    else:
        cate = "NEU"
    return cate

nba['VADER_score'] = nba['Filteredtext-V2'].map(VADER_score)
fifa['VADER_score'] = fifa['Filteredtext-V2'].map(VADER_score)
cyber['VADER_score'] = cyber['Filteredtext-V2'].map(VADER_score)
coldwar['VADER_score'] = coldwar['Filteredtext-V2'].map(VADER_score)
valhalla['VADER_score'] = valhalla['Filteredtext-V2'].map(VADER_score)

nba['VADER_Category'] = nba['VADER_score'].map(VADER_cate)
fifa['VADER_Category'] = fifa['VADER_score'].map(VADER_cate)
cyber['VADER_Category'] = cyber['VADER_score'].map(VADER_cate)
coldwar['VADER_Category'] = coldwar['VADER_score'].map(VADER_cate)
valhalla['VADER_Category'] = valhalla['VADER_score'].map(VADER_cate)

# percentage of each df
def distribution(df, df_name):
    counts = df['VADER_Category'].value_counts()
    total = df.shape[0]
    print("for dataset", df_name, ": ", round(counts['POS']/total*100,2), \
          "% of positive sentinment, and", round(counts['NEG']/total*100,2), \
          '% of negative sentiment and', round(counts['NEU']/total*100,2), \
          '% of neutral sentiment.')
    
distribution(nba, "NBA")
distribution(fifa, "FIFA")
distribution(cyber, "CyberPunk 2077")
distribution(coldwar, "Cold War")
distribution(valhalla, "Valhalla")

#------------------------------------------------------------------------------------------------------
#Topic modeling

### visualize the tokenized data headline
processed_docs = nba["Filteredtext-V5"]
type(processed_docs)

### Bag of words

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

### Filter tokens appear in less than 15 documents or more than 1 document and keep the top 100000 documents
dictionary.filter_extremes(no_below=15,no_above=1, keep_n=100000)

### create dictionary containing words count
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

### LDA model - select ten topics
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

#For each topic, examine the weight of each word appearing in that topic
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

### Repeat same step for other four games 
processed_docs2 = valhalla["Filteredtext-V5"]
processed_docs3 = fifa["Filteredtext-V5"]
processed_docs4 = cyber["Filteredtext-V5"]
processed_docs5 = coldwar["Filteredtext-V5"]


dictionary2 = gensim.corpora.Dictionary(processed_docs2)
dictionary3 = gensim.corpora.Dictionary(processed_docs3)
dictionary4 = gensim.corpora.Dictionary(processed_docs4)
dictionary5 = gensim.corpora.Dictionary(processed_docs5)

### Filter tokens 
dic = [dictionary2,dictionary3,dictionary4,dictionary5]
#proc = [processed_docs2,processed_docs3,processed_docs4,processed_docs5]
#bow_corpus = [bow_corpus2,bow_corpus3,bow_corpus4, bow_corpus5]
#lda_model = [lda_model2,lda_model3,lda_model4,lda_model5]
#new_dic = []
for i in range(len(dic)):
  dic[i].filter_extremes(no_below=15,no_above=1, keep_n=100000)

bow_corpus2 = [dictionary2.doc2bow(doc) for doc in processed_docs2]
bow_corpus3 = [dictionary3.doc2bow(doc) for doc in processed_docs3]
bow_corpus4 = [dictionary4.doc2bow(doc) for doc in processed_docs4]
bow_corpus5 = [dictionary5.doc2bow(doc) for doc in processed_docs5]

lda_model2 = gensim.models.LdaMulticore(bow_corpus2, num_topics=10, id2word=dictionary2, passes=2, workers=2)
lda_model3 = gensim.models.LdaMulticore(bow_corpus3, num_topics=10, id2word=dictionary3, passes=2, workers=2)
lda_model4 = gensim.models.LdaMulticore(bow_corpus4, num_topics=10, id2word=dictionary4, passes=2, workers=2)
lda_model5 = gensim.models.LdaMulticore(bow_corpus5, num_topics=10, id2word=dictionary5, passes=2, workers=2)

#For each topic, examine the weight of each word appearing in that topic
models = [lda_model2,lda_model3,lda_model4,lda_model5]
for i in range(len(models)):
  for idx, topic in models[i].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))



#visualize the dominant topic and its percentage in each document 
def format_topics_sentences(ldamodel=None, corpus=bow_corpus5, texts=processed_docs5):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model5, corpus=bow_corpus5, texts=processed_docs5)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)




