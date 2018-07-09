
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import pandas as pd
import re

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from conllu import parse, parse_tree #using the conllu for parsing (pip install conllu)


# In[2]:


# the corpus I am suggesting is from https://deep-sequoia.inria.fr/versions/
corpus = nltk.corpus.reader.plaintext.PlaintextCorpusReader(".", "sequoia-8.1/sequoia.deep.conll")
data = re.sub(r" +", r"\t", corpus.raw())
sentences = parse(data)


# The corpus contains 3099 sentences.
# 
# Number of sentences for each sub-domain :
# - 561 sentences	Europarl	 file= Europar.550+fct.mrg
# - 529 sentences	EstRepublicain   file= annodis.er+fct.mrg
# - 996 sentences	French Wikipedia file= frwiki_50.1000+fct.mrg
# - 574 sentences	EMEA (dev)  	 file= emea-fr-dev+fct.mrg
# - 544 sentences	EMEA (test) 	 file= emea-fr-test+fct.mrg, among which 101 were removed (because duplicates) in surface version 6.0 and 1.0 deep version.

# In[3]:


#a preview of every sentence
sentences[1]


# In[4]:


print('There are %d sentences in the corpus'%len(sentences))


# In[5]:


sentlen = []

for sent in sentences:
    sentlen.append(len(sent))


# In[6]:


('The average sentence lenght is %.2f'%np.mean(sentlen))


# In[7]:


tags = {}

for sent in sentences:
    for word in sent: 
        if word['xpostag'] in tags:
            tags[word['xpostag']] += 1
        else:
            tags[word['xpostag']] = 1


# In[8]:


plt.subplots(figsize=(18,5))
sns.categorical.barplot(x=list(tags.keys()), y=list(tags.values()),color='c')
plt.title('Count of tags')


# In[9]:


def text(t):
    text = []
    for i in t:
        text.append(i['form'])
    return re.sub('\' ','\'',' '.join(text))


# In[10]:


text(sentences[3])


# In[11]:


# train the NLTK POSTAGGER on the corpus
# create a tupple for every sentence such that (token, tag)
tag_sent = []
for item in sentences:
    sent = []
    for word in item:
        text = word['form']
        tag = word['upostag']
        sent.append((text,tag))
    tag_sent.append(sent)


# In[12]:


#split to train and test sets
size = int(len(tag_sent) * 0.9)
train_sents = tag_sent[:size]
test_sents = tag_sent[size:]


# In[13]:


# train Unigramtagger
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)


# In[14]:


#train a BigramTagger
Bigram_tagger = nltk.BigramTagger(train_sents)
Bigram_tagger.evaluate(test_sents)


# In[15]:


#train a Trigramtagger
Trigram_tagger = nltk.TrigramTagger(train_sents)
Trigram_tagger.evaluate(test_sents)


# In[16]:


#Ensemble the Uni and Bi - gram taggers
t1 = nltk.UnigramTagger(train_sents)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)


# In[17]:


#Ensemble the Uni and Bi and Tri - gram taggers, the performance is slightly down, we stick with Uni and Bi -gram taggers
t3 = nltk.TrigramTagger(train_sents, backoff=t2)
t3.evaluate(test_sents)


# In[18]:


#prepare the curl command
import requests

token = ''

headers = {
    'Authorization': 'LettriaProKey %s'%token,
    'Content-Type': 'application/json',
}


# In[19]:


# we create a list that have full sentence to send to Lettria api, and tupples of tokens to measure accuracy later (sentence, (tokens,tags))
# note, for this draft, no preprocessing of text is done. for example when there is the apostrophe like l'humain
# it will be: (l',humain). it could be interesting to investigate if there is a change in performance
sentences_ = []
for sent in test_sents:
    s = ' '.join(list(map(lambda x: x[0], sent)))
    sentences_.append((s,sent))



# In[20]:


sentences_[1]


# In[21]:


# make Lettria text compliant sentence
testsent = '"' + sentences_[0][0] + '"'
data = '{ "text": %s }'%testsent
data


# In[22]:


response = requests.post('https://api.lettria.com/main', headers=headers, data=data)


# In[23]:


# test response
response.json()


# In[24]:


r = response.json()['postagger']


# In[25]:


for i,j in zip(r,sentences_[0][1]):
    print(i,j)


# In[26]:


#To compute accuracy for this sentence i assumed the following to be True:
# PUNCT == PONCT, NP == N, CC == C, P == P+D. The result is 20 correct tags with test_set
# The three only mistakes: two are on numbers, and one was on the word "juge"
print('Accuracy on the test sentence using Lettria tagger is {:.2f}%'.format(20/len(r)*100))


# In[27]:


testsentntlk = testsent.split()
for i, j in zip(t3.tag(testsentntlk),sentences_[0][1]):
    print(i,j)


# In[28]:


# The three only mistakes: two are on punctuations, and one was on the word "clôt"
print('Accuracy on the test sentence using NTLK tagger is {:.2f}%'.format(20/len(r)*100))


# In[29]:


# More analysis to come

