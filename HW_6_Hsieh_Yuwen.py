import pandas as pd
import nlp
from sklearn.naive_bayes import MultinomialNB as mnbc
from sklearn.naive_bayes import BernoulliNB as bnbc
from sklearn.model_selection import train_test_split as tts
from plotnine import *
from dfply import *
theme_update(plot_background = element_rect(fill = "gold"),panel_background = element_rect(fill = "silver",colour = "blue",size = 1.99,linetype = "solid"),plot_title = element_text(hjust = 0.5))
# Data import
df0=pd.read_csv(r"deception_data_converted_final.tsv",sep = '\t').drop(columns='lie')
# Convert list of reviews into list of token lists
tk=nlp.list_of_token_lists(df0.review,label=df0.sentiment)
# Vectorize data without manual cleaning by using existing vectorizer with unigram.
dfc1=tk.vectorize(method='count',stop_words='english')
# Data splitting
train,test=tts(dfc1,train_size=.8,stratify=dfc1.label,random_state=87)
train_idx=train.index.to_list()
test_idx=test.index.to_list()
# For loop prep. for modeling
train_set=[train]
test_set=[test]
mods=['Multinomial Naive Bayes']
method=['Unigram']
cleaning=['Sklearn']
stop_rm=['Y']
# Vectorize data without manual cleaning by using existing vectorizer with bigrams.
dfc2=tk.vectorize(method='count',stop_words='english',ngram_range=(1,2))
# For loop prep. for modeling
train_set+=[dfc2.iloc[train_idx,:]]
test_set+=[dfc2.iloc[test_idx,:]]
mods+=['Multinomial Naive Bayes']
method+=['Bigrams']
cleaning+=['Sklearn']
stop_rm+=['Y']
# Vectorize data without manual cleaning and stopwords removal by using existing vectorizer with bigrams.
dfc1r=tk.vectorize(method='count')
# For loop prep. for modeling
train_set+=[dfc1r.iloc[train_idx,:]]
test_set+=[dfc1r.iloc[test_idx,:]]
mods+=['Multinomial Naive Bayes']
method+=['Unigram']
cleaning+=['Sklearn']
stop_rm+=['N']
# Vectorize data without manual cleaning by using existing vectorizer with unigrams.
dfb1=tk.vectorize(method='bool',stop_words='english')
# For loop prep. for modeling
train_set+=[dfb1.iloc[train_idx,:]]
test_set+=[dfb1.iloc[test_idx,:]]
mods+=['Bernoulli Naive Bayes']
method+=['Unigram']
cleaning+=['Sklearn']
stop_rm+=['Y']
# Vectorize data without manual cleaning by using existing vectorizer with bigrams.
dfb2=tk.vectorize(method='bool',stop_words='english',ngram_range=(1,2))
# For loop prep. for modeling
train_set+=[dfb2.iloc[train_idx,:]]
test_set+=[dfb2.iloc[test_idx,:]]
mods+=['Bernoulli Naive Bayes']
method+=['Bigrams']
cleaning+=['Sklearn']
stop_rm+=['Y']
# View tokens with non alphabetic characters and more behind the scene.
'''
tk.view_tokens()
tk.view_tokens('\w')
'''
# Cleaning tokens
tk.replace(r'\\','')
tk.replace(r'\.+|-+|!+|/|=',' ')
# vectorize data by frequency count with unigram without removing stopwords
temp=tk.lst_tk_lsts
tk.replace(r"'|’s",'')
tk.stem()
df1cr=tk.vectorize(method='count')
# For loop prep. for modeling
train_set+=[df1cr.iloc[train_idx,:]]
test_set+=[df1cr.iloc[test_idx,:]]
mods+=['Multinomial Naive Bayes']
method+=['Unigram']
cleaning+=['Manual']
stop_rm+=['N']
# Cleaning tokens
tk.lst_tk_lsts=temp
tk.rm_stop_words
tk.replace(r"'|’s",'')
tk.stem()
tk.rm_stop_words
# vectorize data by frequency count with unigram
df1c=tk.vectorize(method='count')
# For loop prep. for modeling
train_set+=[df1c.iloc[train_idx,:]]
test_set+=[df1c.iloc[test_idx,:]]
mods+=['Multinomial Naive Bayes']
method+=['Unigram']
cleaning+=['Manual']
stop_rm+=['Y']
# vectorize data by frequency count with bigram
df2c=tk.vectorize(method='count',ngram_range=(1,2))
# For loop prep. for modeling
train_set+=[df2c.iloc[train_idx,:]]
test_set+=[df2c.iloc[test_idx,:]]
mods+=['Multinomial Naive Bayes']
method+=['Bigrams']
cleaning+=['Manual']
stop_rm+=['Y']
# vectorize data by boolean expression with unigram
df1b=tk.vectorize(method='bool')
# For loop prep. for modeling
train_set+=[df1b.iloc[train_idx,:]]
test_set+=[df1b.iloc[test_idx,:]]
mods+=['Bernoulli Naive Bayes']
method+=['Unigram']
cleaning+=['Manual']
stop_rm+=['Y']
# vectorize data by boolean expression with bigram
df2b=tk.vectorize(method='bool',ngram_range=(1,2))
# For loop prep. for modeling
train_set+=[df2b.iloc[train_idx,:]]
test_set+=[df2b.iloc[test_idx,:]]
mods+=['Bernoulli Naive Bayes']
method+=['Bigrams']
cleaning+=['Manual']
stop_rm+=['Y']
# Visualization of the words
tk.word_freq()
# Naive Bayes fitting
for tr,te,mod in zip(train_set,test_set,mods):
    xtr=tr.drop(columns=['label']);xte=te.drop(columns=['label'])
    ytr=tr['label'];yte=te['label']
    if mod=='Bernoulli Naive Bayes':
        clf=bnbc().fit(xtr,ytr)
    else:
        clf=mnbc().fit(xtr,ytr)
    if tr is train_set[0]:
        scores=[clf.score(xte,yte)]
    else:
        scores+=[clf.score(xte,yte)]
df_comp=pd.DataFrame({'model':mods,'method':method,'clean':cleaning,'test_accuracy':scores,'rm_stopwords':stop_rm}).sort_values('test_accuracy',ascending=1==2)
# Visualization of the results
ggplot(df_comp >> mask(X.rm_stopwords=='Y'),aes(x='model',y='test_accuracy'))+geom_bar(aes(fill='clean'),stat = 'identity',position='dodge')+facet_wrap('~method')+coord_flip()
ggplot(df_comp >> mask(X.model=='Multinomial Naive Bayes',X.method=='Unigram'),aes(x='rm_stopwords',y='test_accuracy'))+geom_bar(aes(fill='clean'),stat = 'identity',position='dodge')+xlab('Removal of stopwords')