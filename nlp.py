def replace(pattern,by,tk_lst):
        '''
        Function: Replace pattern of tokens with regular expression.
        Return: A list of lists containing tokens.
        Parameters:
            pattern - The pattern to be replaced using regular expression.
            by - The string for replacement.
            tk_lst - A list of lists containing tokens.
        '''
        import re
        return [[re.sub(pattern,by,s) for s in lst] for lst in tk_lst]

# Remove some redundant letters at the start or end of words
def trim_repeats(s):
    '''
    Return a string with cleaned repeat letters at the start or end.
    '''
    import numpy as np
    if len(s)<3: return s
    l=list(s)
    for i in (range(1,len(l))):
        if l[i] == l[0]:
            l[i]=''
        else: 
            for j in -np.arange(2,len(l)+1):
                if l[j] == l[-1]:
                    l[j]=''
                else: 
                    return ''.join(l)

def rdnt_mid(s):
    '''
    Clean repeating letters in the middle of the string.
    '''
    import numpy as np
    if type(s) != str: return ''
    if len(s)<5: return s
    ref=s[1]
    rpt=0
    rm_list=[]
    for i in range(2,len(s)-1):
        if s[i]==ref:
            rpt+=1
            if i==len(s)-2 and rpt>0: rm_list+=[s[i-rpt:i+1]]
        else:
            if rpt>0: 
                rm_list+=[s[i-rpt-1:i]]
            ref=s[i]
            rpt=0
    if np.sum([len(ss)>1 for ss in rm_list])>0:
        for k in rm_list:
            s=s.replace(k,k[0])
    return s

def clean_repeats(s):
    '''
    Clean the repeating redundant letters.
    '''
    return rdnt_mid(trim_repeats(s))

def nlp():
    import os
    os.system("start \"\" "+r'"C:\Users\Arnold\Google Drive\IST 664 Natural Language Processing\NLP notes.pdf"')

def word_cloud(s):
    '''
    Function: Create a word_cloud from a string.
    '''
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud = WordCloud(background_color='white').generate(s)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

class list_of_token_lists:
    '''
    Function: Tokenize a list of strings with all non-alphabetic characters stripped.
    Parameters:
        lst(list like object) - A list of strings.
        lower(bool;default True) - Whether to make all tokens lower case.
        label(list like object;default None) - Labels for the strings.
        regex(regular expression) - If not none, regular expression is used     to intialize the tokens.
    Attributes:
        raw - A list of raw strings from input.
        lst_tk_lsts - A list of list of tokens.
        stop_words - A list of stop-words.
        tdm - Sparse matrix of vectoried data if vectorized.
        features - Tokens used for vectorization.
        vecorizer - vecorizer
    '''
    def __init__(self,lst,regex=None,lower=1==1,label=None):
        import nltk
        import re
        self.raw=list(lst)
        self.labels=label
        try: 
            self.labels=list(label)
        except:
            pass
        lst=[str(x) for x in lst]
        if lower: lst=[x.lower() for x in lst]
        if regex == None:
            lst=[x.split() for x in lst]
            lst=replace(r'[^a-zA-Z]+$','',lst)
            lst=replace(r'^[^a-zA-Z]+','',lst)
            self.lst_tk_lsts=[[re.sub('\s+',' ',t) for t in l if t != ''] for l in lst]
        else:
            self.lst_tk_lsts=[re.findall(regex,x) for x in lst]
        self.stop_words=nltk.corpus.stopwords.words('english')+["i'm","we've",  "who've","i've","ain't","they're","we'll","they've","we're","i'd",   "i'll","there're","he'd","could've","he'll","it'd","it'll","might've","must've","would've","who're","who'd","what're","we'd","this'll","they'll","they'd","that'd","she'll","she'd","y'all","ya'll","there've","there'll","there'd","nobody'll","c'mon","thats","todays","youre","whos","well","im","its","what's","that's","he's","one's","today's","yeah","xyz","yelp","wqr","usualy","tea","time","west","thing","dog","dine","dude","eri","eat","abc","use","place","went","one","order","tabl","dish","food","serv","servic","back","peopl","waitr","diner","chese","salad","experi","piza","would","minut","plate","waiter"]

    @property
    def clear_space(self):
        '''
        Function clean up spaces in the tokens.
        '''
        self.lst_tk_lsts=[' '.join([t for t in l if type(t)==str]).split() for l in self.lst_tk_lsts]

    def replace(self,pattern,by):
        '''
        Function: Replace pattern of tokens with regular expression.
        Parameters:
            pattern - The pattern to be replaced using regular expression.
            by - The string for replacement.
        '''
        import re
        self.backup=self.lst_tk_lsts
        self.lst_tk_lsts=[[re.sub(pattern,by,s) for s in lst] for lst in self.lst_tk_lsts]
        if ' ' in by: self.clear_space
        
    @property
    def clean(self):
        '''
        Clean the repeating redundant letters, and remove words less than 3 characters.
        '''
        self.lst_tk_lsts= [[clean_repeats(s) for s in l if len(s)>2] for l in self.lst_tk_lsts]
        self.lst_tk_lsts= [[s for s in l if len(s)>2] for l in self.lst_tk_lsts]

    @property
    def undo(self):
        self.lst_tk_lsts=self.backup

    @property
    def reset(self):
        self.lst_tk_lsts=self.raw

    def view_tokens(self,pat='[^a-zA-Z]',stick='\t'):
        '''
        Function: Check all the tokens with a pattern using regular expression.
        Parameters:
            pat - The regular expression to search for.
            stick - A string indicating how the tokens should be joined to be printed.
        '''
        import re
        all_tokens=list(set([x for lst in self.lst_tk_lsts for x in lst if type(x)==str]))
        all_tokens.sort()
        print(stick.join([s for s in all_tokens if re.search(pat,s)]))

    @property
    def rm_stop_words(self):
        '''
        Function: Remove stopwords.
        '''
        self.lst_tk_lsts=[[t for t in l if (t not in self.stop_words) and type(t)==str] for l in self.lst_tk_lsts]

    def mod_stop_words(self,wds,add=1==1):
        '''
        Function: Add words to the list of stop words.
        Parameters:
            wds - List of stop words to add.
            add - Whether to add or remove words.
        '''
        if add:
            self.stop_words+=[w for w in wds if w not in self.stop_words]
        else:
            self.stop_words=[w for w in self.stop_words if w not in wds]

    @property
    def lemmatize(self):
        '''
        Function: Lemmatize all tokens.
        '''
        self.backup=self.lst_tk_lsts
        import nltk
        wnl = nltk.WordNetLemmatizer()
        self.lst_tk_lsts=[[wnl.lemmatize(t) for t in l] for l in self.lst_tk_lsts]

    @property
    def extract(self):
        '''
        Return: List of strings if no labels are provided. Dataframe with labels and cleaned texts if labels are provided.
        '''
        import pandas as pd
        sl=[' '.join(l) for l in self.lst_tk_lsts]
        if self.labels != None: return pd.DataFrame({'label':self.labels,'text':sl})
        return sl

    def stem(self,Porter=1==1):
        '''
        Function: Stem all tokens.
        Porter: If true Porter stemmer is used, if not Lancaster stemmer is used.
        '''
        self.backup=self.lst_tk_lsts
        import nltk
        stemmer=nltk.LancasterStemmer()
        if Porter: stemmer=nltk.PorterStemmer()
        self.lst_tk_lsts=[[stemmer.stem(t) for t in l if type(t)==str] for l in self.lst_tk_lsts]
    
    def vectorize(self,method='tfidf',input='content',encoding='utf-8',decode_error='strict',strip_accents=None,lowercase=True,preprocessor=None,tokenizer=None,stop_words='english',token_pattern='(?u)\\b\\w\\w+\\b',ngram_range=(1, 1),analyzer='word',max_df=1.0,min_df=1,max_features=None,vocabulary=None,binary=False,norm='l2',smooth_idf=True,sublinear_tf=False,return_df=1==2):
        '''
        Function: Vectorize text data.
        Parameters
        ----------
        method : string {'tfidf','bool','count','tf'}, 'tfidf' by default
            If 'Bool', boolean expression is used. This is useful for discrete probabilistic models that model binary events rather than integer
    counts.
            If 'tfidf', tfidf is used.
            If 'count', word count is used.
            If 'tf', term frequency is used.
        
        return_df: bool
            If true, a dataframe is return with labels if possible.
        
        encoding : string, 'utf-8' by default.
            If bytes or files are given to analyze, this encoding is used to    decode.

        decode_error : {'strict', 'ignore', 'replace'}
            Instruction on what to do if a byte sequence is given to analyze that contains characters not of the given `encoding`. By default, it is 'strict', meaning that a UnicodeDecodeError will be raised. Other values are 'ignore' and 'replace'.

        strip_accents : {'ascii', 'unicode', None}
            Remove accents and perform other character normalization during the preprocessing step.
            'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
            'unicode' is a slightly slower method that works on any characters.
            None (default) does nothing.

            Both 'ascii' and 'unicode' use NFKD normalization from   :func:`unicodedata.normalize`.

        preprocessor : callable or None (default)
            Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.

        tokenizer : callable or None (default)
            Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
            Only applies if ``analyzer == 'word'``.

        analyzer : string, {'word', 'char', 'char_wb'} or callable
            Whether the feature should be made of word or character n-grams.
            Option 'char_wb' creates character n-grams only from text inside word boundaries; n-grams at the edges of words are padded with space.

            If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.
        
        stop_words : string {'english'}, list, or None (default)
            If 'english', a built-in stop word list for English is used.
            There are several known issues with 'english' and you should consider an alternative (see :ref:`stop_words`).

            If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.

            If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.

        token_pattern : string
            Regular expression denoting what constitutes a "token", only used if ``analyzer == 'word'``. The default regexp select tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).

        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

        max_df : float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
            If float, the parameter represents a proportion of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        min_df : float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        max_features : int or None, default=None
            If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.

        vocabulary : Mapping or iterable, optional
            Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. If not given, a vocabulary is determined from the input documents. Indices in the mapping should not be repeated and should not have any gap between 0 and the largest index.

        norm : 'l1', 'l2' or None, optional (default='l2')
            Each output row will have unit norm, either:
            * 'l2': Sum of squares of vector elements is 1. The cosine similarity between two vectors is their dot product when l2 norm has been applied.
            * 'l1': Sum of absolute values of vector elements is 1.
            See :func:`preprocessing.normalize`

        use_idf : boolean (default=True)
            Enable inverse-document-frequency reweighting.

        smooth_idf : boolean (default=True)
            Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.

        sublinear_tf : boolean (default=False)
            Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

        Attributes
        ----------
        vocabulary_ : dict
            A mapping of terms to feature indices.
            
        idf_ : array, shape (n_features)
            The inverse document frequency (IDF) vector; only defined if  ``use_idf`` is True.

        stop_words_ : set
            Terms that were ignored because they either:
                - occurred in too many documents (`max_df`)
                - occurred in too few documents (`min_df`)
                - were cut off by feature selection (`max_features`).
            This is only available if no vocabulary was given.
        '''
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pandas as pd
        if method in ['count','bool']:
            vectorizer = CountVectorizer(encoding=encoding,decode_error=decode_error,strip_accents=strip_accents,lowercase=1==2,preprocessor=preprocessor,tokenizer=tokenizer,stop_words=stop_words,token_pattern=token_pattern,ngram_range=ngram_range,analyzer=analyzer,max_df=max_df,min_df=min_df,max_features=max_features,vocabulary=vocabulary,binary=method=='bool')
        elif method in ['tfidf','tf']:
            vectorizer=TfidfVectorizer(encoding=encoding,decode_error=decode_error,strip_accents=strip_accents,lowercase=1==2,preprocessor=preprocessor,tokenizer=tokenizer,analyzer=analyzer,stop_words=stop_words,token_pattern=token_pattern,ngram_range=ngram_range,max_df=max_df,min_df=min_df,max_features=max_features,vocabulary=vocabulary,norm=norm,use_idf=method=='tf',smooth_idf=smooth_idf,sublinear_tf=sublinear_tf)
        else:
            pass
        sl=[' '.join(l) for l in self.lst_tk_lsts]
        self.tdm=vectorizer.fit_transform(sl)
        self.vectorizer=vectorizer
        self.features=vectorizer.get_feature_names()
        if return_df:
            df=pd.DataFrame(self.tdm.toarray(),columns=self.features)
            try: df['label']=self.labels
            except: print('No labels included')
            return df
    
    def word_freq(self,top_n=10,group=1==1,bi=1==1):
        '''
        Function: Show words frequency & word clouds.
        Paramters-
            group: If true produce frequency & word clouds for each tokens list in the list.
            top_n: Number of top frequency words to show.
            bi: Whether to show bigrams or not.
        '''
        if group and self.labels==None: group=1==2
        from nltk import FreqDist as fd
        import pandas as pd
        if not group:
            tks=[t for l in self.lst_tk_lsts for t in l if type(t)==str]
            ndist = pd.Series(fd(tks)).sort_values(ascending=1==2)
            sm=ndist.sum()
            ndist=ndist.map(lambda x: str(round(x/sm*100,2))+'%')
            word_cloud(' '.join(tks))
            print(ndist[:top_n])
            if bi:
                df_bi=self.vectorize(method='count',ngram_range=(2,2),stop_words='english',return_df=1==1)
                if 'label' in df_bi.columns:
                    df_bi.drop(columns=['label'],inplace=1==1)
                sr=df_bi.sum().sort_values(ascending=1==2)
                sr=sr.map(lambda x: str(round(x/sr.sum()*100,1))+'%')
                print(sr[:top_n])
            return None
        if bi:
            df_bi=self.vectorize(method='count',ngram_range=(2,2),stop_words='english',return_df=1==1)
            df=df_bi.groupby('label').sum()
        sr=pd.Series(self.lst_tk_lsts,index=self.labels)
        for i in list(set(self.labels)):
            print(i,':')
            sr0=sr[i].to_list()
            tks=[t for k in sr0 for t in k]
            ndist = pd.Series(fd(tks)).sort_values(ascending=1==2)
            sm=ndist.sum()
            ndist=ndist.map(lambda x: str(round(x/sm*100,2))+'%')
            word_cloud(' '.join(tks))
            print(ndist[:top_n],'\n')
            if bi:
                sr1=df.loc[i,:].sort_values(ascending=1==2)
                sr1=sr1.map(lambda x: str(round(x/sr1.sum()*100,1))+'%')
                print(sr1[:top_n],'\n')