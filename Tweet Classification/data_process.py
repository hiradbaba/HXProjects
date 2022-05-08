import numpy as np
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import pandas as pd
import nltk
import sklearn.preprocessing
import matplotlib.pyplot as plt
import pickle
import re
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
''' Responsible for preprocessing, loading , saving and transforming data'''
class Preprocessor:
    '''Loads the dataset, Download nltk corpus words and download english stop words'''
    def __init__(self,path):
        print("Loading Data...")
        self.df = pd.read_csv(path,encoding = "ISO-8859-1")
        self.words = set(nltk.corpus.words.words())
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        

    '''Removes null values from data set and applies character removal'''
    def clean_data(self):
        df = pd.DataFrame({'text':self.df['text'],'gender':self.df['gender']})
        not_null_df = df[df['gender'].notna()]
        not_null_df = not_null_df[not_null_df['text'].notna()]
        nf = not_null_df[not_null_df['gender'] != 'unknown']
        #character removal
        nf['text'] = nf['text'].apply(lambda x:self.remove_trivial(x))
        df = nf[nf['text'].notna()]
        df = df[df['gender'].notna()]
        self.df = df
        print("Clean data: ",len(self.df))

    '''Applies stemming,Lemmatization and single character removal on raw data'''
    def nlp_pipeline(self):
        self.df['text'] = self.df['text'].apply(lambda x: self.stem_lem(x))
        print("NLP data: ",len(self.df))

    '''Applies data cleaning and nlp pipeline on raw data and saves it into a file with respect to the name in parameters'''
    def nlp_data(self,save=False,name='tweet_final2'):
        self.clean_data()
        self.nlp_pipeline()
        self.df = self.df[self.df['text'].notna()]
        self.df = self.df[self.df['gender'].notna()]
        if save:
           new_df = pd.DataFrame(data={'text':self.df['text'],'gender':self.df['gender']}) 
           new_df.to_csv(f'data/{name}.csv',index=False)
           #encoding issue with converting ISO format to utf which results turning records with strange characters to null
           new_df = pd.read_csv(f'data/{name}.csv')
           df = new_df[new_df['text'].notna()]
           df.to_csv(f'data/{name}.csv',index=False)
           self.df = df
    '''Trains a Doc2Vec model on our dataset and saves it for furthur use'''
    def create_d2v(self,size=50,window=3,min_count=1,epochs=1500,save=False,name='d2v2'):
        sentences = self.df['text'].to_numpy()
        tokenized = []
        for s in sentences:
            tokenized.append(word_tokenize(s))
        sent_train = tokenized[:int(0.8*len(tokenized))]
        tagged_data = [TaggedDocument(d,[i]) for i,d in enumerate(sent_train)]
        model = Doc2Vec(tagged_data,vector_size = size, window=window, min_count=min_count, epochs=epochs)
        if save:
            with open(f'nlp_models/{name}.pkl','wb') as f:
                pickle.dump(model,f)
        return model
    '''loads doc2vec model to convert a single text into a vector'''
    def infer_vector(self,d2v_name,tweet):
        tweet = self.remove_trivial(tweet)
        tweet = self.stem_lem(tweet)
        with open(f'nlp_models/{d2v_name}.pkl','rb') as f:
            d2v = pickle.load(f)
            return d2v.infer_vector(word_tokenize(tweet))

    '''Transform data into vectors with d2v model and stores it in a csv file'''
    def transform_data(self,d2v_name,save=False,name='transform_data2'):
        with open(f'nlp_models/{d2v_name}.pkl','rb') as f:
            d2v = pickle.load(f)
            tf = self.df['text'].apply(lambda x: d2v.infer_vector(word_tokenize(x)))
            X = np.array([np.array(x) for x in tf.values])
            le = LabelEncoder()
            y = le.fit_transform(self.df['gender'].to_numpy())
            with open('nlp_models/encoder2.pkl','wb') as f:
                pickle.dump(le,f)
            matrix = np.column_stack((X,y))
            if save:
                pd.DataFrame(matrix).to_csv(f"data/{name}.csv",index=False)
            return matrix
    '''Loads dataset and encoder to create a new dataset, splits data with test_size of 0.2 and retur'''
    def create_training_test_set(self,name,encoder =None):
        print("Creating test/training set and saving to last_*.csv")
        df = np.loadtxt(f'data/{name}.csv',skiprows=1,delimiter=',')
        X,y = df[:,:-1],df[:,-1].astype('int')
        if encoder is not None:
            with open(f'nlp_models/{encoder}.pkl','rb') as f:
                le = pickle.load(f)
            y = le.inverse_transform(y)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        pd.DataFrame(np.column_stack((X_train,y_train))).to_csv('data/last_train.csv',index=False)
        pd.DataFrame(np.column_stack((X_test,y_test))).to_csv('data/last_test.csv',index=False)
        return X_train,X_test,y_train,y_test

    '''Removes a texts with a single character'''
    def single_character_remove(self,text):
        return " ".join([word for word in text.split() if len(word)>1])
    '''Apply stemming to text'''   
    def stem(self,text):
        ps = nltk.PorterStemmer()
        return " ".join([ps.stem(word) for word in text.split()])

    '''Apply lemmatization to text'''
    def lemmatize(self,text):
        wn = nltk.WordNetLemmatizer()
        lst = [wn.lemmatize(word) for word in text.split()]
        return " ".join(lst)
    
    '''Apply 3 above functions together'''
    def stem_lem(self,text):
        return self.single_character_remove(self.lemmatize(self.stem(text)))

    '''Remove trivial characters, http links,stop words, numbers and converts text to lower case'''
    def remove_trivial(self,string):
        words = self.words
        stop_words = self.stop_words
        #remove links
        string = re.sub(r"http\S+", "", string)
        lst = string.split()
        #remove user mentions
        n_lst = [s for s in lst if not s.startswith('@')]
        string =  " ".join(n_lst)
        #convert to lower case
        string = string.lower()
        #remove stop words
        string = " ".join(w for w in nltk.wordpunct_tokenize(string) if w not in stop_words)
        #remove numbers
        string = re.sub(r"[^a-zA-Z]"," ", string)
        #remove extra spaces
        string = re.sub(' +', ' ', string)
        return string


    
    
    

   


        