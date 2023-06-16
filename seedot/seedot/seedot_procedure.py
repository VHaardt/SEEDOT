import pandas as pd
import nltk
nltk.download('wordnet')
import tqdm
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('omw-1.4')
from sklearn.linear_model import LogisticRegression
import json
from gensim.models import FastText
import os
from inspect import getsourcefile
import codecs
from io import open
import json


f_list = [2, 3]

class score_functions(object):
    """
    Check score functions values
    """
    def __init__(self, data = None, score_function = None, model = "logistic"):
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("A valid data must be specified.")
        
        if model != "logistic":
            raise ValueError("Currently, the only implemented model is: model = 'logistic'.")

        if score_function is None or score_function not in f_list:
            raise ValueError("A valid score_function (2, 3) must be specified.")
        
        # Check if 'Scores' and 'Text' columns exist
        if 'Score' not in data.columns or 'Text' not in data.columns:
            raise ValueError("The 'data' DataFrame must have 'Scores' and 'Text' columns.")

        # Check the values in the 'Scores' column
        allowed_scores = ['Positive', 'Negative', 'Neutral']
        if not data['Score'].isin(allowed_scores).all():
            raise ValueError("The 'Scores' column must only contain the values 'Positive', 'Negative', and 'Neutral'.")

        # Check the data types in the 'Text' column
        if data['Text'].dtypes != object:
            raise ValueError("The 'Text' column must only contain strings.")
        self.data = data
        self.score_function = score_function
        self.model = model
    
    def division(self):   
        data = self.data[self.data['Score'] != 'Neutral']
        dfl = data
        df_not_na = data[~(data['Text'].isna())]
        raw_text = df_not_na['Text']
        #assegno i valori alla mia y
        y = df_not_na['Score'].tolist()
        text = raw_text.str.lower().str.replace('[^\w\s\d]',' ', regex=True) # \d tiene anche i caratteri numerici
        text = text.str.split()
        lemmatizer = WordNetLemmatizer()
        text = text.apply(lambda x: [lemmatizer.lemmatize(sent) for sent in x])
        for row in text:
            [row.remove(i) for i in row if len(i) < 2]
        train_sentences = []
        for row in text:
            train_sentences.append(' '.join([item for item in row ]))
        vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=2000)
        X = vectorizer.fit_transform(train_sentences)
        #salva in un dizionario i nomi delle features, associando a un codice
        feature_names = vectorizer.get_feature_names_out()
        X = X.toarray()
        X = np.array(X)
        y = np.array(y) 
        M = LogisticRegression(class_weight=None)
        M.fit(X,y)
        pred = M.predict(X)
        coef=M.coef_

        dfl = data.dropna(subset=['Text']).reset_index(drop=True)
        analyzer = SentimentIntensityAnalyzer()
        dfl['scores'] = dfl['Text'].apply(lambda review: analyzer.polarity_scores(review))
        dfl['compound']  = dfl['scores'].apply(lambda score_dict: score_dict['compound'])

        
        if self.score_function == 2:
            j = 0.1
            diz = {}
            while j <= 2.5:
                df1 = pd.DataFrame({"Coeff": coef[0], "Parola": feature_names, "Punteggio": 0})
                df1.sort_values(by=["Coeff"], ascending=False, inplace=True)

                for i in range(len(df1)):
                    if df1.Coeff[i] > j:
                        df1.loc[i, 'Punteggio'] = 4
                    elif df1.Coeff[i] < -j:
                        df1.loc[i, 'Punteggio'] = -4
                    else:
                        df1.drop(i, inplace=True)
                new_words = dict([([a, b]) for a, b in zip(df1['Parola'], df1['Punteggio'])])
                dfl = dfl.dropna(subset=['Text']).reset_index(drop=True)
                #Vader mio
                SIA = SentimentIntensityAnalyzer()
                SIA.lexicon.update(new_words)
                dfl['scores2'] = dfl['Text'].apply(lambda review: SIA.polarity_scores(review))
                dfl['compound2']  = dfl['scores2'].apply(lambda score_dict: score_dict['compound'])
                Fine = pd.DataFrame({'Score': dfl.Score,'Text': dfl.Text,'Vader': dfl.compound,'Vader_new': dfl.compound2})
                Fine['Vader'] = dfl['compound'].apply(lambda x: 'Positive' if x > 0 else 'Negative' )
                Fine['Vader_new'] = dfl['compound2'].apply(lambda x: 'Positive' if x > 0 else 'Negative' )
                v=0
                vn=0
                for i in range(len(Fine)):
                    if Fine.Score[i]==Fine.Vader[i]:
                        v+=1
                    if Fine.Score[i]==Fine.Vader_new[i]:
                        vn+=1
                j += 0.1
                diz[j] = round(((vn / len(Fine)) * 100), 2)
                i = 0
            max_value = max(diz.values())
            max_key = max(diz, key=diz.get)
        elif self.score_function == 3:
            j=0.1
            diz={}
            diz2={}
            while j <= 2.5:
                z=0.1
                while z < j:
                    df1 = pd.DataFrame({"Coeff": coef[0], "Parola": feature_names, "Punteggio": 0})
                    df1.sort_values(by=["Coeff"], ascending=False, inplace=True)

                    for i in range(len(df1)):
                        if df1.Coeff[i] > j:
                            df1.loc[i, 'Punteggio'] = 4
                        elif df1.Coeff[i] > z:
                            df1.loc[i, 'Punteggio'] = 3
                        elif df1.Coeff[i] < -z:
                            df1.loc[i, 'Punteggio'] = -3
                        elif df1.Coeff[i] < -j:
                            df1.loc[i, 'Punteggio'] = -4
                        else:
                            df1.drop(i, inplace=True)
                    new_words = dict([([a, b]) for a, b in zip(df1['Parola'], df1['Punteggio'])])
                    dfl = dfl.dropna(subset=['Text']).reset_index(drop=True)
                    #Vader mio
                    SIA = SentimentIntensityAnalyzer()
                    SIA.lexicon.update(new_words)
                    dfl['scores2'] = dfl['Text'].apply(lambda review: SIA.polarity_scores(review))
                    dfl['compound2']  = dfl['scores2'].apply(lambda score_dict: score_dict['compound'])
                    Fine = pd.DataFrame({'Score': dfl.Score,'Text': dfl.Text,'Vader': dfl.compound,'Vader_new': dfl.compound2})
                    Fine['Vader'] = dfl['compound'].apply(lambda x: 'Positive' if x > 0 else 'Negative' )
                    Fine['Vader_new'] = dfl['compound2'].apply(lambda x: 'Positive' if x > 0 else 'Negative' )
                    v=0
                    vn=0
                    for i in range(len(Fine)):
                        if Fine.Score[i]==Fine.Vader[i]:
                            v+=1
                        if Fine.Score[i]==Fine.Vader_new[i]:
                            vn+=1
                    diz2[z]=round(((vn/len(Fine))*100),2)
                    z+=0.1
                    i=0
                diz[j]=diz2
                diz2={}
                j+=0.1

            max_value = float('-inf')
            max_outer_key = None
            max_inner_key = None

            for outer_key, inner_dict in diz.items():
                if inner_dict:
                    inner_max_value = max(inner_dict.values())
                    if inner_max_value > max_value:
                        max_value = inner_max_value
                        max_outer_key = outer_key
                        max_inner_key = max(inner_dict, key=inner_dict.get)
        if self.score_function == 2:
            max_key = round(max_key, 1)
            p = print("Maximum accuracy value -> ", max_value, "\nOptimal division -> ", "+4:", max_key, " ", "-4:", -max_key)
            return max_value, max_key
        elif self.score_function == 3:
            max_outer_key = round(max_outer_key, 1)
            max_inner_key = round(max_inner_key, 1)
            p = print("Maximum accuracy value -> ", max_value, "\nOptimal division -> ", "+4:", max_outer_key, " ", "+3:", max_inner_key," ", "-4:", -max_outer_key, " ", "-3:", -max_inner_key)
            return max_value, max_outer_key, max_inner_key
                    





class seedot(object):
    """
    Apply seedot procedure
    """
    def __init__(self, data = None, tokens=2000, model = "logistic", score_function = 1, division = None, embedding = None):
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("A valid data must be specified.")
        # Check if 'Score' and 'Text' columns exist
        if 'Score' not in data.columns or 'Text' not in data.columns:
            raise ValueError("The 'data' DataFrame must have 'Score' and 'Text' columns.")
        if tokens <= 0:
            raise ValueError("tokens must be greater then 0.")
        # Check the values in the 'Score' column
        allowed_scores = ['Positive', 'Negative', 'Neutral']
        if not data['Score'].isin(allowed_scores).all():
            raise ValueError("The 'Score' column must only contain the values 'Positive', 'Negative', and 'Neutral'.")

        # Check the data types in the 'Text' column
        if data['Text'].dtypes != object:
            raise ValueError("The 'Text' column must only contain strings.")
        if division is not None and not isinstance(division, list):
            raise ValueError("The division parameter must be ether a list or None.")
        if isinstance(division, list) and len(division) not in [1, 2]:
            raise ValueError("len(division) must be ether 1 or 2.")
        if embedding is not None and embedding not in [1, 2]:
            raise ValueError("A valid embedding (None, 1, 2) must be specified.")
        self.data = data
        self.tokens = tokens
        self.model = model
        self.score_function = score_function
        self.division = division
        self.embedding = embedding
        ####

    def domain_dictionary(self):
        data = self.data[self.data['Score'] != 'Neutral']
        df_not_na = data[~(data['Text'].isna())]
        raw_text = df_not_na['Text']
        y = df_not_na['Score'].tolist()
        ####
        lemmatizer = WordNetLemmatizer()
        text = raw_text.str.lower().str.replace('[^\w\s\d]',' ', regex=True) # \d tiene anche i caratteri numerici
        text = text.str.split()
        text = text.apply(lambda x: [lemmatizer.lemmatize(sent) for sent in x])
        for row in text:
            [row.remove(i) for i in row if len(i) < 2]
        train_sentences = []
        for row in text:
            train_sentences.append(' '.join([item for item in row ]))
        ####
        #BOW
        vectorizer = CountVectorizer(ngram_range=(1, 1), max_features = self.tokens) #number of features changable
        X = vectorizer.fit_transform(train_sentences)
        feature_names = vectorizer.get_feature_names_out()
        X = X.toarray()
        X = np.array(X)
        y = np.array(y)

        ### Model
        if self.model == "logistic":
            #### Logisitic Model
            M = LogisticRegression(class_weight=None)
            M.fit(X,y)
            pred = M.predict(X)
        else:
            raise ValueError("For now the only implemented model is: model = 'logistic'.")
        coef=M.coef_
        ### Score
        if self.score_function == 1: #method 1
            if self.division is None:
                df1 = pd.DataFrame({"Coeff":coef[0],"Parola":feature_names,"Punteggio":0})
                df1.sort_values(by=["Coeff"], ascending = False, inplace = True)
                for i in range(len(df1)):
                    if df1.Coeff[i]>2: 
                        df1.Punteggio[i]=4
                    elif df1.Coeff[i]>1.5: 
                        df1.Punteggio[i]=3.5
                    elif df1.Coeff[i]>1: 
                        df1.Punteggio[i]=3
                    elif df1.Coeff[i]>.5: 
                        df1.Punteggio[i]=2.5
                    elif df1.Coeff[i]<-.5: 
                        df1.Punteggio[i]=-2.5
                    elif df1.Coeff[i]<-1: 
                        df1.Punteggio[i]=-3
                    elif df1.Coeff[i]<-1.5: 
                        df1.Punteggio[i]=-3.5
                    elif df1.Coeff[i]<-2: 
                        df1.Punteggio[i]=-4
                    else:
                        df1.drop(i, inplace=True)
            else:
                raise ValueError("If score_function = 1 it is required division = None.")
        elif self.score_function == 2: #method 2
            if isinstance(self.division, list) and len(self.division) == 1:
                df1 = pd.DataFrame({"Coeff": coef[0], "Parola": feature_names, "Punteggio": 0})
                df1.sort_values(by=["Coeff"], ascending=False, inplace=True)
                for i in reversed(range(len(df1))):
                    if df1.Coeff[i] > self.division[0]:  
                        df1.loc[i, 'Punteggio'] = 4
                    elif df1.Coeff[i] < -self.division[0]:  
                        df1.loc[i, 'Punteggio'] = -4
                    else:
                        df1.drop(i, inplace=True)
            else:
                raise ValueError("If score_function = 2 it is required division to be a 1 element list.")
        elif self.score_function == 3: #method 3
            if isinstance(self.division, list) and len(self.division) == 2:
                df1 = pd.DataFrame({"Coeff": coef[0], "Parola": feature_names, "Punteggio": 0})
                df1.sort_values(by=["Coeff"], ascending=False, inplace=True)
                for i in range(len(df1)):
                    if df1.Coeff[i] > self.division[0]:  # coefficiente maggiore di 0.6, punteggio 4
                        df1.loc[i, 'Punteggio'] = 4
                    elif df1.Coeff[i] > self.division[1]:  # coefficiente maggiore di 0.5, punteggio 3
                        df1.loc[i, 'Punteggio'] = 3
                    elif df1.Coeff[i] < -self.division[1]:  # coefficiente minore di -0.5, punteggio -3
                        df1.loc[i, 'Punteggio'] = -3
                    elif df1.Coeff[i] < -self.division[0]:  # coefficiente minore di -0.6, punteggio -4
                        df1.loc[i, 'Punteggio'] = -4
                    else:
                        df1.drop(i, inplace=True)
            else:
                raise ValueError("If score_function = 3 it is required division to be a 2 element list.")
        else:
            raise ValueError("A valid score_function (1, 2, 3) must be specified.")
        new_words =  dict([([a,b]) for a,b in zip(df1['Parola'], df1['Punteggio'])])
        ### embedding
        ft_model = FastText(sg=0, hs=1, sentences=train_sentences, vector_size=100, window=5, min_count=5, epochs=2, min_n=2, max_n=6)
        if self.embedding == 1:
            diz={}
            for i in list(new_words.keys()):
                for j in range(0,9):
                    if new_words[i]==4 or new_words[i]==-4:
                        if ft_model.wv.most_similar(i)[j][1]>0.99 and ft_model.wv.most_similar(i)[j][0] not in list(new_words.keys()): #Riferita a ogni parola del dizionario, si selezionata la più simile con punteggio maggiore di 0.99 e si inserisce 4
                            diz[ft_model.wv.most_similar(i)[j][0]]=new_words[i]
                        else:
                            break
            new_words.update(diz)
        elif self.embedding == 2:

            vad_path = "Vader.txt"
            _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
            lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), vad_path)
            vad = pd.read_csv(lexicon_full_filepath, delimiter="\t",names=["nome","valore"])
            d =  dict([([a,b]) for a,b in zip(vad['nome'], vad['valore'])])
            diz={}
            a=0
            for i in list(new_words.keys()):
                for j in range(0,9):
                    if new_words[i]==4 or new_words[i]==-4:
                        if ft_model.wv.most_similar(i)[j][1]>0.99 and ft_model.wv.most_similar(i)[j][0] not in list(new_words.keys()) :
                            if ft_model.wv.most_similar(i)[j][0] in list(d.keys()):
                                diz[ft_model.wv.most_similar(i)[j][0]]= (new_words[i] + float(d[ft_model.wv.most_similar(i)[j][0]]))/2
                            else:
                                diz[ft_model.wv.most_similar(i)[j][0]]=new_words[i]
                        else:
                            pass
            new_words.update(diz)
        else:
            pass #no embedding
        SentimentIntensityAnalyzer().lexicon.update(new_words)
        seedot_dictionary = SentimentIntensityAnalyzer().lexicon
        p = print("seedot_dictionary has been created.")
        return seedot_dictionary
