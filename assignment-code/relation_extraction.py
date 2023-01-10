
import os
import re
import pandas as pd
import nltk
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import WhitespaceTokenizer 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Spark NLP").master("local[4]").getOrCreate()

lemmatizer = WordNetLemmatizer()
tk = WhitespaceTokenizer()

nltk.download("stopwords")
stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_md')


#load the feature from the previous step
train_file_path = os.path.join(os.getcwd(), 'features_train.csv')
test_file_path = os.path.join(os.getcwd(), 'features_test.csv')
feature_train_path='features_train.csv'
feature_test_data='features_test.csv'
train_data = './data/semeval_train.txt'
test_data ='./data/semeval_test.txt'
ner_res_path = 'nlp_ner_orgtext_remove_a_with_label_pos_elables.parquet'
re_results_path = 're_res.csv'

class Feature:
    '''
    This class computes features in the input sentence that belong to entities(E1, E2) such as common word features, 
    relation direction, and information about the sentence’s dependency parse tree.  
    
    Common Word Features: Headwords of E1 and E2; Bag of words and bigrams in E1 and E2; 
    Words or bigrams around E1/E2; Bag of words or bigrams between the two entities; Entity Type; Entity POS tag.
    
    Relation direction Feature: This feature reveals the relation direction between E1 and E2.
    
    Dependency Parsing Feature: This feature reveals the shortest path between E1 and E2 in the dependency graph.
    '''

 
    def __init__(self, input_sentence, input_relation):

        self.input_sentence = input_sentence
        self.input_relation = input_relation
        self.e1, self.e2,self.e1_e2 = None, None,None
        self.head_e1,self.head_e2,self.head_e1_e2 = None, None, None
        
        self.before_e1, self.between_e1_e2, self.after_e2 =  None, None, None
        self.e1_ner, self.e2_ner,self.e1_postag, self.e2_postag = None, None, None, None
        self.e1_e2_ner= None
        self.relation, self.direction = None, None
        self.parse_list,self.shortest_path = None,None

    def __init__(self, input_sentence, input_relation,id):

        self.input_sentence = input_sentence
        self.input_relation = input_relation
        self.id = id
        self.e1, self.e2,self.e1_e2 = None, None,None
        self.head_e1,self.head_e2,self.head_e1_e2 = None, None, None
        
        self.before_e1, self.between_e1_e2, self.after_e2 =  None, None, None
        self.e1_ner, self.e2_ner,self.e1_postag, self.e2_postag = None, None, None, None
        self.e1_e2_ner= None
        self.relation, self.direction = None, None
        self.parse_list,self.shortest_path = None,None

        
class FeatureExtraction:
    '''
    This class is for extract features
    '''
    def __init__(self, input_sentence, input_relation):
        self.input_sentence = input_sentence
        self.input_relation = input_relation
        self.feature = Feature(input_sentence,input_relation, None)
    
    def extract_features(self):
        '''
        This founction is entrypoint of extract features
        '''
        self.pre_process()
        self.headwords()
        self.bow_and_bigrams_e1_e2()
        self.words_around_e1_e2()
        self.bow_and_bigrams_between_e1_e2()
        self.entity_type()
        self.entity_pos()
        self.relation_direction()
        self.dependency_parsing_path()
        return self.feature
    
    def pre_process(self):
        '''
            preprocessing the train dataset. We will remove the "\"", and find the entities in the sentence.
        '''
        '''
            get input_sentence 
            dataset example:
                "The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."  
        '''
        self.feature.input_sentence = re.findall('".*"', self.input_sentence)[0].strip("\"")
        self.feature.sentence_string = self.feature.input_sentence.replace('<e1>', ' ').replace('</e1>', ' ').replace('<e2>',' ').replace('</e2>', ' ')
        self.feature.e1 = re.findall('(?<=<e1>).*(?=</e1>)', self.feature.input_sentence)[0].strip()
        self.feature.e2 = re.findall('(?<=<e2>).*(?=</e2>)', self.feature.input_sentence)[0].strip()
        
    def headwords(self):
        '''
            Getting the headwords of entities and linking them
        '''
        self.feature.head_e1 = self.feature.e1.strip().split()[0]
        self.feature.head_e2 = self.feature.e2.strip().split()[0]
        self.feature.head_e1_e2 = self.feature.head_e1+ " " + self.feature.head_e2
    
    def bow_and_bigrams_e1_e2(self):
        '''
            Linking entities and in the pipeline, we will use it to build the bow and bigram features
        '''
        self.feature.e1_e2 = self.feature.e1+ " " + self.feature.e2
        
    
    def words_around_e1_e2(self):
        '''
            Linking words around entites and in the pipeline, we will use them to build the bow and bigram features
        '''
        self.feature.before_e1 = re.findall('.*(?=<e1>)', self.feature.input_sentence)[0].strip()
        self.feature.after_e2 = re.findall('(?<=</e2>).*', self.feature.input_sentence)[0].strip()
        

        
    def bow_and_bigrams_between_e1_e2(self):
        '''
            Linking words between entites and in the pipeline, we will use it to build the bow and bigram features
        '''
        self.feature.between_e1_e2 = re.findall('(?<=</e1>).*(?=<e2>)', self.feature.input_sentence)[0].strip()
        
    def entity_type(self):
        '''
            get the enities' entity type and link them
        '''
        doc = nlp(self.feature.sentence_string)
        nlp_res = []
        for ent in doc.ents:
            nlp_res.append([ent.text, ent.label_])
        for tag in nlp_res:
            if tag[0] == self.feature.e1:
                self.feature.e1_ner = tag[1]
            if tag[0] == self.feature.e2:
                self.feature.e2_ner = tag[1]
                
        if self.feature.e1_ner != None and self.feature.e2_ner != None: 
            self.feature.e1_e2_ner = self.feature.e1_ner + ' ' + self.feature.e2_ner
     
       
    def entity_pos(self):
        '''
            get the enity's pos tag
        '''
        doc = nlp(self.feature.sentence_string)
        for tok in doc:
            if self.feature.e1 in tok.text:
                self.feature.e1_postag = tok.pos_
            elif self.feature.e2 in tok.text:
                self.feature.e2_postag = tok.pos_
    
        

    def relation_direction(self):
        '''
            build the relation_direction features
        '''
        self.feature.input_relation = self.input_relation.replace('(', '**').replace(')', '')
        line_split = self.feature.input_relation.split('**')
        self.feature.relation = line_split[0].strip()
        if len(line_split) > 1:
            self.feature.direction = line_split[1].strip()


    def dependency_parsing_path(self):
        '''
            Get the shortest_path between thw entities.
        '''
        e1 = re.sub(r'[^\w\s]', '', self.feature.e1)
        e2 = re.sub(r'[^\w\s]', '', self.feature.e2)
        sent_string = self.feature.sentence_string.replace(self.feature.e1, e1).replace(self.feature.e2, e2)
        doc = nlp(sent_string)
    
        parse_list = []
        edges = []
        for token in doc:
            parse_list.append([token.text, token.tag_, token.head.text, token.dep_])
            
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),
                              '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
        entity1 = e1.strip().split()[0].lower()
        entity2 = e2.strip().split()[0].lower()

        self.feature.parse_list = parse_list
        self.feature.shortest_path = nx.shortest_path(graph, source=entity1, target=entity2)
        self.feature.shortest_path = ' '.join(self.feature.shortest_path)



class TextSelector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer takes in a key parameter during initialization, and it selects a specific column 
    from a dictionary or Pandas dataframe when transform is called.
    
    The transformer can be used as part of a pipeline to transform the Pandas dataframe before feeding it to a model.
    """
    
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer takes in a key parameter during initialization, and it selects a specific column 
    from the input data when transform is called.
    
    The transformer can be used as part of a pipeline to transform the data before feeding it to a model.
    """
    
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

      
class FeatureExtractionNERRes:
    """
    Redefine features accroding to entities extracted from our own dataset, and remove relation direction feature. 
    Redefine Dependency Parsing Feature, if we can not find the shortest path between labeled E1 and E2 
    in the dependency graph, we return empty.
    
    This class is similar to FeatureExtraction class. It is for processing our project's NER results.
    """
    def __init__(self, input_sentence,entities,id):
        self.input_sentence = input_sentence
        self.entities = entities
        self.feature = Feature(input_sentence,id= id,input_relation=None)
    
    def extract_features(self):
        self.pre_process()
        self.headwords()
        self.bow_and_bigrams_e1_e2()
        self.words_around_e1_e2()
        self.bow_and_bigrams_between_e1_e2()
        self.entity_type()
        self.dependency_parsing_path()
        return self.feature
    
    def pre_process(self):
    
        '''
            get input_sentence 
            dataset example:
                "The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."  
        '''
        self.feature.input_sentence =  self.input_sentence
        self.feature.sentence_string = self.feature.input_sentence.replace('<e1>', ' ').replace('</e1>', ' ').replace('<e2>',' ').replace('</e2>', ' ')
        self.feature.e1 = re.findall('(?<=<e1>).*(?=</e1>)', self.feature.input_sentence)[0].strip()
        self.feature.e2 = re.findall('(?<=<e2>).*(?=</e2>)', self.feature.input_sentence)[0].strip()
        
    def headwords(self):
        self.feature.head_e1 = self.feature.e1.strip().split()[0]
        self.feature.head_e2 = self.feature.e2.strip().split()[0]
        self.feature.head_e1_e2 = self.feature.head_e1+ " " + self.feature.head_e2
    
    def bow_and_bigrams_e1_e2(self):
        self.feature.e1_e2 = self.feature.e1+ " " + self.feature.e2
        
    
    def words_around_e1_e2(self):
        self.feature.before_e1 = re.findall('.*(?=<e1>)', self.feature.input_sentence)[0].strip()
        self.feature.after_e2 = re.findall('(?<=</e2>).*', self.feature.input_sentence)[0].strip()
        

        
    def bow_and_bigrams_between_e1_e2(self):
        self.feature.between_e1_e2 = re.findall('(?<=</e1>).*(?=<e2>)', self.feature.input_sentence)[0].strip()
        
    def entity_type(self):
        '''
        Find entities' pos tag and entity type
        '''
        for entity in self.entities:
            
            if entity.text.lower() == self.feature.e1.lower():
                self.feature.e1_ner = entity.label
                self.feature.e1_postag = entity['pos-tags']
            if entity.text.lower() == self.feature.e2.lower():
                self.feature.e2_ner = entity.label
                self.feature.e2_postag = entity['pos-tags']
        if self.feature.e1_ner != None and self.feature.e2_ner != None: 
            self.feature.e1_e2_ner = self.feature.e1_ner + ' ' + self.feature.e2_ner




    def dependency_parsing_path(self):
        '''
            Get the shortest_path between thw entities.
        '''
        e1 = re.sub(r'[^\w\s]', '', self.feature.e1)
        e2 = re.sub(r'[^\w\s]', '', self.feature.e2)
        sent_string = self.feature.sentence_string.replace(self.feature.e1, e1).replace(self.feature.e2, e2)
        doc = nlp(sent_string)
    
        parse_list = []
        edges = []
        for token in doc:
            parse_list.append([token.text, token.tag_, token.head.text, token.dep_])
            
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),
                              '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
        #if we can not find the shortest path between labeled E1 and E2 in the dependency graph, we return ''. 
        try:
            entity1 = e1.strip().split()[0].lower()
            entity2 = e2.strip().split()[0].lower()
            self.feature.shortest_path = nx.shortest_path(graph, source=entity1, target=entity2)
            self.feature.shortest_path = ' '.join(self.feature.shortest_path)
        except:
            self.feature.shortest_path = ''
        self.feature.parse_list = parse_list


def load_dataset(file_path):
    '''
    Load dataset, read sentences and relations, extract features and store them in a list, and then 
    convert the list into a Pandas dataframe used to train our SVM classifier.
    
    Inputs
    ----------
    A file contains sentences with annotated entitie(E1, E2) and relations.

    Returns
    ----------
    A Pandas dataframe contains features extracted from input contents.
        
    '''
    features = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                input_sentence = line
                input_relation = next(file)
                
                feature_extract = FeatureExtraction(input_sentence, input_relation)
                fature = vars(feature_extract.extract_features())
                features.append(fature)
                next(file)
                next(file)
    except StopIteration:
        pass
    
    return pd.DataFrame.from_records(features)

def load_ner_dataset(file_path):
    '''
    Load dataset, read sentences and relations, extract features and store them in a list, and then 
    convert the list into a Pandas dataframe used to train our SVM classifier.
    
    Inputs
    ----------
    A file contains sentences with annotated entitie(E1, E2).

    Returns
    ----------
    A Pandas dataframe contains features extracted from input contents.
    '''
    
    features = []
    try:
        ner_res = spark.read.parquet(file_path)
        count =0
        for labeled_text,entity,id in ner_res.select(col('labeled_text'),col('entity'),col('id')).distinct().collect():
            if entity == None or len(entity) < 2:
                continue
            if re.findall('(?<=</e2>).*', labeled_text) == None or len(re.findall('(?<=</e2>).*', labeled_text)) == 0:
                continue
            feature_extract = FeatureExtractionNERRes(labeled_text,entity,id)
            fature = vars(feature_extract.extract_features())
            features.append(fature)
                
    except StopIteration:
        pass
    
    return pd.DataFrame.from_records(features)


# ## Modeling
'''
a simple whitespace tokenizer, split a string into a list of words based on the spaces between them
'''
def whitespace_tokenizer(str_input):
    ret_words  = tk.tokenize(str_input) 
    return ret_words


def def_pipeline():
    """
    We define a scikit-learn pipeline for text classification that combines the output of multiple feature extractors, 
    and a classifier.

    We extract features from the text data and store them in a Pandas dataframe. 

    We apply a text selector to select a specific column of the input Pandas dataframe 
    and a TfidfVectorizer to convert the text into a numerical feature representation. 

    The final classifier takes the combined features as input and predicts a class label for each sample.
    """

    return Pipeline([
        ('features', FeatureUnion([
            ('e1', Pipeline([
                ('selector', TextSelector('e1')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('e2', Pipeline([
                ('selector', TextSelector('e2')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('e1_e2', Pipeline([
                ('selector', TextSelector('e1_e2')),
                ('tfidf', TfidfVectorizer(tokenizer=whitespace_tokenizer)),
            ])),
            ('e1_e2_bigram', Pipeline([
                ('selector', TextSelector('e1_e2')),
                ('tfidf', TfidfVectorizer(ngram_range = (2, 2))),
            ])),
            ('head_e1', Pipeline([
                ('selector', TextSelector('head_e1')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('head_e2', Pipeline([
                ('selector', TextSelector('head_e2')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('head_e1_e2', Pipeline([
                ('selector', TextSelector('head_e1_e2')),
                ('tfidf', TfidfVectorizer(tokenizer=whitespace_tokenizer)),
            ])),
            ('head_e1_e2_bigram', Pipeline([
                ('selector', TextSelector('head_e1_e2')),
                ('tfidf', TfidfVectorizer(ngram_range = (2, 2))),
            ])),
            ('e1_ner', Pipeline([
                ('selector', TextSelector('e1_ner')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('e2_ner', Pipeline([
                ('selector', TextSelector('e2_ner')),
                ('tfidf', TfidfVectorizer()),
            ])),
            
            ('e1_e2_ner', Pipeline([
                ('selector', TextSelector('e1_e2_ner')),
                ('tfidf', TfidfVectorizer()),
            ])),

            ('e1_postag', Pipeline([
                ('selector', TextSelector('e1_postag')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('e2_postag', Pipeline([
                ('selector', TextSelector('e2_postag')),
                ('tfidf', TfidfVectorizer()),
            ])),  

            ('before_e1', Pipeline([
                ('selector', TextSelector('before_e1')),
                ('tfidf', TfidfVectorizer(tokenizer=whitespace_tokenizer)),
            ])),    

            ('after_e2', Pipeline([
                ('selector', TextSelector('after_e2')),
                ('tfidf', TfidfVectorizer(tokenizer=whitespace_tokenizer)),
            ])),  

            ('between_e1_e2', Pipeline([
                ('selector', TextSelector('between_e1_e2')),
                ('tfidf', TfidfVectorizer(tokenizer=whitespace_tokenizer)),
            ])),    

            ('before_e1_bigram', Pipeline([
                ('selector', TextSelector('before_e1')),
                ('tfidf', TfidfVectorizer(ngram_range = (2, 2))),
            ])),    

            ('after_e2_bigram', Pipeline([
                ('selector', TextSelector('after_e2')),
                ('tfidf', TfidfVectorizer(ngram_range = (2, 2))),
            ])),  

            ('between_e1_e2_bigram', Pipeline([
                ('selector', TextSelector('between_e1_e2')),
                ('tfidf', TfidfVectorizer(ngram_range = (2, 2))),
            ])),  

            ('shortest_path', Pipeline([
                ('selector', TextSelector('shortest_path')),
                ('tfidf', TfidfVectorizer()),
            ]))
            ,  

            ('sentence_string', Pipeline([
                ('selector', TextSelector('sentence_string')),
                ('tfidf', TfidfVectorizer()),
            ]))

        ])),
        ('clf', LinearSVC(C=1000))
        ])

def process_training_test_dataset(train_data = './data/semeval_train.txt',test_data ='./data/semeval_test.txt',
                                  feature_train_path='features_train.csv',feature_test_data='features_test.csv'):
    #load training and test dataset and extract features
    features_training=load_dataset(train_data)
    features_test=load_dataset(test_data)

    #save features
    features_training.to_csv(feature_train_path)
    features_test.to_csv(feature_test_data)
    

def train_model(train_file_path,test_file_path):

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    train_df = pd.concat([train_df,test_df])

    #convert col type, otherwise it cannot be used to trian model
    col_to_change = ['input_sentence', 'input_relation', 'e1', 'e2', 'e1_e2',
        'head_e1', 'head_e2', 'head_e1_e2', 'before_e1', 'between_e1_e2',
        'after_e2', 'e1_ner', 'e2_ner',
        'e1_postag', 'e2_postag', 'e1_e2_ner', 'relation', 'direction', 'sentence_string']
    for col in col_to_change:
        train_df[col] = train_df[col].values.astype('U')


    train_df['Y'] = train_df["relation"] + ' ' + train_df["direction"]

    X = train_df
    Y = train_df.loc[:, 'Y']
    #split training dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=111)
    classifier = def_pipeline()
    # ## Training

    # training model
    classifier.fit(X_train, y_train)
    return classifier

# ## Tuning
# # defining parameter range
# param_grid = {'clf__C': [0.1, 1, 10, 100, 1000,10000]}

# #use GridSearchCV to do hyperparameter tuning
# clf = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
# clf.fit(X_train, y_train)
# print("Best Score: ", clf.best_score_)
# print("Best Params: ", clf.best_params_)

# from sklearn.metrics import classification_report, confusion_matrix
# grid_predictions = clf.predict(X_test)
  
# # print classification report
# print(classification_report(y_test, grid_predictions))


# ## NER Results Process
def process_ner_res(classifier ,ner_res_path='nlp_ner_orgtext_remove_a_with_label_pos_elables.parquet',results_path='re_res.csv'):

    # evaluate on our own dataset after recognizing and labeling entities E1 and E2 in text
    ner_res = load_ner_dataset(ner_res_path)
    
    #predict besed on the ner results
    preds = classifier.predict(ner_res)
    #concatenate ner_res with preds
    ner_res['preds'] = preds

    #load relation-wikidata dictionary
    relation_link = pd.read_csv('relation_linking_dictionary.csv', names=['relation','relation_name','link'], header=None)

    #link relation and wikidata link
    df_left = pd.merge(ner_res, relation_link, how='left', left_on='preds', right_on='relation')
    df_left.to_csv(results_path)



def process_relation_extraction():
    '''
    This is the entry point of the whole RE task 
    '''
    # process training and testing dataset
    process_training_test_dataset(train_data,test_data,feature_train_path,feature_test_data)
    #Training Model
    classifier = train_model(train_file_path,test_file_path)
    #Process ner results and do relation extraction
    process_ner_res(classifier ,ner_res_path=ner_res_path,results_path=re_results_path)