from pyspark.sql import SparkSession
import json
from elasticsearch import Elasticsearch
import re
import textdistance
from pyspark.sql.types import *
# import sparknlp
from pyspark.sql import SparkSession

#Start a spark instace
# spark  = sparknlp.start() # start spark

spark = SparkSession \
    .builder \
    .appName("spark wdps") \
    .config("spark.executor.allowSparkContext", True) \
    .getOrCreate()
sc = spark.sparkContext
path='nlp_ner_orgtext_remove_a_with_label_pos_elables.parquet'


def escape_elasticsearch_query(query):
    '''
    remove unuseful characters,query:the input query string
    output: query removed specific characters

    '''
    return re.sub(r'[^\w\s]', '', query) #remove unuseful characters

#define an entity candidate class  
class candidate:
    id = '' # id got from original file
    label = '' #pos label
    score = 0.0
    url = '' # the Wikipedia url of candidate
    description='' #the Wikipedia description of candidate
    def __init__(self):
        pass
    def __init__(self, id, label,url,description): # constuctor with parameters
        self.id = id
        self.label = label 
        self.url = url 
        self.description = description
    def __repr__(self):
        return f"<candidate label:{self.label} score:{self.score} url:{self.url} description:{self.description}>"

    def __str__(self): # print
        return f"From str method of candidate: label is {self.label}, score is {self.score}, url is {self.url}, description is {self.description}"

def getCandidates(entity):  
    '''
    input : entity is a string, the text of entity mention
    return: the generated entity candidates
    ''' 
    return search(entity)
 


def cosine_distance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return textdistance.cosine(u.split(),v.split())

def rankCandidates(candidates,entity,sent_text):
    '''
    Conduct candidate ranking
    parameters: candidates is a list containing all candidates entity of a mention,entity is the txt of a given mention
    We calculate the cosine distance between the mention text and the title of each candidate entity
    and sort by the result 
    '''
    if candidates is None:  
        return []
    for candidate in candidates:
        candi_str = candidate.label
        
        score = textdistance.cosine(entity,candi_str) #This is actually the title of a candidate entity
        candidate.score = score
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True) # sort the cosine distance

def getLinking(x):
    '''
    This is the function we used in rdd map
    We generate candidate entity generation, rank them and get the entity with the top score
    We return a list with each item containing the id in original text, the text of entity mention and the linking in wikipedia
    '''
    entities = x.entity # we use the text of entity
    sent_text = x.org_text # the original text
    if entities == None: #Sometimes no entity has been recognized
        return

    res_list = []
    for entity in entities:
        candidates = getCandidates(entity.text) #Candidate entity generation
        candidateList = rankCandidates(candidates,entity.text,sent_text) #Candidate ranking
        candidate = candidateList[0] if candidateList else None #After sort we get the entity with the highest score
        linking = candidate.url if candidate is not None and candidate.score >= 0 else None # get wikipedia url
        
        res_list.append((x.id,entity.text,linking))

    return res_list


def search(query): 
    '''
    We built a local es 
    bascially the input is a string which represents the entity mention and we return a list of possible candidates
    we find in wikipedia which may match the input entity mention
    '''
    
    e = Elasticsearch(size="50", timeout=30, max_retries=10, retry_on_timeout=True) #search es
    p = { "query" : { "query_string" : { "query" : escape_elasticsearch_query(query)}}}
    candidates = []
    try:
        response = e.search(index="enwiki", body=json.dumps(p)) #Query the local es
        if response: # we may get more than one items
            for hit in response['hits']['hits']:
                if 'title' not in hit['_source'] and 'text' not in hit['_source']:
                    continue
                label = hit['_source']['title'] if 'title' in hit['_source'] else hit['_source']['text']
                id = hit.get('id')
                temp = candidate(id,label,url = hit['_source'].get('url'),description = hit['_source'].get('text'))

                candidates.append(temp)
    except Exception as e:
        print("Exception:",e,p)
    return candidates

'''
 This is the main fuction that used to run the entity linking pipeline
 res is the output of the whole pipeline 
 save the res to parquet file
''' 
def entity_linking():
    df = spark.read.parquet(path) #open the NER results 
    res = df.rdd.map(getLinking).filter(lambda x: len(x) > 0).flatMap(lambda x:x)
    res.toDF(['id','entity','linking']).write.partitionBy("id").mode("overwrite").parquet("entity_linking_res.parquet")

