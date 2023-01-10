#!/usr/bin/env python
# coding: utf-8

# ! pip install -q pyspark==3.1.2 spark-nlp
# get_ipython().system(' pip install -q pyspark==3.1.2 spark-nlp')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import en_core_web_md
import nltk
from nltk.corpus import stopwords #stop wors
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import gzip
import re
import sparknlp
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import *

spark = SparkSession \
    .builder \
    .appName("spark wdps") \
    .config("spark.executor.allowSparkContext", True) \
    .getOrCreate()
sc = spark.sparkContext
nltk.download('wordnet')

nltk.download('stopwords')
nltk.download('punkt')
NER_type = ["DATE","TIME","CARDINAL","ORDINAL","QUANTITY","PERCENT","MONEY"] # ruled type list ,we ignore some entities
nlp = en_core_web_md.load()

# INPUT='./data/warcs/sample.warc.gz'
KEYNAME = "WARC-TREC-ID"

def get_pos(text): 
    '''
     get the pos tagging for each token
     input a text and we get the pos tag of each token
    '''
    doc = nlp(text) # ner algorithm
    pos_map={}
    for token in doc:
        pos_map[str(token)]=token.pos_
    return pos_map
    

def new_ner(text):
    '''
    Name Entity Recognition
    Input is a paragraph of text and we return a list of (text,label and pos)
    '''
    # for each entity we are interested in the text,label and pos
    doc = nlp(text)
    pos_map={}
    for token in doc:
        pos_map[str(token)]=token.pos_
    res=[]
    entitys = [X for X in doc.ents if X.label_ not in NER_type]
    for X in entitys:
        if len(X.text)==1: # only one word, then do not split it
            if X.text in pos_map:
                res.append((X.text,X.label_,pos_map[X.text]))
            else:
                res.append((X.text,X.label_,' ')) # no valid pos
        else:
            pos=''
            strs=X.text.split(' ') # if it contains more than one token than concat them with '-'
            for s in strs:
                pos+=pos_map[s] if s in pos_map else ' '
                pos+='-'
            pos=pos[0:len(pos)-1]
            res.append((X.text,X.label_,pos))
    return res
            

def split_records(stream): #provided by professor
    payload = ''
    for line in stream:
        if line.strip() == "WARC/1.0":
            yield payload
            payload = ''
        else:
            payload += line
    yield payload

def process_html(payload): # provided by professor
    html_flag =False
    html_content=''
    for line in payload.splitlines():
        if line.startswith("<html"):
            html_flag =True
        if html_flag :
            html_content += line
    if html_flag==False:
        return 'no_html'
    return  html_content

def remove_upprintable_chars(s):  #remove un printable characters
    return ''.join(x for x in s if x.isprintable())


def clean_non_english(txt): # remove non-english characters
    txt = re.sub(r'\W+', ' ', txt)
    txt = txt.lower()
    txt = txt.replace("[^a-zA-Z]", " ")
    word_tokens = word_tokenize(txt)
    filtered_word = [w for w in word_tokens if all(ord(c) < 128 for c in w)]
    filtered_word = [w + " " for w in filtered_word]
    return "".join(filtered_word)

def preprocess_single(text): 
        '''
        remove non-English words
        remove punctuactions and stopwords, as well as some special characters
        tokenize the text and remove unprintable chars
        '''
        text=clean_non_english(text)
        text = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。§?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', " ", text)
        text = re.sub('[\001\002\003\004\005\006\007\n\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x80\x90\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+', '', text)
        tokens= word_tokenize(text)
        token_list =[remove_upprintable_chars(word) for word in tokens]
        
        filtered=[w for w in token_list if(w.casefold() not in stopwords.words('english'))] 
        #return token_list
        return ' '.join(filtered)
    #return stemmed_tokens


def html_to_text(html_content): #parse html to text with Beautifulsoup
    soup = BeautifulSoup(html_content, "html.parser")
    soup.prettify()
    [s.extract() for s in soup(['iframe', 'script', 'style','[document]','noscript','header','meta','head', 'input','a'])] #Unuseful labels
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    stripped_text = re.sub('[\001\002\003\004\005\006\007\n\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x80\x90\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+', '', stripped_text)
    return stripped_text


def cut_sentences(content): 
    '''
    For the convenience of relation extraction, we cut text into sentences
    '''
    end_flag = ['?', '!', '.', '？', '！', '。', '…']
    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        # concat characters
        tmp_char += char
        # if already at the last position
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break		
        # if this char is in end flag
        if char in end_flag:
            # then judge the next one
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''			
    return sentences


# what if I RETURN A LIST 
def find_entities1(payload):
    '''
     RETURN A LIST with each item
     containing original key ,one sentence and the cleaned text
    '''
    res_list=[]
    if payload == '':
        return
    # The variable payload contains the source code of a webpage and some
    # additional meta-data.  We first retrieve the ID of the webpage, which is
    # indicated in a line that starts with KEYNAME.  The ID is contained in the
    # variable 'key'
    key = None
    for line in payload.splitlines():
        if line.startswith(KEYNAME):
            key = line.split(': ')[1]
            break
    if key is None:
        return 
    html_content = process_html(payload)
    if html_content=='no_html':
        return 
    texts=html_to_text(html_content)
    texts=' '.join(texts.split()) #remove extra spaces
    sentences=cut_sentences(texts) # cut to sentences
    for sent in sentences:
        res_list.append((key,sent,preprocess_single(sent)))
    
    yield res_list 


def ner(text):# only keep the text of entity
    NER_type = ["DATE","TIME","CARDINAL","ORDINAL","QUANTITY","PERCENT","MONEY"]
    doc = nlp(text)
    entity = [(X.text, X.label_) for X in doc.ents if X.label_ not in NER_type]
    entity = [X.text for X in doc.ents if X.label_ not in NER_type]
    entity = list(set(entity))
    return entity

def add_elabels(org_text):
    '''
    We add <e1></e1> and <e2></e2> labels to original text, for the convenience of relation extraction
    '''
    tmp_text =org_text+""
    tmp_text =tmp_text.lower()
   
    entities=ner(preprocess_single(org_text))
    start=0
    pos_map={}
    count=0 # How many valid entities we have find
    for i in range(len(entities)):
        if count==2: # a crude solution: for each sentence we only find the first two entities and figure out their relation
            break
        ent=entities[i]
        
        #if org_text.find(ent,start)==-1:# this entity is not in original text
        if tmp_text.find(ent,start)==-1:
            continue
        #pos_map[ent]=org_text.find(ent,start)
        pos_map[ent]=tmp_text.find(ent,start)
        count+=1
        start=pos_map[ent]+len(ent)
    if count==0: # no entity in orginal text
        return org_text
    start1=pos_map[list(pos_map.keys())[0]] # the start idx of the first entity
    end1=start1+len(list(pos_map.keys())[0]) # the end idx of the first entity
    res=""
    res+=org_text[0:start1] # string processing
    res+="<e1>"
    res+=org_text[start1:end1]
    res+="</e1>" 
    if count==1: # only one entity
        res+=org_text[end1:]
        return res
    if count==2: # two entities
        start2=pos_map[list(pos_map.keys())[1]]
        end2=start2+len(list(pos_map.keys())[1])
        res+=org_text[end1:start2] # string btw two entities
        res+="<e2>"
        res+=org_text[start2:end2]
        res+="</e2>"
        res+=org_text[end2:]
    return res
        
    
'''
def add_elabels(org_text):
    tmp_text=""+org_text
    tmp_text=tmp_text.lower()
    entities=ner(preprocess_single(org_text))
    start=0
    pos_map={}
    count=0 # How many valid entities we have find
    for i in range(len(entities)):
        if count==2: # a crude solution: for each sentence we only find the first two entities and figure out their relation
            break
        ent=entities[i]
        if tmp_text.find(ent,start)==-1:# this entity is not in original text
            continue
        pos_map[ent]=tmp_text.find(ent,start)
        count+=1
        start=pos_map[ent]+len(ent)
    if count==0: # no entity in orginal text
        return org_text
    start1=pos_map[list(pos_map.keys())[0]] # the start idx of the first entity
    end1=start1+len(list(pos_map.keys())[0]) # the end idx of the first entity
    res=""
    res+=org_text[0:start1] # string processing
    res+="<e1>"
    res+=list(pos_map.keys())[0]
    res+="</e1>" 
    if count==1: # only one entity
        res+=org_text[end1:]
        return res
    if count==2: # two entities
        start2=pos_map[list(pos_map.keys())[1]]
        end2=start2+len(list(pos_map.keys())[1])
        res+=org_text[end1:start2] # string btw two entities
        res+="<e2>"
        res+=list(pos_map.keys())[1]
        res+="</e2>"
        res+=org_text[end2:]
    return res
'''
def preprocessing(INPUT='./data/warcs/sample.warc.gz'):     
    #count =0
    entity_list=[]

    with gzip.open(INPUT, 'rt', errors='ignore') as fo:
        for record in split_records(fo):
                #count+=1
                entities = find_entities1(record)
                for ent in entities:
                    for key, org_text,text in ent:
                        entity_list.append((key,org_text,text))
    # count
    rdd =sc.parallelize(entity_list)
    deptColumns = ["id","org_text","text"]
    # convert to dataframe since sparknlp apply the model on dataframe
    df = rdd.toDF(deptColumns)
    schema = ArrayType(StructType([
        StructField("text", StringType(), False),
        StructField("label", StringType(), False),
        StructField("pos-tags", StringType(), False)
        
    ]))

    ner_f = udf(new_ner, schema)
    spark.udf.register(name="new_ner", f=ner_f) #user designed function, coduct NER
    df_withEntity = df.withColumn("entity",ner_f('text'))
    add_elabels_f=udf(add_elabels,StringType())
    spark.udf.register(name="add_elabels", f=add_elabels_f)

    df_withlables=df_withEntity.withColumn("labeled_text",add_elabels_f('org_text'))
    df_withlables.write.mode("overwrite").format("parquet").save("nlp_ner_orgtext_remove_a_with_label_pos_elables.parquet")


