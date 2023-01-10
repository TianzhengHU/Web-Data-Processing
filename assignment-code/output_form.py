from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import lit,col
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import sparknlp
import pandas as pd

spark = sparknlp.start()

''' 
This function is used to load the entities and their links from parquet file
'''
def get_linking(org_id,entity):
    query="select entity,linking, id from df_ent_lk where id= "+"'"+str(org_id)+"'"+" and entity="+"'"+ entity.lower()+"'"
    #print(query)
    return spark.sql(query).collect()
    
'''
This function restructure the entitis , relations and their links into more readalbe format
'''

def predict_format():
    '''
    based on  the result of relation extraction, entity linking and relation linking
    We write the output into a file
    '''
    filename = 'write_data.txt' #the output file
    f=open(filename,'a')#append
    df_ent_lk=spark.read.parquet('entity_linking_res.parquet')
    df_ent_lk.createOrReplaceTempView("df_ent_lk")#create a temporary view for sql query
    

    df_relation_lk=pd.read_csv('re_res.csv')

    for i in range(len(df_relation_lk)):
        
        org_id =df_relation_lk['id'][i] # the id in original text
       
        ent1=df_relation_lk['e1'][i]# entity 1
        ent2=df_relation_lk['e2'][i]# entity 2
        relation=df_relation_lk['relation_name'][i] # the relation name
        link=df_relation_lk['link'][i] #relation linking
        query_e1="select entity,linking, id from df_ent_lk where id= "+"'"+str(org_id)+"'"+" and entity="+"'"+ ent1.lower()+"'" #url of entity 1
        query_e2="select entity,linking, id from df_ent_lk where id= "+"'"+str(org_id)+"'"+" and entity="+"'"+ ent2.lower()+"'" #url of entity 2
        if  spark.sql(query_e1).collect() is None or len(spark.sql(query_e1).collect())==0 :
            continue
        else:
            
            e1_lk=spark.sql(query_e1).collect()[0].linking
        if  spark.sql(query_e2).collect() is None or len(spark.sql(query_e2).collect())==0:
            continue
        else:
            
            e2_lk=spark.sql(query_e2).collect()[0].linking
        f.write("ENTITY: "+str(org_id)+'\t'+str(ent1)+'\t'+str(e1_lk)+'\n')
        f.write("ENTITY: "+str(org_id)+'\t'+str(ent2)+'\t'+str(e2_lk)+'\n')
        f.write("RELATION: "+str(org_id)+'\t'+str(e1_lk)+'\t'+str(e2_lk)+'\t'+str(relation)+'\t'+str(link)+'\n') #Write to the file
        
    
        
        
