# WDPS Group 22

## Project Structure
- preprocessing.py 
  - Process web data and do NER. Results will be saved into nlp_ner_orgtext_remove_a_with_label_pos_elables.parquet
- entity_linking.py
  - Based on the NER result, it will do entity linking from elastic search. The results will be stored in entity_linking_res.parquet
- relation_extraction.py 
  - Based on the NER result, it will perform relation extraction. The training dataset is semeval_train.txt under the data folder. the results are in the re_res.csv
- output_form.py 
  - It will output the predictions based on the results of entity_linking.py and relation_extraction.py
## Setup Environment

There are two ways which can set up the environment. 

### Docker

You can run the following command to run the Docker container.
```

docker run -d \
        --name elasticsearch \
    -e "discovery.type=single-node" \
    -v <The parent folder path of elastic search index data>:/app/wdps/ \
    -v <wdps code folder>:/app/assignment \
    --privileged \
    -p 9200:9200 \
    -p 9300:9300 \
--interactive --tty wangjycode/elasticsearch:v3
```

### Setup locally

1. install python, java
2. run setup.sh to install python dependencies
3. install elastic search locally (The wikidata index data you can find [here](!https://drive.google.com/file/d/17bwDzWIuZGcCcNgji1jIZb253_tKDox2/view?usp=sharing)) or you can run 
```
docker run -d \
        --name elasticsearch \
    -e "discovery.type=single-node" \
    -v <The parent folder path of elastic search index data>:/app/wdps/\
    --privileged \
    -p 9200:9200 \
    -p 9300:9300 \
--interactive --tty wangjycode/elasticsearch:v1
```


***1. Note: Before running the whole pipeline, you should have an elastic search cluster loaded with Wikipedia***
***2. Note: the ES data path is '/app/wdps/data', so please put them in the right position***


### Run code

You can run each python file to do different tasks. You can also run ```starter_code.py``` to run the whole pipeline.

**Example:**
```
python starter_code.py data/warcs/sample.warc.gz
```