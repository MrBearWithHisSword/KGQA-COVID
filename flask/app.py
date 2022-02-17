import json
import pandas as pd
import torch
import torch.nn.functional as F
from collections import OrderedDict
import re
from flask import Flask, render_template, request, url_for, session, config
from py2neo import Graph, Node, Relationship
from pyopenie import OpenIE5
from transformers import pipeline

# Initialize Information Extractors
extractor = OpenIE5('http://localhost:8000')
ner = pipeline('ner',  grouped_entities=True, model='dbmdz/electra-large-discriminator-finetuned-conll03-english')
emb = pipeline('feature-extraction', model='sentence-transformers/bert-base-nli-mean-tokens', return_tensors='pt')
# Connect to Neo4j Graph Database
# Local
neo4j_url = "neo4j://localhost:7687"
neo4j_user = "neo4j"
neo4j_passwd = "hcshi"
graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_passwd))
# graph = Graph("neo4j://localhost:7687", auth=("neo4j", "hcshi"))
# AuraDB
# graph = Graph("neo4j+s://6a44177f.databases.neo4j.io", auth=("neo4j", "tko59wPz4-UWe9Obd8qDEI1lU68LzWsTYIvS4iFzAN0"))

# Load Nodes' Name for KBQA
# Since we conduct a simle entity liking sololy based on the entities' names, we can retrive the entities' names in advance for simplicity.
# # Save node names and relation types
# nodes_df = pd.read_csv('../data/merged/sampled_nodes.csv', low_memory=False)
# node_names = list(set(nodes_df.iloc[:,2]))
# with open('names.json', 'w') as f:
#   json.dump(node_names, f, indent=4)

pattern = 'biolink:(.*)$'
prog = re.compile(pattern)
extract_type = lambda s: prog.findall(s)[0]
# edges_df = pd.read_csv('../data/merged/sampled_edges.csv', low_memory=False)
# edge_types = list(set(edges_df.iloc[:,2]))
# relation_types = list(map(extract_type, edge_types))
# with open('relations.json', 'w') as f:
#   json.dump(relation_types, f, indent=4)

# Load node names and relation types
with open('names.json', 'r') as f:
  node_names = json.load(f)

with open('relations.json', 'r') as f:
  relation_types = json.load(f)

node_name2id = OrderedDict({k:v for v, k in enumerate(node_names)})
relation_type2id = OrderedDict({k.upper():v for v, k in enumerate(relation_types)})

# Load embeddings of nodes and relations for a simple entity linking.
def fetct_embedding(sent):
  return torch.mean( torch.tensor(emb(sent))[0,1:-1], dim=0, keepdim=True)
  # return torch.tensor(emb(sent))[0,1:-1]

# relation_embs = torch.concat([fetct_embedding(rel) for rel in relation_types], dim=0)
# entity_embs = torch.concat([fetct_embedding(str(entity_name)) for entity_name in node_names], dim=0)
# torch.save(entity_embs, 'entity_embs.pt')
# torch.save(relation_embs, 'relation_embs.pt')
relation_embs = torch.load('relation_embs.pt')
entity_embs = torch.load('entity_embs.pt')


def question2query(question):
  # A simple information extraction based KBQA method.
  # Extract Information
  # NER
  entities = ner(question)
  print(entities)
  if len(entities) == 0:
    return ""
  elif len(entities) == 1:
    entity0 = entities[0]
    mention_str0, confidence0 = question[entity0['start']:entity0['end']], entity0['score']
  else:
    entity0, entity1 = entities[0], entities[1]
    mention_str0, confidence0 = question[entity0['start']:entity0['end']], entity0['score']
    mention_str1, confidence1 = question[entity1['start']:entity1['end']], entity1['score']
  
  # Extract quired relation.
  # Simply use the first extraction to get the quired relation. (We assume there is only one 1-order relationship in the query)
  extractions = extractor.extract(question)
  print(extractions)
  extract_relation = extractions[0]['extraction']['rel']['text']

  # Convert the question to Cypher query. (Since we dont have the training data to train a NN to do this, we simply define some rules for demontration)
  # Simple Entity Linking with Cosine Similarity
  mention_emb0 = fetct_embedding(mention_str0)
  mention_entity_sim = F.cosine_similarity(mention_emb0, entity_embs)
  linked_entity_id = torch.argmax(mention_entity_sim)
  linked_entity_name = node_names[linked_entity_id]
  # Compute the relation for query
  extract_relation_emb = fetct_embedding(extract_relation)
  exist_relations = graph.query(cypher="MATCH(n {name:'%s'})-[rel]-(m) RETURN COLLECT(DISTINCT rel.predicate);" % (linked_entity_name)).to_table()
  exist_relation_ids = [relation_type2id[extract_type(term).upper()] for term in exist_relations[0][0]]
  relation_mask = torch.zeros(len(relation_embs))
  relation_mask[torch.tensor(exist_relation_ids).long()] = 1.0
  relation_sim = F.cosine_similarity(extract_relation_emb, relation_embs)
  masked_sim = relation_sim * relation_mask
  # exist_relation_ids = [relation_type2id[extract_type(tb[0][0]['predicate']).upper()] for tb in exist_relations]
  query_relation_id = torch.argmax(masked_sim)
  query_relation_type = relation_types[query_relation_id]
  # Construct Cypher Query
  match_part = "MATCH((n {name: '%s'})-[rel: %s]-(m))" % (linked_entity_name, query_relation_type.upper())
  if 'how many' in question.lower():
    return_part = " RETURN COUNT(m)"
    question_type = 0
  else:
    return_part = " RETURN COLLECT(DISTINCT m.name)"
    question_type = 1
  
  query = match_part + return_part
  ref_query = match_part[:5] + " p=" + match_part[5:] + " RETURN p LIMIT 100"
  # Merge top-2 extractions:
  # query = 'MATCH (n {name:"COVID-19"})-[rel]-(m) RETURN (n)-[rel]-(m) LIMIT 1;'
  return query, ref_query, question_type, entity0['start'], entity0['end'], extract_relation, linked_entity_name, query_relation_type

app = Flask(__name__)
@app.route("/", methods=['post', 'get'])
def search():
  # return render_template('QA.html')
  content = request.form.get("content")
  if content is None:
    content = ""
    return render_template('KGQA.html', question=content, query="", res="", ref_query="MATCH p=((n {name: 'COVID-19'})-[rel*1..3]-(m)) RETURN p LIMIT 1000", neo4j_url=neo4j_url, neo4j_user=neo4j_user, neo4j_passwd=neo4j_passwd)
  else:
    # content = "How many disease are related to COVID-19?"
    query, ref_query, question_type, et_start, et_end,  extract_relation, linked_entity_name, query_relation_type  = question2query(question=content)
    # query =  'MATCH(n {name:"COVID-19"})-[rel]-(m) RETURN (n)-[rel]-(m) LIMIT 1;'
    res = graph.query(cypher=query).to_table()
    if question_type == 0:
      transformed_res = res[0][0]
    else:
      transformed_res = ','.join(res[0][0])
    return render_template('KGQA.html', question=content, query=query, res=transformed_res, ref_query=ref_query, neo4j_url=neo4j_url, neo4j_user=neo4j_user, neo4j_passwd=neo4j_passwd, 
    extracted_entity=content[et_start:et_end],  extract_relation=extract_relation, linked_entity_name=linked_entity_name, query_relation_type= query_relation_type)
  
if __name__ == "__main__":

 
  app.run('0.0.0.0', port=8888)