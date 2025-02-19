import os
import argparse
import pandas as pd
from models.bert import BERT
from models.tfidf import TFIDF
from utils import read_config_file
from models.fasttext import FASTTEXT
# from models.bert_keras import BERTKERAS

parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('type', type=str, help="The type of architecture to use.", choices=[
                    'tfidf', 'fasttext', 'bert', 'bert_keras'])
parser.add_argument('model_file', type=str,
                    help='Path to the model file')
parser.add_argument('source', type=str,
                    help='Path to the source document', default=".")
parser.add_argument('target', type=str,
                    help='Path to the target document', default=".")
args = parser.parse_args()

config = read_config_file(args.config)

model = None
if args.type == "tfidf":
    model = TFIDF(config)
elif args.type == "fasttext":
    model = FASTTEXT(config)
elif args.type == "bert":
    model = BERT(config)
# elif args.type == "bert_keras":
#     model = BERTKERAS(config)
    
model.load(args.model_file)

similarity = model.compare(args.source, args.target)

print(similarity)
