import os
import argparse
import pandas as pd
from models.bert import BERT
from models.tfidf import TFIDF
from utils import read_config_file
from models.fasttext import FASTTEXT

parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('type', type=str, help="The type of architecture to use.", choices=['tfidf','fasttext', 'bert'])
parser.add_argument('dataset_path', type=str,
                    help='Path to dataset csv file')
parser.add_argument('--data_dir', type=str,
                    help='Path to dataset', default="./datasets")
parser.add_argument('--output_dir', type=str,
                    help='Path to output directory.', default="./pretrained")
parser.add_argument('--model_ext', type=str,
                    help='Path to output directory.', default="pkl")
args = parser.parse_args()

config = read_config_file(args.config)
dataset_path = os.path.join(args.data_dir, args.dataset_path)
datasets = pd.read_csv(dataset_path)

datasets["source"] = args.data_dir + "/documents/" + datasets["source"]
datasets["target"] = args.data_dir + "/documents/" + datasets["target"]

model = None
if args.type == "tfidf":
    model = TFIDF(config)
elif args.type == "fasttext":
    model = FASTTEXT(config)
elif args.type == "bert":
    model = BERT(config)

model.train(datasets)

output_path = os.path.join(args.output_dir, os.path.basename(args.config).replace(".json", f".{args.model_ext}"))
model.save(output_path=output_path)