import os
import argparse
import pandas as pd
from tqdm import tqdm
from models.bert import BERT
from models.tfidf import TFIDF
from utils import read_config_file
from models.fasttext import FASTTEXT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('type', type=str, help="The type of architecture to use.", choices=['tfidf','fasttext', 'bert'])
parser.add_argument('model_file', type=str, help="The type of architecture to use.")
parser.add_argument('test_set', type=str,
                    help='Path to dataset csv file.')
parser.add_argument('--data_dir', type=str,
                    help='Path to dataset.', default="./datasets")
parser.add_argument('--output_dir', type=str,
                    help='Path to output directory.', default=".")
args = parser.parse_args()

config = read_config_file(args.config)

model = None
if args.type == "tfidf":
    model = TFIDF(config)
elif args.type == "fasttext":
    model = FASTTEXT(config)
elif args.type == "bert":
    model = BERT(config)
    

model.load(args.model_file)

test_samples = pd.read_csv(os.path.join(args.data_dir, args.test_set))

y_true, y_pred, y_pred_dist = [], [], []

for idx, row in tqdm(test_samples.iterrows(), total=test_samples.shape[0]):
    source = os.path.join(args.data_dir, "documents", row['source'])
    target = os.path.join(args.data_dir, "documents", row['target'])
    
    dist, cls = model.compare(source, target)
    
    y_true.append(row['class'])
    y_pred.append(cls)
    y_pred_dist.append(dist)

pd.DataFrame(
    data=list(zip(
        test_samples['source'].tolist(),
        test_samples['target'].tolist(),
        y_true, y_pred, y_pred_dist)),
    columns=['source', 'target', 'y_true', 'y_pred', 'y_pred_dist'],
).to_csv(
    os.path.join(
        args.output_dir,
        os.path.basename(args.config).replace(".json", ".eu.eval.csv")
    ),
    index=False
)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"Precision: {precision_score(y_true, y_pred)}")
print(f"Recall: {recall_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")