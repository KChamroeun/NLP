import json
import numpy as np
from tqdm import tqdm
from typing import List
from gensim import utils
from utils import preprocess
import matplotlib.pyplot as plt
from gensim.utils import tokenize
from scipy.spatial import distance
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import FastText
from gensim.models.callbacks import CallbackAny2Vec




class Iter:
    def __init__(self, items: List[str]) -> None:
        self.items = items
        
    def __iter__(self):
        for item in self.items:
            with open(item, "r") as sample_file:
                words = preprocess(sample_file.read())
                yield words

class FASTTEXT():
    def __init__(self, config) -> None:
        self.model = FastText(
            sg={
                "cbow": 0,
                "skipgram": 1,
            }[config["model"]],
            vector_size=config["vector_size"],
            window=config["window"],
            epochs=config["epochs"],
            alpha=config["alpha"],
            min_n=config["min_n"],
            max_n=config["max_n"],
            workers=config.get("workers", 4)
        )
        self.decision_threshold = config['decision_threshold']
        self.similarity_measure = config["similarity_measure"]
        self.log_file = config["log_file"]
        self.patience = config["patience"]


    def train(self, datasets: str):
        datasets = datasets[datasets['class'] == 1]['source']
        self.model.build_vocab(
            corpus_iterable=Iter(datasets.tolist())
        )

        total_examples = self.model.corpus_count

        for epoch in tqdm(range(self.model.epochs), desc="Training epochs"):
            self.model.train(
                Iter(datasets.tolist()),
                epochs=1,  
                total_examples=total_examples
            )
            
        
    def save(self, output_path: str):
        self.model.save(output_path)
        
    def compare(self, source, target):
        source = self.__doc_to_vec(source)
        target = self.__doc_to_vec(target)
        if self.similarity_measure == "cosine_similarity":
            dist = distance.cosine(source, target)
        elif self.similarity_measure == "euclidean_distance":
            dist = distance.euclidean(source, target)
        return dist, int(dist < self.decision_threshold)
        
    def __doc_to_vec(self, doc):
        with open(doc, "r") as doc_file:
            words = preprocess(doc_file.read())
            words = [self.model.wv[w] for w in words]
            words = np.array(words)
            words = np.mean(words, axis=0)
            return words
        
    def load(self, model_path: str):
        self.model = FastText.load(model_path)
        
