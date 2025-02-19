import re
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess



# if __name__ == "__main__":
#     print(preprocess('\xa0Farencvaros\xa0'))


# exit()

class TFIDF:
    def __init__(self, config) -> None:
        self.similarity_measure = config["similarity_measure"]
        self.decision_threshold = config["decision_threshold"]
        self.model = TfidfVectorizer(
            input="filename",
            analyzer=preprocess,
        )
    
    def train(self, datasets: pd.DataFrame):
        datasets = datasets[datasets['class'] == 1]
        self.model = self.model.fit(datasets['source'])
    
    def compare(self, source, target):
        source = np.array(self.model.transform([source]).todense())[0]
        target = np.array(self.model.transform([target]).todense())[0]
        # dist = distance.cosine(source, target)
        # dist = distance.euclidean(source, target)
        if self.similarity_measure == "cosine_similarity":
            dist = distance.cosine(source, target)
        elif self.similarity_measure == "euclidean_distance":
            dist = distance.euclidean(source, target)
        return dist, int(dist < self.decision_threshold)
    
    def save(self, output_path: str):
        with open(output_path, "wb") as outfile:
            pickle.dump(self.model, outfile)
            
    def load(self, model_path: str):
        with open(model_path, "rb") as infile:
            self.model = pickle.load(infile)
    
if __name__ == "__main__":
    model = TFIDF()
    model.compare(None, None)
    model.train()