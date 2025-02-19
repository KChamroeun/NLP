import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from khmernltk import word_tokenize




def read_config_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as file:
        return json.load(file)
    


def preprocess(text):
    text = re.sub("[/#]", '\u0020', text)
    text = re.sub("\d+", '\u0020', text)
    text = text.replace('\xa0', '\u0020')
    text = re.sub('\u0020+', '\u0020', text)
    text = re.sub('[()“«»]', '', text)
    text = re.sub('៕។៛ៗ៚៙៘,.? ', '', text) 
    text = re.sub('០១២៣៤៥៦៧៨៩0123456789', '', text)
    text = re.sub('᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿', '', text)
    khmer_stopwords = set([" ", "ៗ", "។ល។", "៚", "។", "៕", "៖", "៙", "-", "...", "+", "=", ",", "–", "/",
                         "នេះ", "នោះ", "អ្វី", "ទាំង", "គឺ", "ដ៏", "ណា", "បើ",
                         "ខ្ញុំ", "អ្នក", "គាត់", "នាង", "ពួក", "យើង", "ពួកគេ", "លោក", "គេ", "នៃ", "នឹង", "នៅ",
                         "ដែល", "ដោយ", "ក៏", "ហើយ", "ដែរ", "ទេ", "ផង", "វិញ", "ខាង", "អស់",
                          "និង" , "ដែល", "ជា" ,"តែ" ,"ដើម្បី" , "បាន", "យ៉ាង", "ទៀត",
                         "ប៉ុន្តែ" ,"ដោយសារ", "ពេលដែល" ,"ហើយ", "ដូចជា" , "ដូច្នេះ", "ពេលណាមួយ", "ទៅវិញ", "តែម្តង",
                         "ជាមួយ" ,"ដូចគ្នានិង","រួចហើយទេ","ឬ" , "ដោយ", 'ជាមួយនឹង'
                         ])
    text = text.lower()
    words = text.split(" ")
    words = [w for w in words if w not in khmer_stopwords]
    return words
    
def text_segmentation(text):
  '''
  segment raw text

  input type: string <document>

  return a list of words

  '''
  segmented_text = word_tokenize(text, return_tokens = True)
  return segmented_text

def remove_stopwords(text):
  '''
  remove predefined stop words

  input type: a list of words

  return a list of words without the stop words
  '''

  khmer_stopwords = set([" ", "ៗ", "។ល។", "៚", "។", "៕", "៖", "៙", "-", "...", "(", ")", "+", "=", ",", "»", "«", "–", "/",
                         "នេះ", "នោះ", "អ្វី", "ទាំង", "គឺ", "ដ៏", "ណា", "បើ",
                         "ខ្ញុំ", "អ្នក", "គាត់", "នាង", "ពួក", "យើង", "ពួកគេ", "លោក", "គេ", "នៃ", "នឹង", "នៅ",
                         "ដែល", "ដោយ", "ក៏", "ហើយ", "ដែរ", "ទេ", "ផង", "វិញ", "ខាង", "អស់",
                          "និង" , "ដែល", "ជា" ,"តែ" ,"ដើម្បី" , "បាន", "យ៉ាង", "ទៀត",
                         "ប៉ុន្តែ" ,"ដោយសារ", "ពេលដែល" ,"ហើយ", "ដូចជា" , "ដូច្នេះ", "ពេលណាមួយ", "ទៅវិញ", "តែម្តង",
                         "ជាមួយ" ,"ដូចគ្នានិង","រួចហើយទេ","ឬ" , "ដោយ", 'ជាមួយនឹង'
                         ])
  cleaned_text = [word for word in text.split(" ") if word.lower() not in khmer_stopwords]

  return cleaned_text


def preprocess_text(text):
  '''
  - segment text into words based level
  - remove stop words

  type: string <text>
  '''
#   preprocess_text = text_segmentation(text)
  preprocess_text = remove_stopwords(preprocess_text)
  return preprocess_text





def cosine_similarity(vector1, vector2):
  dot_product = np.dot(vector1, vector2)
  norm_vector1 = np.linalg.norm(vector1)
  norm_vector2 = np.linalg.norm(vector2)
  cosine_value = dot_product / (norm_vector1 * norm_vector2)
  return cosine_value

def calculate_cosine_similarity(documents, tfidf_data):
  num_doc = len(documents)
  cosine_similarity_matrix = np.zeros((num_doc, num_doc))
  for i in range(tfidf_data.shape[0]):
    for j in range(i, tfidf_data.shape[0]):
      cosine_similarity_matrix[i, j] = cosine_similarity(tfidf_data.iloc[i], tfidf_data.iloc[j])
      cosine_similarity_matrix[j, i] = cosine_similarity_matrix[i, j]
  return cosine_similarity_matrix


def euclidean_distance(vector1, vector2):
  euclidean_vector = np.linalg.norm(vector1 - vector2)
  return euclidean_vector

def calculate_euclidean_distance(documents, tfidf_data):
    num_doc = len(documents)
    euclidean_distance_matrix = np.zeros((tfidf_data.shape[0], tfidf_data.shape[0]))
    for i in range(tfidf_data.shape[0]):
        for j in range(i, tfidf_data.shape[0]):
            # Correctly calling the euclidean_distance function with parentheses
            euclidean_distance_matrix[i, j] = euclidean_distance(tfidf_data.iloc[i], tfidf_data.iloc[j])
            euclidean_distance_matrix[j, i] = euclidean_distance_matrix[i, j]  # Reusing the computed distance
    return euclidean_distance_matrix