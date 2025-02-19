from keras_nlp.models import BertTokenizer, BertBackbone, BertPreprocessor
from utils import preprocess


class BERTKERAS:
    def __init__(self, config) -> None:
        self.preset_name = config["pretrained_weight"]

    def build_tokenizer(self, datasets):
        # vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        # for filepath in datasets:
        #     with open(filepath, "r") as input_file:
        #         vocab += preprocess(input_file.read())
        # vocab = list(set(vocab))

        # self.tokenizer = BertTokenizer(vocab=)
        # self.tokenizer = BertTokenizer.from_preset(self.preset_name)
        pass

    def train(self, datasets: str):
        raise NotImplementedError()

    def save(self, output_path: str):
        raise NotImplementedError()

    def compare(self, source, target):
        source = self.preprocessor(self.file_to_text(source))
        target = self.preprocessor(self.file_to_text(target))

        source = self.model(source)
        target = self.model(target)

        print(source)
        print(target)

    def file_to_text(self, filepath):
        with open(filepath, "r") as in_file:
            return in_file.read()

    def load(self, model_path: str):
        self.model = BertBackbone.from_preset(self.preset_name)
        self.tokenizer = BertTokenizer.from_preset(self.preset_name)
        self.preprocessor = BertPreprocessor(self.tokenizer)
