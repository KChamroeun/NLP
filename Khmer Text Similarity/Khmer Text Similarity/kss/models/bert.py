import torch 
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

class PrepDataset(Dataset):
    def __init__(self, datasets, tokenizer, max_len):
        # print(f"Loading data from: {datasets}")
        # self.data = pd.read_csv(datasets, header=None)
        self.data = datasets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_file = self.data.iloc[idx]['source']
        target_file = self.data.iloc[idx]['target']
        label = self.data.iloc[idx]['class']

        with open(source_file, 'r', encoding='utf-8') as file:
            source_text = file.read()
        
        with open(target_file, 'r', encoding='utf-8') as file:
            target_text = file.read()
        
        inputs = self.tokenizer.encode_plus(
            source_text,
            target_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,       
            return_tensors='pt'

        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids'].squeeze(0)),
            'attention_mask': torch.tensor(inputs['attention_mask'].squeeze(0)),
            'labels': torch.tensor(label) 
        }
        
        
        
class BERT:
    def __init__(self, config) -> None:
        # print("Initializing BERT model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config["model_name"]) #'bert-base-uncased'
        # self.train_dataset = PrepDataset(self.file_path, self.tokenizer, self.max_len)
        # self.dataloader = DataLoader(self.train_dataset, batch_size= config["batch_size"], shuffle=True)
        self.model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)
        self.model.to(self.device)
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.max_len = config["max_len"]        
        self.batch_size = config["batch_size"]
        self.similarity_measure = config["similarity_measure"]
        self.decision_threshold = config["decision_threshold"]
        
    def train(self, datasets: str):
        # print(f"Training with dataset at: {datasets}")
        self.train_dataset = PrepDataset(datasets, self.tokenizer, self.max_len)
        self.dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(self.dataloader) * self.epochs)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Average loss for epoch {epoch + 1}: {avg_loss:.4f}")
            
            
            ## add training loop to stop the training once best result is reached
            ## add log to training for graphs
        
    def compare(self, source, target):
        source = self.get_embedding(source).flatten()
        target = self.get_embedding(target).flatten()

        if self.similarity_measure == "cosine_similarity":
            dist = distance.cosine(source, target)
        elif self.similarity_measure == "euclidean_distance":
            dist = distance.euclidean(source, target)
        return dist, source, target
    # , int(dist < self.decision_threshold)

        
    def get_embedding(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_input)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings
        
    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
    def load(self, model_path: str):
        self.model = BertModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        