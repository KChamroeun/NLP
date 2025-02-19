import torch 
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
            return_tensors='pt',
            
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        
        # return {
        #     'input_ids': torch.tensor(inputs['input_ids'].squeeze(0)),
        #     'attention_mask': torch.tensor(inputs['attention_mask'].squeeze(0)),
        #     'labels': torch.tensor(label) 
        # }
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
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
        self.patience = config.get("patience", 2)  
        
    # def train(self, datasets: str):
    #     # print(f"Training with dataset at: {datasets}")
    #     self.train_dataset = PrepDataset(datasets, self.tokenizer, self.max_len)
    #     self.dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, shuffle=True)
    #     optimizer = AdamW(self.model.parameters(), lr=self.lr)
    #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(self.dataloader) * self.epochs)
        
    #     self.model.train()
    #     for epoch in range(self.epochs):
    #         total_loss = 0
    #         for batch in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
    #             input_ids = batch['input_ids'].to(self.device)
    #             attention_mask = batch['attention_mask'].to(self.device)
    #             labels = batch['labels'].to(self.device)
                
    #             outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
    #             loss = outputs.loss
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
    #             optimizer.zero_grad()
                
    #             total_loss += loss.item()
            
    #         avg_loss = total_loss / len(self.dataloader)
    #         print(f"Average loss for epoch {epoch + 1}: {avg_loss:.4f}")
    
    def train(self, train_data, val_data):
        self.train_dataset = PrepDataset(train_data, self.tokenizer, self.max_len)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.val_dataset = PrepDataset(val_data, self.tokenizer, self.max_len)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(self.train_dataloader) * self.epochs)
        
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Logging lists
        training_logs = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            
            train_acc, train_loss = self._train_epoch(self.train_dataloader, optimizer, scheduler, loss_fn)
            print(f'Train loss {train_loss} accuracy {train_acc}')
            
            val_acc, val_loss, val_precision, val_recall, val_f1 = self._eval_epoch(self.val_dataloader, loss_fn)
            print(f'Val loss {val_loss} accuracy {val_acc} precision {val_precision} recall {val_recall} f1 {val_f1}')
            
            # Log the metrics
            training_logs['epoch'].append(epoch + 1)
            training_logs['train_loss'].append(train_loss)
            training_logs['train_acc'].append(train_acc)
            training_logs['val_loss'].append(val_loss)
            training_logs['val_acc'].append(val_acc)
            training_logs['val_precision'].append(val_precision)
            training_logs['val_recall'].append(val_recall)
            training_logs['val_f1'].append(val_f1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_state.bin')
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print('Early stopping')
                break
        
        print('Training complete')
        
        # Convert logs to DataFrame for easy visualization
        logs_df = pd.DataFrame(training_logs)
        logs_df.to_csv('training_logs.csv', index=False)
        print('Training logs saved to training_logs.csv')

    def _train_epoch(self, data_loader, optimizer, scheduler, loss_fn):
        self.model.train()
        losses = []
        correct_predictions = 0
        
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

    def _eval_epoch(self, data_loader, loss_fn):
        self.model.eval()
        losses = []
        correct_predictions = 0
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                loss = loss_fn(outputs.logits, labels)
                
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        val_acc = correct_predictions.double() / len(data_loader.dataset)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        return val_acc, np.mean(losses), val_precision, val_recall, val_f1
            
            
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
        