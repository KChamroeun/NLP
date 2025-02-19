import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np

class PrepDataset(Dataset):
    def __init__(self, datasets, tokenizer, max_len):
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

class BERT3:
    def __init__(self, config) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        self.model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)
        self.model.to(self.device)
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.max_len = config["max_len"]
        self.batch_size = config["batch_size"]
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, correct_bias=False)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.history = defaultdict(list)
        self.output_dir = config.get("output_dir", "output")

    def train(self, datasets, val_df = 'datasets/test.csv'):
        self.train_dataset = PrepDataset(datasets, self.tokenizer, self.max_len)
        self.val_dataset = PrepDataset(val_df, self.tokenizer, self.max_len)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        total_steps = len(self.train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        best_accuracy = 0
        patience = 3
        patience_counter = 0

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)

            train_acc, train_loss = self.train_epoch(self.train_dataloader, len(self.train_dataloader.dataset), scheduler)

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval_model(self.val_dataloader, len(self.val_dataloader.dataset))

            print(f'Val loss {val_loss} accuracy {val_acc}')
            print()

            self.history['train_acc'].append(train_acc)
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), f'{self.output_dir}/best_model_state.bin')
                best_accuracy = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

        self.plot_training_history()

    def train_epoch(self, data_loader, n_examples, scheduler):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for d in tqdm(data_loader, total=len(data_loader)):
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)

    def eval_model(self, data_loader, n_examples):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in tqdm(data_loader, total=len(data_loader)):
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    def plot_training_history(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='train accuracy')
        plt.plot(self.history['val_acc'], label='validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='train loss')
        plt.plot(self.history['val_loss'], label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig('training_history.png')
        plt.show()
