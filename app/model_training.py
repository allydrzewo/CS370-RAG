import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR

def load_data():
    # This function loads your dataset (replace with actual dataset path)
    dataset = load_dataset('./data/raw_data/ros2_docs.txt')
    return dataset

def fine_tune_model():
    model_name = 'bert-base-uncased'  
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = load_data()
    train_data, val_data = train_test_split(dataset['train'], test_size=0.2) 

    # Tokenization
    train_encodings = tokenizer(train_data['text'], truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_data['text'], truncation=True, padding=True, max_length=512)

    # Convert to Dataset format
    train_inputs = torch.tensor(train_encodings['input_ids'])
    val_inputs = torch.tensor(val_encodings['input_ids'])
    train_labels = torch.tensor(train_data['labels'])
    val_labels = torch.tensor(val_data['labels'])

    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # Training loop
    model.train()
    for epoch in range(3):  # 3 epochs
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Save fine-tuned model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

    return model, tokenizer