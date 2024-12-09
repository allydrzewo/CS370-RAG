# Utility functions for preprocessing, embedding extraction, etc.

from transformers import AutoTokenizer

def preprocess_data(text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(text, truncation=True, padding=True, return_tensors="pt")