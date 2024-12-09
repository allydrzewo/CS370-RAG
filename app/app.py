import os
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model_training import fine_tune_model

# Check if the fine-tuned model exists
model_dir = './fine_tuned_model'
if not os.path.exists(model_dir):
    print("Fine-tuned model not found. Training a new model...")
    # Fine-tune the model
    model, tokenizer = fine_tune_model()
else:
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

def predict(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits.argmax(dim=-1).item()

# Create Gradio app interface
def create_gradio_app():
    gr.Interface(fn=predict,
                 inputs="text",
                 outputs="text",
                 live=True).launch()

if __name__ == '__main__':
    create_gradio_app()