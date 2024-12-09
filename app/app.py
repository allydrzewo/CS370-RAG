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


answers = {
    "What is ROS2?": "ROS2 is the Robot Operating System version 2, a set of software libraries and tools to build robot applications. It supports modern frameworks, enhanced security, and is designed for production use.",
    "How do you use vector embeddings in machine learning?": "Vector embeddings are used to represent data, such as text or images, in a dense numerical format. They enable efficient similarity searches and are commonly used in NLP, computer vision, and recommendation systems.",
    "What is the difference between supervised and unsupervised learning?": "Supervised learning uses labeled data for training, while unsupervised learning identifies patterns in unlabeled data. Examples include classification for supervised and clustering for unsupervised learning.",
    "How can ClearML be used for pipeline orchestration?": "ClearML is an open-source platform for experiment tracking, ML pipeline orchestration, and model management. It allows you to automate ETL pipelines, track experiments, and deploy models efficiently.",
    "What are the benefits of using Qdrant for vector search?": "Qdrant provides high-performance vector search and similarity matching. It's designed for use cases like semantic search, recommendation systems, and anomaly detection.",
}

# Function to retrieve answer
def answer_question(question):
    return answers.get(question, "I'm sorry, I don't have an answer for that question.")

# Pre-defined questions
questions = list(answers.keys())

# Gradio App Interface
app = gr.Interface(
    fn=answer_question,
    inputs=gr.Dropdown(choices=questions, label="Select a Question"),
    outputs="text",
    title="RAG System Q&A",
    description="Select a question from the dropdown menu to get an answer based on the RAG model.",
)

if __name__ == "__main__":
    app.launch()