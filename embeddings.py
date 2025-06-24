import torch
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import os

# Check GPU availability
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

# Load dataset
with open('police_qa_dataset.json', 'r') as f:
    data = json.load(f)

questions = [entry['question'] for entry in data]
answers = [entry['answer'] for entry in data]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\']', '', text)
    text = text.replace('\u2019', "'")
    return text

cleaned_questions = [clean_text(q) for q in questions]
cleaned_answers = [clean_text(a) for a in answers]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

print("Generating embeddings for questions...")
question_embeddings = [get_embedding(q) for q in cleaned_questions]

# Define save directory
save_dir = 'D:/offline_copbot/models/'
os.makedirs(save_dir, exist_ok=True)

# Save tokenizer and model
print("Saving tokenizer and model...")
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

# Save embeddings, questions, and answers
print("Saving embeddings and data...")
np.save(os.path.join(save_dir, 'question_embeddings.npy'), np.array(question_embeddings))
with open(os.path.join(save_dir, 'cleaned_questions.json'), 'w') as f:
    json.dump(cleaned_questions, f)
with open(os.path.join(save_dir, 'cleaned_answers.json'), 'w') as f:
    json.dump(cleaned_answers, f)

# Test the chat function before saving
def chat(query):
    query_embedding = get_embedding(clean_text(query))
    similarities = cosine_similarity([query_embedding], question_embeddings)[0]
    best_idx = similarities.argmax()
    if similarities[best_idx] > 0.8:  # Confidence threshold
        return cleaned_answers[best_idx]
    return "I’m not sure about that. Please contact your local police station."

# Test
"""print("Testing chat function:")
print(chat("How do I report a stolen vehicle?"))
print(chat("how can i legally change my religion?"))"""

print("Model, tokenizer, and embeddings saved successfully to", save_dir)