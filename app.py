import torch
import numpy as np
import mysql.connector
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# 🔹 Load Gemini API
GEMINI_API_KEY = "api-key"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# 🔹 Load fine-tuned BERT Model
save_dir = 'D:/offline_copbot/models/'
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModel.from_pretrained(save_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 🔹 MySQL Connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="passwd",  # Replace with your actual password
    database="tpk"
)
cursor = conn.cursor()

# 🔹 Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\']', '', text)  
    return text

# 🔹 Load MySQL data
def load_data():
    cursor.execute("SELECT question, answer FROM qa_pairs")
    data = cursor.fetchall()
    questions, answers = zip(*data) if data else ([], [])
    return list(questions), list(answers)

cleaned_questions = [clean_text(q) for q in load_data()[0]]
cleaned_answers = [clean_text(a) for a in load_data()[1]]

# 🔹 Load question embeddings
question_embeddings = np.load(os.path.join(save_dir, 'question_embeddings.npy'), allow_pickle=True).tolist()

# 🔹 Function to get BERT embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# 🔹 Check if question is relevant
def is_relevant_match(query, matched_question):
    query_words = set(clean_text(query).split())
    matched_words = set(clean_text(matched_question).split())
    overlap = len(query_words.intersection(matched_words)) / len(query_words)
    return overlap > 0.5

# 🔹 Query Gemini API
def query_gemini(question):
    try:
        prompt = f"Answer this question only if it relates to police procedures, legal rights, or reporting crimes: {question}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else None
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return None

# 🔹 Add new data to MySQL
def add_to_dataset(question, answer):
    cursor.execute("INSERT INTO qa_pairs (question, answer) VALUES (%s, %s)", (question, answer))
    conn.commit()
    cleaned_questions.append(clean_text(question))
    cleaned_answers.append(clean_text(answer))
    question_embeddings.append(get_embedding(clean_text(question)))
    print(f"✅ Added to database: {question} -> {answer}")

# 🔹 Flask API Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    cleaned_query = clean_text(user_message)
    
    # Get BERT embedding for query
    query_embedding = get_embedding(cleaned_query)
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], question_embeddings)[0]
    best_idx = similarities.argmax()
    score = similarities[best_idx]
    matched_question = cleaned_questions[best_idx]

    if score > 0.95 and is_relevant_match(cleaned_query, matched_question):
        bot_response = cleaned_answers[best_idx]
    else:
        gemini_response = query_gemini(cleaned_query)
        if gemini_response:
            bot_response = gemini_response
            add_to_dataset(cleaned_query, gemini_response)  
        else:
            bot_response = "I'm not sure about that. Please contact your local police station."

    return jsonify({"response": bot_response, "score": float(score)})

# 🔹 Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
