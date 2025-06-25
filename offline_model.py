import torch
import json
import mysql.connector
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import tkinter as tk
import os
from tkinter import scrolledtext
import threading

# Gemini API setup
GEMINI_API_KEY = "AIzaSyAtMz4D8ZsF3cjPx6czQDF25Q93jFqxsfs"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Load saved model
save_dir = 'D:/offline_copbot/models/'
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModel.from_pretrained(save_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# MySQL setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Lynx_123@",  # Replace with your actual MySQL password
    database="tpk"
)
cursor = conn.cursor()

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\']', '', text)
    text = text.replace('\u2019', "'")
    return text

# Load data from MySQL
def load_data_from_mysql():
    cursor.execute("SELECT question, answer FROM qa_pairs")
    data = cursor.fetchall()
    questions, answers = zip(*data) if data else ([], [])
    return list(questions), list(answers)

# Load and clean data
cleaned_questions = [clean_text(q) for q in load_data_from_mysql()[0]]
cleaned_answers = [clean_text(a) for a in load_data_from_mysql()[1]]
question_embeddings = np.load(os.path.join(save_dir, 'question_embeddings.npy'), allow_pickle=True).tolist()

# Embedding function
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Query Gemini
def query_gemini(question):
    try:
        prompt = f"Answer this question only if it relates to police procedures, legal rights, or reporting crimes: {question}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else None
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return None

# Add to MySQL
def add_to_dataset(question, answer):
    cursor.execute("INSERT INTO qa_pairs (question, answer) VALUES (%s, %s)", (question, answer))
    conn.commit()
    cleaned_questions.append(clean_text(question))
    cleaned_answers.append(clean_text(answer))
    question_embeddings.append(get_embedding(clean_text(question)))
    print(f"Added new Q&A pair to MySQL: {question} -> {answer}")

# Relevance check
def is_relevant_match(query, matched_question):
    query_words = set(clean_text(query).split())
    matched_words = set(clean_text(matched_question).split())
    overlap = len(query_words.intersection(matched_words)) / len(query_words)
    return overlap > 0.5

# Chat function (with Gemini for now)
stop_flag = False
def chat(query):
    global stop_flag
    cleaned_query = clean_text(query)
    query_embedding = get_embedding(cleaned_query)
    similarities = cosine_similarity([query_embedding], question_embeddings)[0]
    best_idx = similarities.argmax()
    score = similarities[best_idx]
    matched_question = cleaned_questions[best_idx]

    if score > 0.95 and is_relevant_match(query, matched_question):
        response = cleaned_answers[best_idx]
    else:
        if cleaned_query not in cleaned_questions:
            gemini_response = query_gemini(query)
            if gemini_response:
                response = gemini_response
                add_to_dataset(query, gemini_response)
            else:
                response = "I’m not sure about that. Please contact your local police station."
        else:
            response = "I’m not sure about that. Please contact your local police station."
    
    return response, score

# GUI setup
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline Police Chatbot (with Gemini for Development)")

        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.root.geometry(f"{window_width}x{window_height}")

        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=0)
        self.root.grid_columnconfigure(3, weight=0)
        self.root.configure(bg="#f0f0f0")

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, bg="#ffffff", fg="#000000", font=("Arial", 12)
        )
        self.chat_display.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
        self.chat_display.tag_configure("user", foreground="#1e90ff", font=("Arial", 12, "bold"))
        self.chat_display.tag_configure("bot", foreground="#228b22", font=("Arial", 12))

        # Input field
        self.input_field = tk.Entry(root, font=("Arial", 12))
        self.input_field.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.input_field.bind("<Return>", lambda event: self.send_message())  # Bind Enter key to send

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.send_button.grid(row=1, column=1, padx=5, pady=10)

        # Stop button
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_message, bg="#f44336", fg="white", font=("Arial", 12))
        self.stop_button.grid(row=1, column=2, padx=5, pady=10)

        # Clear button
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_chat, bg="#ff9800", fg="white", font=("Arial", 12))
        self.clear_button.grid(row=1, column=3, padx=5, pady=10)

        self.running = False
        self.typing = False
        self.after_id = None

    def send_message(self):
        if self.running:
            return
        query = self.input_field.get().strip()
        if not query:
            return

        self.chat_display.insert(tk.END, f"You: {query}\n", "user")
        self.input_field.delete(0, tk.END)
        self.running = True
        self.stop_flag = False

        def process_response():
            response, score = chat(query)
            if not self.stop_flag:
                self.type_response(f"Bot: {response} (Score: {score:.4f})\n")
            self.running = False

        threading.Thread(target=process_response).start()

    def type_response(self, response):
        self.typing = True
        self.chat_display.insert(tk.END, "", "bot")
        self.current_response = response
        self.current_index = 0

        def type_char():
            if self.current_index < len(self.current_response) and not self.stop_flag:
                self.chat_display.insert(tk.END, self.current_response[self.current_index], "bot")
                self.current_index += 1
                self.chat_display.see(tk.END)
                self.after_id = self.root.after(10, type_char)
            else:
                self.typing = False
                self.after_id = None
                self.chat_display.insert(tk.END, "\n", "bot")

        type_char()

    def stop_message(self):
        global stop_flag
        stop_flag = True
        self.running = False
        self.typing = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.chat_display.insert(tk.END, "\nBot: Response stopped.\n", "bot")
        self.chat_display.see(tk.END)

    def clear_chat(self):
        self.chat_display.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

    # Clean up MySQL connection
    cursor.close()
    conn.close()