import json
import mysql.connector

conn = mysql.connector.connect(
    host="localhost", user="root", password="passwd", database="tpk"
)
cursor = conn.cursor()

with open('D:/copbot/police_qa_dataset.json', 'r') as f:
    data = json.load(f)

for entry in data:
    cursor.execute("INSERT INTO qa_pairs (question, answer) VALUES (%s, %s)", 
                   (entry['question'], entry['answer']))
conn.commit()
print("""Data inserted successfully!""")
conn.close()