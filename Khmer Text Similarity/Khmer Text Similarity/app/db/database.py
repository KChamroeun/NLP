import csv
import psycopg2
import numpy as np
import pandas as pd
from tqdm import tqdm
from supabase import create_client, Client

url: str = "https://evdaaabkuvrgufbvhmsd.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV2ZGFhYWJrdXZyZ3VmYnZobXNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTc0MzkyMjUsImV4cCI6MjAzMzAxNTIyNX0.Yd9weYFXPJEIeR293pKHh8bOBaGVuQjcH5gj14dR8Gw"
supabase: Client = create_client(url, key)


def connect_to_db():
    return psycopg2.connect(
        host="aws-0-ap-southeast-1.pooler.supabase.com",
        database="postgres",
        user="postgres.evdaaabkuvrgufbvhmsd",
        password="!KsJ!8Eqr%Pr@4V"
    )


def insert_data_from_csv(filepath):
    conn = connect_to_db()
    cur = conn.cursor()
    datasets = pd.read_csv(filepath)
    for _, row in tqdm(datasets.iterrows(), total=len(datasets)):
        # Extract data
        title = row['title']
        file_name = row['filename']
        vector = row['embedding']

        with open(f'/Users/bormeychanchem/Desktop/fyp2/kss/datasets/documents/{row["filename"]}') as content_file:
            content = content_file.read()

        # Format the vector to be in the pgvector format
        vector = "{" + vector.strip("[]") + "}"

        # Prepare the SQL command
        sql_command = """
        
        INSERT INTO documents (title, filename, embedding, content)
        VALUES (%s, %s, %s, %s);
        """
        cur.execute(sql_command, (title, file_name, vector, content))

    # Commit changes and close connection
        conn.commit()
    cur.close()
    conn.close()


def fetch_document_vectors():
    response = supabase.table("documents").select("title, embedding, content").execute()
    documents = []
    for record in response.data:
        title = record['title']
        vector = np.array(record['embedding'])
        content = record['content']
        documents.append((title, vector, content))

    return documents

# insert_data_from_csv('/Users/bormeychanchem/Desktop/fyp2/app/db/new_input_data_title.csv')
