import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

# 1. Test Supabase
print("\nTesting Supabase connection...")

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

response = supabase.table("documents").select("id").limit(1).execute()

print("Supabase connected successfully!")
print(f"Documents found: {len(response.data)}")


# 2. Test OpenAI Embeddings
print("\nTesting OpenAI Embeddings...")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="This is my first RAG Lab embedding."
)

embedding = response.data[0].embedding

print("OpenAI connected successfully!")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")