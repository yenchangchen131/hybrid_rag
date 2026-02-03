import os
from openai import OpenAI
from dotenv import load_dotenv

# 載入 .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """
    取得文字的 Embedding 向量
    """
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

if __name__ == "__main__":
    # Test
    emb = get_embedding("測試")
    print(f"Embedding length: {len(emb)}")
