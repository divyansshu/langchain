from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key = api_key)


texts = [
    "New Delhi is the capital city of India and the seat of the government.",
    "The capital of India is New Delhi, located in the northern part of the country.",
    "Mumbai is a major financial center in India and home to the Bollywood film industry.",
    "Paris is the capital of France and a global center for art, fashion, and culture.",
    "Apple released a new smartphone this year with an improved camera and longer battery life.",
    "Many developers prefer Python for machine learning and data science tasks because of its libraries.",
    "TensorFlow and PyTorch are popular deep learning frameworks used for building neural networks.",
    "How to make a basic omelette: beat eggs, season, and cook in butter until set.",
    "Baking a chocolate cake requires flour, sugar, eggs, cocoa powder, and baking powder.",
    "The Hubble Space Telescope has provided deep views of distant galaxies and nebulae.",
    "Global warming and climate change are driven by increasing greenhouse gas emissions.",
    "Football (soccer) is a popular sport worldwide with major tournaments such as the FIFA World Cup.",
    "Lionel Messi is a famous football player known for his dribbling and goal-scoring ability."
]

result = [
    np.array(e.values) for e in client.models.embed_content(
        model='gemini-embedding-001',
        contents=texts,
        config=types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY')).embeddings
]

embeddings = np.vstack(result)

def query_similar(query, top_k=1):
    """Return top_k texts most similar to query using cosine similarity."""
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=[query],
        config=types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY')
    )
    q_emb = np.array(response.embeddings[0].values).reshape(1,-1)
    sims = cosine_similarity(q_emb, embeddings)[0] # shape: (n_texts,)
    idx = np.argsort(sims)[::-1][:top_k]
    return [(texts[i], float(sims[i])) for i in idx]


if __name__ == '__main__':
    query = input('Enter query: ').strip()
    top = query_similar(query, top_k=1)
    for text, score in top:
        print(f'{score:.4f} -- {text}')