from google import genai
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=api_key)

result = client.models.embed_content(
    model = 'gemini-embedding-001',
    contents = 'What is the meaning of life?'
)

print(result.embeddings)