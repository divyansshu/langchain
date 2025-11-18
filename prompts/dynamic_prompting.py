from google import genai
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import load_prompt

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY') 
client = genai.Client(api_key=api_key)

st.header('Research Tool')

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# user_input = st.text_input('Enter Prompt')

template = load_prompt('template.json')

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input 
})

if st.button('Summarize'):
    response = client.models.generate_content(
        model = 'gemini-2.5-flash',
        contents = prompt
    )
    st.write(response.text)