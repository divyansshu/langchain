from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

chat_history = [SystemMessage(content='You are a helpful AI chat Assistant')]

model = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash',
        google_api_key=api_key
        )

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content = response.content))
    print('AI: ', response.content)
print(chat_history)
