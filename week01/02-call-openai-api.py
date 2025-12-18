import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)

response = client.chat.completions.create(model="deepseek-chat", messages=[
    {"role": "user", "content": "Hello world!"}
])
print(response.choices[0].message.content)