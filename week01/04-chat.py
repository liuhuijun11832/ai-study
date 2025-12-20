import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("V3_API_KEY")
client = OpenAI(
    base_url="https://api.vveai.com/v1",
    api_key=api_key
)

def query(user_prompt):
    """
    发送用户提示到 OpenAI API 并返回响应内容

    参数:
        user_prompt (str): 用户输入的提示内容

    返回:
        str: AI 的响应内容
    """
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role":"user", "content":user_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"{str(e)}"


if __name__ == "__main__":
    print(query("早上好"))