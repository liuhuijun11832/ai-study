import logging
import os
from pathlib import Path

import dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels
from llama_index.llms.openai_like import OpenAILike
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

logging.basicConfig(level=logging.ERROR)

dotenv.load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"密钥为:{api_key}");
print("正在解析文件...")
script_path = Path(__file__).resolve()
script_dir = script_path.parent

docs_path = script_dir / "docs"

documents = SimpleDirectoryReader(str(docs_path)).load_data()

print("正在创建索引...")
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=DashScopeEmbeddings(
        model=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
    )
)

print("正在创建提问引擎...")
query_engine = index.as_query_engine(
    streaming=True,
    llm=OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    is_chat_model=True
)
)

print("正在生成回复...")
streaming_response = query_engine.query("我们公司项目管理应该用什么工具")
print("回答是：")
streaming_response.print_response_stream()
