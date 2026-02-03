import os
from pathlib import Path
from typing import Union, List

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.openai_like import OpenAILike


## 增加自定义loader解析为document

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError(
        "PaddleOCR is not installed. Please run 'pip install \"paddlepaddle<=2.6\" and \"paddleocr<3.0\"'")


class ImageOCRReader(BaseReader):
    def __init__(self, lang='ch', use_gpu=False, **kwargs):
        super().__init__()
        self.lang = lang
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, **kwargs)

    def load_data(self, file: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        if isinstance(file, (str, Path)):
            files = [file]
        else:
            files = file
        documents = []
        for image_path in files:
            image_path_str = str(image_path)
            result = self._ocr.ocr(image_path_str, cls=True)

            if not result or not result[0]:
                print(f"waring: No text detected in {image_path_str}")
                continue

            text_blocks = []
            confidences = []

            for i, line in enumerate(result[0]):
                text = line[1][0]
                confidence = line[1][1]
                text_blocks.append(f"[Text Block {i + 1}] (conf: {confidence}): {text}")
                confidences.append(confidence)
            full_text = "\n".join(text_blocks)

            avg_confidence = np.mean(confidences) if confidences else 0.0

            doc = Document(
                text=full_text,
                metadata={
                    'image_path': image_path_str,
                    "ocr_model": 'PP-OCRv4',
                    "language": self.lang,
                    "num_text_blocks": len(text_blocks),
                    "avg_confidence": float(avg_confidence)
                }
            )
            documents.append(doc)

        return documents


def setup_env():
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")

    Settings.llm = OpenAILike(api_key=api_key, api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                              is_chat_model=True, model="qwen-plus")

    Settings.embed_model = DashScopeEmbedding(model_name= DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3, api_key=api_key)

    print("LlamaIndex env setup complete")

def main():
    # --- 1. 准备工作：创建示例图片 ---
    print("--- Step 1: Preparing sample images ---")
    data_dir = Path("homework_data/ocr_images")
    data_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in data_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    # --- 2. 使用 ImageOCRReader 加载图像并生成 Document ---
    print("\n--- Step 2: Loading images with ImageOCRReader ---")
    # 需要预先安装GPU驱动：https://www.paddleocr.ai/latest/quick_start.html
    reader = ImageOCRReader(lang='en', use_gpu=True)
    documents = reader.load_data(image_files)

    print(f"Successfully loaded {len(documents)} documents from images.")
    for doc in documents:
        print("\n--- Document ---")
        print(f"Text: {doc.text[:100]}...")
        print(f"Metadata: {doc.metadata}")

    # --- 3. 配置 LlamaIndex 环境 ---
    print("\n--- Step 3: Setting up LlamaIndex environment ---")
    setup_env()

    # --- 4. 构建索引并进行查询 ---
    print("\n--- Step 4: Building index and querying ---")
    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 查询1: 关于 LlamaIndex 的问题
    question1 = "What is LlamaIndex?"
    print(f"\nQuerying: {question1}")
    response1 = query_engine.query(question1)
    print(f"Response: {response1}")

    # 查询2: 关于截图内容的问题
    question2 = "截图里的用户名是什么？"
    print(f"\nQuerying: {question2}")
    response2 = query_engine.query(question2)
    print(f"Response: {response2}")

    # 查询3: 关于路牌的问题
    question3 = "红色的牌子上写了什么？"
    print(f"\nQuerying: {question3}")
    response3 = query_engine.query(question3)
    print(f"Response: {response3}")

    # 查询4: 家庭作业的ocr输出结果里，我是在哪个文件路径下？
    question4 = "homework-02的ocr输出结果里，我是在哪个文件路径下？"
    print(f"\nQuerying: {question4}")
    response4 = query_engine.query(question4)
    print(f"Response: {response4}")

if __name__ == "__main__":
    main()