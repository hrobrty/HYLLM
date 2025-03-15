from pdfminer.high_level import extract_pages  # pip3 install pdfminer.six
from pdfminer.layout import LTTextContainer
import chromadb  # pip install chromadb
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import os

# 不在同一个目录时候
load_dotenv(dotenv_path='/Users/hrobrty/Desktop/AGI学习资料/.env')

if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("未找到 API_KEY，请检查 .env 文件！")
    st.stop()

api_key = os.getenv('DASHSCOPE_API_KEY')
base_url = os.getenv('DASHSCOPE_BASE_URL')

client = OpenAI(api_key=api_key, base_url=base_url)


def get_embeddings(texts, model=os.getenv('EMBEDDING_MODEL')):
    data = client.embeddings.create(input=texts, model=model).data
    # for x in data:
    # print([x.embedding])  #以下是这个循环的简写
    return [x.embedding for x in data]

# 从 PDF 文件中（按指定页码）提取文字,min_line_length字符数大于或等于10，那么这行文本就有资格被保留

def extract_text_from_pdf(filename, page_numbers=None, min_line_length=10):
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成一段一段的内容
    # 按照自然段落处理
    lines = full_text.split('\n\n')
    for text in lines:
        text = text.strip()
        if not text:
            continue
        para_text = ' '.join(line.strip()
                             for line in text.split('\n') if line.strip())
        if len(para_text) >= min_line_length:
            paragraphs.append(para_text)

    return paragraphs


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        """本地模式:保存向量内容到本地"""
        chroma_client = chromadb.PersistentClient(
            path=f'./chroma/{collection_name}')
        # 删除集合
        # chroma_client.delete_collection(collection_name)
        # 创建一个 存储对象collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

        # 向collection中添加文档与向量
    def add_documents(self, documents):
        batch_size = 10  # 假设这是你的批次大小
        current_id = 0  # 初始化当前ID
        for i in range(0, len(documents), batch_size):
            # documents[0:10] documents[11:20]  ...
            batch = documents[i:i + batch_size]
            # 生成ID列表，长度与当前批次中的文档数量相同
            batch_ids = [f"id{current_id + j}" for j in range(len(batch))]
            self.collection.add(
                embeddings=self.embedding_fn(batch),  # 每个文档的向量
                documents=batch,  # 文档的原文
                ids=batch_ids  # 每个文档的 id
            )
            # 更新current_id为下一个批次的起始ID
            current_id += len(batch)


if __name__ == '__main__':
    """
     1,加载数据
     2，转为文本
     3，切片
     4，向量话
     5，存入数据库
    """
    # 加载文档
    # file_path = '/Users/hrobrty/Desktop/AGI学习资料/6,RAG基础/RAG基础/RAG源代码/人事管理流程.pdf'
    file_path = '/Users/hrobrty/Desktop/AGI学习资料/6,RAG基础/RAG基础/RAG源代码/民法典.pdf'
    paragraphs = extract_text_from_pdf(file_path, min_line_length=2)
    print('paragraphs:', paragraphs[0:10])  # 输出10个chunks

    # 创建一个向量数据库对象
    # 法律咨询:legal_advice
    # 人事管理: personal_administration
    vector_db = MyVectorDBConnector("legal_advice", get_embeddings)
  # 往向量数据库中添加文档(向量化保存)
    print('chunks:', len(paragraphs))
    vector_db.add_documents(paragraphs)
