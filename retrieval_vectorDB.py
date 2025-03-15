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
    return [x.embedding for x in data]


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.PersistentClient(
            path=f'./chroma/{collection_name}')
        # 创建一个 存储对象collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    # query 查询字段
    # top_n 返回结果个数
    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(  # 基于向量相似度匹配封装在query的函数中
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results


if __name__ == '__main__':
    # 创建一个向量数据库对象
    vector_db = MyVectorDBConnector("personal_administration", get_embeddings)
    user_query = "员工假期有多少"
    # 检索: 从向量数据库中检索出相似度最高的三个
    results = vector_db.search(user_query, 5)
    print('======匹配的结果=======')
    print('results: ', results)
    for i in results['documents'][0]:
        print(i+'\n')
