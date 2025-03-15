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

# 创建大模型
client = OpenAI(api_key=api_key, base_url=base_url)

# 向量化


def get_embeddings(texts, model=os.getenv('EMBEDDING_MODEL')):
    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

# 大模型处理


def get_completion(prompt, model=os.getenv('MODEL_NAME')):
    '''封装 openai 接口'''
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content


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


class RAG_Bot:
    def __init__(self, vector_db: MyVectorDBConnector, llm_api, n_results=10):
        self.vector_db = vector_db  # 向量数据库
        self.llm_api = llm_api  # 大模型api函数
        self.n_results = n_results  # 从向量数据库检索匹配的结果

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)
        print('search_results:', search_results)
        # 2. 构建 Prompt
        info = search_results['documents'][0]
        prompt = f"""
            你是一个问答机器人。
            你的任务是根据下述给定的"已知信息"回答用户问题。
            确保你的回复完全依据下述已知信息。不要编造答案。
            如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

            已知信息:
            {info}

            用户的问题：
            {user_query}
            请用中文回答用户问题。
            """
        print(prompt)
        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response


if __name__ == '__main__':
    # 创建一个向量数据库对象
    vector_db = MyVectorDBConnector("personal_administration", get_embeddings)
    # 创建一个RAG机器人
    bot = RAG_Bot(vector_db, get_completion)

    # 查询信息
    user_query = "什么情况下可能被辞退处理"

    resoponse = bot.chat(user_query)
    print("答案:")
    print(resoponse)
