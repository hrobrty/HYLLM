import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# 不在同一个目录时候
load_dotenv(dotenv_path='/Users/hrobrty/Desktop/AGI学习资料/.env')

if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("未找到 API_KEY，请检查 .env 文件！")
    st.stop()

api_key = os.getenv('DASHSCOPE_API_KEY')
base_url = os.getenv('DASHSCOPE_BASE_URL')

# 创建OpenAI的客户端
client = OpenAI(api_key=api_key, base_url=base_url)


def get_embedding(texts, model=os.getenv('EMBEDDING_MODEL')):
    data = client.embeddings.create(input=texts, model=model).data
    # for x in data:
    # print([x.embedding])  #以下是这个循环的简写
    return [x.embedding for x in data]


test_query = ['我']
vec = get_embedding(test_query)
print('=====向量======')
print(vec)
print('=====维度======')
print(len(vec[0]))
