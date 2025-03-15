import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm

# 不在同一个目录时候
load_dotenv(dotenv_path='/Users/hrobrty/Desktop/AGI学习资料/.env')

if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("未找到 API_KEY，请检查 .env 文件！")
    st.stop()

api_key = os.getenv('DASHSCOPE_API_KEY')
base_url = os.getenv('DASHSCOPE_BASE_URL')

# 创建OpenAI的客户端
client = OpenAI(api_key=api_key, base_url=base_url)


def get_embeddings(texts, model=os.getenv('EMBEDDING_MODEL')):
    data = client.embeddings.create(input=texts, model=model).data
    # for x in data:
    # print([x.embedding])  #以下是这个循环的简写
    return [x.embedding for x in data]


'''余弦距离 -- 值越大越相似
dot(a, b)为a和b对应元素的乘积之和除以欧式范数的乘积,得到两个向量间的余弦相似度
norm(a)向量的长度(向量有方向和长度)
'''
# 1.计算余弦相似度


def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

# 2.计算欧式距离


def l2(a, b):
    '''欧式距离 -- 越小越相似'''
    x = np.asarray(a)-np.asarray(b)  # 转为numpy数组
    return norm(x)  # 向量x的欧式范数即长度,即两个向量a和b之间的欧式距离


query = "国际争端"

# 文档
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
    '今天的天气很适宜户外活动'
]

query_vec = get_embeddings([query])[0]
doc_vecs = get_embeddings(documents)

# 余弦相似度是比较角度
print("=====余弦相似度=====")
print("自己对比: ", cos_sim(query_vec, query_vec))

print("==与documents对比==")
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))

# 欧式距离是比较大小
print("=====欧式距离=====")
print("自己对比: ", l2(query_vec, query_vec))

print("==与documents对比==")
for vec in doc_vecs:
    print(l2(query_vec, vec))
