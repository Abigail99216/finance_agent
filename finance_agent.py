!git clone https://www.modelscope.cn/datasets/BJQW14B/bs_challenge_financial_14b_dataset.git
import streamlit as st
import os
import sqlite3
import time
import jwt
import requests
from tqdm import tqdm
import pandas as pd
import jieba, json, pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#实际key,过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


#api调用
def ask_glm(content):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token("c9bc35e8e7c1c076a8aaba862efb19af.DhiaibnU9Mys34de", 1000)
    }

    data = {
        "model": "glm-4",
        "messages": [{"role": "user", "content": content}]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
#rerank_model.cuda() #这个要有英伟达gpu才能调用
rerank_model.eval()

#加载questions集
questions = json.load(open("博金杯金融问答_query.json"))

#读取训练数据
conn = sqlite3.connect('/content/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db')
cursor = conn.cursor()

# 获取所有表的名称
sql2 = """select name from sqlite_master where type='table' order by name"""
cursor.execute(sql2)
tables = cursor.fetchall()
# 初始化一个空的 DataFrame
csv_content= []
for table_idx in range(len(tables)):
  table_name = tables[table_idx][0]
  df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
conn.close()

#读取训练数据2
pdf_txt_dir = '/content/bs_challenge_financial_14b_dataset/pdf_txt_file'  # 替换为实际的文件夹路径
txt_content = []
txt_idx = 1  # 初始化 txt 编号

for filename in os.listdir(pdf_txt_dir):
  if filename.endswith('.txt'):
    filepath = os.path.join(pdf_txt_dir, filename)
    with open(filepath, 'r') as f:
      content = f.read()
    txt_content.append({
        'txt_num': 'txt_' + str(txt_idx),  # 使用 txt_idx 编号
        'content': content
    })
    txt_idx += 1  # 递增 txt 编号

all_content = csv_content.copy()
all_content.extend(txt_content)

all_content_words = [jieba.lcut(x['content']) for x in all_content]
bm25 = BM25Okapi(all_content_words)

for query_idx in tqdm(range(len(questions))):
  doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]['question']))
  #print(doc_socres)
  #找到得分最高的5个页面的索引
  max_score_page_idxs = doc_scores.argsort()[-5:]
  #将问题和索引配对
  pairs = []
  for idx in max_score_page_idxs:
    pairs.append((questions[query_idx]['question'], all_content[idx]['content']))
    #编码，包括填充和截断，并转化为pytorch张量
    inputs = tokenizer(pairs, padding = True, truncation = True, return_tensors = 'pt', max_length = 512)
    #一种上下文管理器，禁用了梯度计算
    with torch.no_grad():
      inputs = {key: inputs[key] for key in inputs.keys()}
      scores = rerank_model(**inputs, return_dict = True).logits.view(-1,).float()
      max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
      questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)
      prompt = '''你是一个金融专家，帮我结合给定的资料，回答一个问题。如果问题无法从资料中获得，请输出结合给定的资料，无法回答问题。
      资料：{0}

      问题：{1}
      '''.format(
          all_content[max_score_page_idx]['content'],
          questions[query_idx]["question"]
      )
      answer = ask_glm(prompt)['choices'][0]['message']['content']
      questions[query_idx]['answer'] = answer

with st.sidebar:
    st.title('金融专家gpt')
    st.write('支持的大模型包括ChatGLM3和4')
    # 初始化的对话
    if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "你好我是ChatGLM，有什么可以帮助你的？"}]
    
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好我是ChatGLM，有什么可以帮助你的？"}]
    
st.sidebar.button('清空聊天记录', on_click=clear_chat_history)

# Streamlit 应用程序界面
def main():
    st.title('金融专家助手')
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    messages = st.container(height=300)

if __name__ == "__main__":
    main()



