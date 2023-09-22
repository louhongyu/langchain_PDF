#导入os来管理openaikey
import os

#引入OpenAI作为主要的大模型服务,注意这里的OpenAI是大写的
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

#导入stremlit来做UI和web的便捷展示
import streamlit as st

#从langchain对文件加载器中导入PyPDFLoder TODO:每个命名是有规则的
from langchain.document_loaders import PyPDFLoader

#从langchain的向量仓库中导入chroma
from langchain.vectorstores import Chroma

#从langchain导入向量工具
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#导入openaikey
os.environ['OPENAI_API_KEY']='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'

#创建OpenAI实例
llm=OpenAI(temperature=0.9)

embeddings = OpenAIEmbeddings()

#用PyPDFLoader上传PDF文件
loader=PyPDFLoader('数字疗法行业报告.pdf')

#用loader去分割PDF为一页页的
pages=loader.load_and_split()

#将文件的每一页用chroma转为向量后存到仓库
store=Chroma.from_documents(pages,collection_name='数字疗法行业报告.pdf')

#建立向量信息实例
vectorstore_info=VectorStoreInfo(
    name="数字疗法行业报告",
    description="一份数字疗法行业报告的PDF文件",
    vectorstore=store
)

#将文件向量信息放入langchain中toolkit
toolkit=VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor=create_vectorstore_agent(
    toolkit=toolkit,
    llm=llm,
    verbose=True
)

#todo:我的标题呢🦜️🔗
#用st在网页端做一个用户输入提示词的窗口
prompt=st.text_input('请输入提示词')

#如果用户输入提示词后点击enter
if prompt:
    #把提示词传入llm作为响应输出
    # response=llm(prompt)

    response=agent_executor(prompt)
    #在st的网页显示写入响应结果
    st.write(response)

    #用with在st的界面上做一个查找相似的内容扩展
    with st.expander('近似的内容'):
        #在向量数据库中按提示词的要求输出相似度高的内容作为搜索结果
        search=store.similarity_search_with_score(prompt)
        #将最优的解写入st页面
        st.write(search[0][0].page_content)




