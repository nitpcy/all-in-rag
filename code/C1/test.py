import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv,find_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings # 注释掉未使用的导入
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek # 注释掉未使用的导入
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

#加载环境变量，embedding,llm
load_dotenv(find_dotenv())
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=2048,
#     api_key=os.getenv("DEEPSEEK_API_KEY")
# )
embeddings = OpenAIEmbeddings(model='Qwen/Qwen3-Embedding-8B',base_url="https://api.siliconflow.cn/v1")
llm = ChatOpenAI(model='Qwen/Qwen3-30B-A3B-Instruct-2507',base_url="https://api.siliconflow.cn/v1")

# --- 1. 加载文档并切分 ---
markdown_path = "data/C1/markdown/easy-rl-chapter1.md"
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)
# --- 2. 文本嵌入和向量存储 ---
vectorstore = FAISS.from_documents(chunks, embeddings)
retrieval = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. 构建问答链 ---
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:""")

def format_docs(docs):
    """将检索到的文档块列表格式化为单一字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

chain = {"context": retrieval | format_docs,"question":RunnablePassthrough()} | prompt | llm | StrOutputParser()
# chain = chain.with_config({"run_name": "MyCustomChain"})
# --- 4.生成回复 ---

question = "主要讲了什么内容？"
# answer = chain.invoke(question,{"run_name": "MyCustomtest1",})
# answer = chain.invoke(question)

@traceable
def rag(question: str) -> str:
    return chain.invoke(question)

answer = rag(question)
print(f"问题: {question}\n")
print(f"回答: {answer}\n")

