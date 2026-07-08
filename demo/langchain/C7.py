import os

from langchain_openai import ChatOpenAI

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# 文档加载
markdown_path='easy-rl-chapter1.md'
loader = TextLoader(
    "easy-rl-chapter1.md",
    encoding="utf-8"
)
docs=loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
texts=text_splitter.split_documents(docs)

# 初始化中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(texts)

question='文中举了哪些例子'
retrieved_docs=vectorstore.similarity_search(question,k=3) # 查询

docs_content='\n\n'.join(doc.page_content for doc in retrieved_docs) # 准备上下文

prompt=ChatPromptTemplate.from_template(
    """
    请确保你的回答完全基于这些上下文。
        如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”
          
        上下文:
        {context}
        
        问题: {question}
        
        回答:"""
)

llm = ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("GLM_KET"),
    base_url="https://aihubmix.com/v1"
)

answer=llm.invoke(prompt.format(question=question,context=docs_content))
print(answer)