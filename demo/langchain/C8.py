import os
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
import logging
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7892"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7892"

logging.basicConfig(level=logging.INFO)

wiki=[]

try:
    loader = WikipediaLoader(
        query="Transformer architecture",
        lang="en",
        load_max_docs=10
    )

    docs=loader.load()
    for doc in docs:
        original=doc.metadata
        print(original)
        metadata={
            'title': original.get('title', '未知标题'),
            'source': original.get('source', '未知ID'),
        }

        doc.metadata=metadata
        wiki.append(doc)
except Exception as e:
    print(f"加载失败：{e}")

print("wiki数量:", len(wiki))

embed_model=HuggingFaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5')
vectorstore=Chroma.from_documents(wiki,embed_model)

# 配置元数据的字段信息
metadata_field_info=[
    AttributeInfo(
        name='title',
        description='文档标题',
        type='string',
    ),
    AttributeInfo(
        name='source',
        description='文档出处',
        type='string',
    )
]

llm=ChatDeepSeek(
    model='deepseek-chat',
    temperature=0,
    api_key=os.getenv('DEEPSEEK_KEY')
)

# SelfQueryRetriever支持自然语言查询 + 元数据过滤的检索器
retriever=SelfQueryRetriever.from_llm(
    llm=llm, # 调用模型把用户模型转换为 向量搜索和过滤条件
    vectorstore=vectorstore,
    document_contents='Wikipedia article content', # 文档正文是什么
    metadata_field_info=metadata_field_info, # 元数据有哪些字段
    enable_limit=True,
    verbose=True, # 日志
)

queries = [
    "查找标题中包含 Transformer 的文章",
    "查找摘要介绍深度学习的文章",
    "查找来源是 Wikipedia 的文章",
    "查找标题包含 Attention 的文章",
    "查找摘要中提到 neural network 的文章",
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"查询: {query}")
    print(f"{'='*50}")

    results = retriever.invoke(query)

    if results:
        print(f"找到 {len(results)} 篇文档\n")

        for i, doc in enumerate(results, 1):
            print(f"--- 文档 {i} ---")

            print("标题:")
            print(doc.metadata.get("title", "未知"))

            print("\n来源:")
            print(doc.metadata.get("source", "未知"))

            print("\n摘要:")
            print(doc.metadata.get("summary", "未知")[:300])

            print("\n正文片段:")
            print(doc.page_content[:500])

            print("\nMetadata:")
            print(doc.metadata)

            print("\n")
    else:
        print("未找到相关文档")