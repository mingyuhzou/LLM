import logging
import hashlib
from typing import List,Dict,Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger=logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self,vectorstore:FAISS,chunks:List[Document]):
        """
        初始化检索优化模块
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档快列表
        """
        self.vectorstore=vectorstore
        self.chunks=chunks
        self.setup_retrievers()
    def setup_retrievers(self):
        logger.info('正在设置检索器...')

        # 向量检索器
        self.vector_retriever= self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5
            }
        )

        # BM25检索器
        self.bm25_retriever=BM25Retriever.from_documents(
            self.chunks,
            k=5
        )
    def hybrid_search(self,query,top_k)->List[Document]:
        """
        混合检索，使用RRF重排
        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表

        """

        # 获取检索结果
        vector_docs=self.vector_retriever.invoke(query)
        bm25_docs=self.bm25_retriever.invoke(query)

        rerank_docs=self._rrf_rerank(vector_docs,bm25_docs)
        return rerank_docs[:top_k]
    def metadata_filter_search(self,query:str,filters:Dict[str,Any],top_k: int =5) -> List[Document]:
        """
        带元数据过滤的检索,必须包含过滤条件
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量

        Returns:
            过滤后的文档列表
        """

        docs=self.hybrid_search(query,top_k)

        filter_docs=[]
        for doc in docs:
            match=True
            for k,v in filters.items():
                if k in doc.metadata:
                    if isinstance(v,List):
                        if doc.metadata[k] not in v:
                            match=False
                            break
                    else:
                        if doc.metadata[k]!=v:
                            match=False
                            break
                else:
                    match = False
                    break
            if match:
                filter_docs.append(doc)
                if len(filter_docs)>top_k:
                    break
        return filter_docs

    def _rrf_rerank(self,vector_docs,bm25_docs,k=60) -> List[Document]:
        """
        RRF重排
        Args:
            vector_docs: 向量检索结果
            bm25_docs: 向量检索结果
            k: RRF参数，用于平滑排名
        Returns:
            重排后的参数文档
        """

        doc_scores={}
        doc_objects={}

        # 计算RRF分数
        for rank,doc in enumerate(vector_docs):
            doc_id=hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id]=doc

            rrf_score=1.0/(k+rank+1)
            doc_scores[doc_id]=doc_scores.get(doc_id,0)+rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        for rank,doc in enumerate(bm25_docs):
            doc_id=hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id]=doc

            rrf_score=1.0/(k+rank+1)
            doc_scores[doc_id]=doc_scores.get(doc_id,0)+rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 排序
        sorted_docs=sorted(doc_scores.items(),key=lambda x:x[1],reverse=True)

        reranked_docs=[]
        for doc_id,final_score in sorted_docs:
            if doc_id in doc_objects:
                doc= doc_objects[doc_id]
                doc.metadata['rrf_score']=final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")
        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs





