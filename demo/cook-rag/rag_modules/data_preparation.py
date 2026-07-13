import logging
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Any

import dotenv
from langchain_core import documents
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger=logging.getLogger(__name__)
dotenv.load_dotenv()

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS=list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS=['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self,data_path):

        self.data_path=data_path
        self.documents: List [ Document] = []
        self.chunks: List[Document] =[]
        self.parent_child_map: Dict[str, str] = {}

    def load_documents(self):
        logger.info(f'正在从 {self.data_path} 加载文档... ')
        data_path_obj=Path(self.data_path)
        
        documents=[]
        
        for md_file in data_path_obj.rglob('*.md'): # 递归查找所有符合条件的文件或目录
            try:
                with open(md_file,'r',encoding='utf-8') as f:
                    content=f.read()
                # 为父文档分配确定性的唯一ID
                try:
                    data_root=Path(self.data_path).resolve()
                    relative_path=Path(md_file).resolve().relative_to(data_root).as_posix() # 把文档的绝对路径转换为相对数据目录的相对路径，as_posix将Path对象转换为字符串
                except Exception:
                    relative_path = Path(md_file).as_posix()
                # 编码
                parent_id=hashlib.md5(relative_path.encode('utf-8')).hexdigest()

                # 创建Document对象 只对父文档
                doc=Document(
                    page_content=content,
                    metadata={
                        'source':str(md_file),
                        'parent_id':parent_id,
                        'doc_type':'parent' # 标记为父文档
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        for doc in documents:
            self._enhance_metadata(doc)

        self.documents=documents
        logger.info(f'成功加载 {len(documents)} 个文档')
        return documents
    def _enhance_metadata(self,doc):
        """
        增强文档元数据，难度 类别
        """
        file_path=Path(doc.metadata.get('source','')) # Path("C1/test.md")->('C1', 'test.md')
        path_parts=file_path.parts

        doc.metadata['category']='其他'
        for k,v in self.CATEGORY_MAPPING.items():
            if k in path_parts:
                doc.metadata['category']=v
                break

        doc.metadata['dish_name']=file_path.stem# 获取不包含后缀的文件名

        content=doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_support_categories(cls):
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self):
        '''
        对文档分块 得到chunk也就是子文档
        '''
        if not self.documents:
            raise ValueError('请先加载文档')

        chunks=self._markdown_header_split()

        for i,chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                chunk.metadata['chunk_id']=str(uuid.uuid4()) # 分割失败，生成一个随机ID
            chunk.metadata['batch_index']=i
            chunk.metadata['chunk_size']=len(chunk.page_content)

        self.chunks=chunks
        logger.info(f'Markdown分块完成，共生成 {len(chunks)} 个chunk')
        return chunks

    def _markdown_header_split(self) -> List[str]:
        '''
        使用markdown标题分割器进行结构化分割
        '''

        headers_to_split_on=[
            ('#','主标题'),
            ('##', '一级标题'),
            ('###', '三级标题'),
        ]

        markdown_splitter=MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,# 保留标题
        )

        all_chunks=[]

        for doc in self.documents:
            try:
                content_preview=doc.page_content[:200]
                has_header=any(line.strip('#') for line in content_preview.split('\n'))

                if not has_header:
                    logger.warning(f"文档 {doc.metadata.get('dish_name','未知')} 内容中没有发现Markdown标题")
                    logger.debug(f'内容预览：{content_preview}')

                md_chunks=markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('dish_name','未知')} 分割成{len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks)<=1:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为子文档构建与父文档的关系
                parent_id=doc.metadata['parent_id']

                for i, chunk in enumerate(md_chunks):
                    child_id=str(uuid.uuid4())
                    # 合并元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        'chunk_id':child_id,
                        'parent_id':parent_id,
                        'type':'child',
                        'chunk_index':i
                    })
                    self.parent_child_map[child_id]=parent_id
                all_chunks.extend(md_chunks)
            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)
        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def filter_document_by_category(self,category):
        """
        按分类过滤文档
        """
        return [doc for doc in self.documents if doc.metadata['category']==category]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }

    def export_metadata(self,output_path):
        '''
        导出元数据到JSON文件
        '''
        import json

        metadata_list=[]
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据已导出到: {output_path}")
    def get_parent_documents(self,child_chunks):
        '''
        根据chunk获取对应的父文档
        '''

        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance={}
        parent_docs_map={}

        for chunk in child_chunks:
            parent_id=chunk.metadata['parent_id']
            if parent_id:
                parent_relevance[parent_id]=parent_relevance.get(parent_id, 0) + 1

                # 缓存
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get('parent_id') == parent_id:
                            parent_docs_map[parent_id]=doc
                            break
        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids=sorted(parent_docs_map.keys(),key=lambda x:parent_relevance[x],reverse=True)

        # 构建去重后的父文档列表
        parent_docs=[]
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        parent_info=[]
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs