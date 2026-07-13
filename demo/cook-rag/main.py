import os
from pathlib import Path
from typing import List
import sys
from dotenv import load_dotenv
import logging
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

sys.path.append(str(Path(__file__).parent)) # 把当前 Python 文件所在的目录加入模块搜索路径 (sys.path)，这样可以导入同目录下或其子目录中的模块。
load_dotenv()

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self,config : RAGConfig = None ):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """

        self.config = config or DEFAULT_CONFIG
        self.data_module=None
        self.index_module=None
        self.retrieval_module=None
        self.generation_module=None
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f'数据路径不存在： {self.config.data_path}')

        if not os.getenv('DEEPSEEK_API_KEY'):
            raise ValueError(f'请设置API_KEY')

    def initialize_system(self):
        """初始化所有模块"""

        self.data_module = DataPreparationModule(self.config.data_path)
        self.index_module=IndexConstructionModule(model_name=self.config.embedding_model,index_save_path=self.config.index_save_path)
        self.generation_module = GenerationIntegrationModule(model_name=self.config.llm_model,temperature=self.config.temperature,max_tokens=self.config.max_tokens)
        print("✅ 系统初始化完成！")
    def build_knwoledge_base(self):
        """构建知识库"""
        print('\n正在构建知识库')

        vectorstore=self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            print('加载食谱文档...')
            self.data_module.load_documents()
            print('进行文本分块')
            chunks=self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            print("加载食谱文档...")
            self.data_module.load_documents()

            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            print("保存向量索引...")
            self.index_module.save_index()

        print('初始化检索优化...')
        self.retrieval_module = RetrievalOptimizationModule(vectorstore,chunks)

        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def ask_question(self,question:str,stream:bool=False):
        """
        回答用户问题
        Args:
            question: 用户问题
            stream: 是否流式输出
        Returns:
            生成的回答或生成器
        """

        if not all([self.retrieval_module,self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 查询路由
        route_type=self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 智能查询重写
        if route_type=='List':
            # 列表查询保持原查询
            rewritten_query=question
        else:
            # 智能重写
            rewritten_query=self.generation_module.query_rewrite(question)

        # 检索相关子块
        print('检索相关文档')
        filters=self._extract_filter_from_query(question)
        if filters:
            print(f'应用过滤条件')
            relevant_chunks=self.retrieval_module.metadata_filter_search(rewritten_query,filters,top_k=self.config.top_k)
        else:
            relevant_chunks=self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"


        if route_type=='List':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            relevant_docs=self.data_module.get_parent_documents(relevant_chunks)

            doc_names=[]
            for doc in relevant_docs:
                dish_name=doc.metadata.get('dish_name','未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档：{', '.join(doc_names)}")
            return self.generation_module.generate_list_answer(question,relevant_chunks)
        else:
            # 详细查询，需要完整文档并生成详细回答
            relevant_docs=self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")

            # 根据路由回答自动选择回答模式
            if route_type=='detail':
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(relevant_chunks,relevant_docs)
                return self.generation_module.generate_step_by_step_answer(relevant_chunks,relevant_docs)
            else:
                if stream:
                    return self.generation_module.generate_basic_answer_stream(relevant_chunks,relevant_docs)
                return self.generation_module.generate_basic_answer(relevant_chunks,relevant_docs)

    def _extract_filter_from_query(self,query:str)->dict:
        """从用户提问中提取元数据过滤条件"""

        filters={}

        # 分类
        category_keywords=DataPreparationModule.get_support_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category']=cat
                break
        # 难度
        difficulty_keywords=DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        self.initialize_system()

        self.build_knwoledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input=input('\n您的问题：').strip()
                if user_input.lower() in ['退出','quit','exit','']:
                    break

                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    for chunk in self.ask_question(user_input,stream=use_stream):
                        print(chunk,end='',flush=True)
                    print('\n')
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

            print("\n感谢使用尝尝咸淡RAG系统！")

def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()