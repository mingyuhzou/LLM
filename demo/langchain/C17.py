import asyncio
import os

from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import QueryResponseDataset, FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.deepseek import DeepSeek
import dotenv

dotenv.load_dotenv()

Settings.llm=DeepSeek(model='deepseek-chat',temperature=0.1,aip_key=os.getenv('DEEPSEEK_API_KEY'))
Settings.embed_model=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en')

async def main():
    reader=SimpleDirectoryReader(input_files=['data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf'])
    docs=reader.load_data()

    print('加载响应评估数据集')
    # LlamaIndex用于评估的数据集格式，包含多个QueryResponsePair（query response reference_contexts(optimal)）
    response_eval_dataset=QueryResponseDataset.from_json('c6_response_eval_dataset.json')

    # 句子窗口检索
    sentence_parser=SentenceWindowNodeParser.from_defaults(
        window_size=3, # 扩展的句子
        window_metadata_key='window', # 窗口文本在metadata中的保存名称
        original_text_metadata_key='original_text' # 在metadata中保存原始句子的名称
    )
    sentence_nodes=sentence_parser.get_nodes_from_documents(docs)
    sentence_index=VectorStoreIndex(sentence_nodes)

    sentence_query_engine=sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key='window') # 替换node内容
        ]
    )

    # 常规分块检索
    base_parser=SentenceSplitter(chunk_size=512)
    base_nodes=base_parser.get_nodes_from_documents(docs)
    base_index=VectorStoreIndex(base_nodes)

    base_query_engine=base_index.as_query_engine(similarity_top_k=2)
    base_retriever=base_index.as_retriever(similarity_top_k=2)

    # 初始化响应评估器
    faithfulness_evaluator=FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator=RelevancyEvaluator(llm=Settings.llm)

    # 4. 执行响应评估对比
    print("开始执行响应评估对比...")
    evaluators = {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator}
    queries = response_eval_dataset.queries

    # 句子窗口检索响应评估
    print("\n=== 评估句子窗口检索 ===")
    sentence_runner=BatchEvalRunner(evaluators,workers=2,show_progress=True)
    sentence_response_results=await sentence_runner.aevaluate_queries(
        queries=queries,query_engine=sentence_query_engine,
    )

    # 常规分块检索响应评估
    print("\n=== 评估常规分块检索 ===")
    base_runner=BatchEvalRunner(evaluators,workers=2,show_progress=True)
    base_response_results=await base_runner.aevaluate_queries(
        queries=queries,query_engine=base_query_engine,
    )

    # 5. 分析并打印对比结果
    print("\n" + "="*60)
    print("响应评估结果对比")
    print("="*60)

    def calc_response_score(results, metric):
        if results and results.get(metric):
            scores = results[metric]
            return sum(r.passing for r in scores) / len(scores)
        return 0

    # 句子窗口检索结果
    sentence_faith = calc_response_score(sentence_response_results, "faithfulness")
    sentence_rel = calc_response_score(sentence_response_results, "relevancy")

    # 常规分块检索结果
    base_faith = calc_response_score(base_response_results, "faithfulness")
    base_rel = calc_response_score(base_response_results, "relevancy")

    print(f"\n句子窗口检索:")
    print(f"  忠实度: {sentence_faith:.1%}")
    print(f"  相关性: {sentence_rel:.1%}")

    print(f"\n常规分块检索:")
    print(f"  忠实度: {base_faith:.1%}")
    print(f"  相关性: {base_rel:.1%}")

    # 简单对比
    if sentence_faith > base_faith and sentence_rel > base_rel:
        print(f"\n✅ 句子窗口检索在两个维度上都优于常规分块检索")
    elif sentence_faith > base_faith or sentence_rel > base_rel:
        print(f"\n⚖️  句子窗口检索在某些维度上有优势")
    else:
        print(f"\n❌ 句子窗口检索未显示明显优势")

if __name__ == "__main__":
    asyncio.run(main())