import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document], rerank_model_path: str = None, initial_k: int = 15):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.initial_k = initial_k
        self.reranker = None
        
        # [新增] 加载重排序模型
        if rerank_model_path:
            self.setup_reranker(rerank_model_path)

        self.setup_retrievers()

    # [新增] 初始化重排序模型的方法
    def setup_reranker(self, model_path: str):
        """加载 Cross-Encoder 重排序模型"""
        logger.info(f"正在加载重排序模型: {model_path}")
        try:
            # device='cuda' 如果你有显卡，否则会自动切 CPU
            self.reranker = CrossEncoder(model_path, device=None, max_length=512)
            logger.info("重排序模型加载完成")
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {e}，将降级为不使用重排序")
            self.reranker = None

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.initial_k}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.initial_k
        )

        logger.info("检索器设置完成")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        # 使用RRF重排
        fused_docs = self._rrf_rerank(vector_docs, bm25_docs)

        # 3. Cross-Encoder 精排 (Re-rank)
        if self.reranker:
            logger.info(f"正在对前 {len(fused_docs)} 个文档进行重排序...")
            final_docs = self._cross_encoder_rerank(query, fused_docs, top_k)
            return final_docs
        else:
            # 如果没加载重排序模型，直接截取 RRF 的结果
            return fused_docs[:top_k]

    # [新增] 重排序的具体实现
    def _cross_encoder_rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """使用 Cross-Encoder 对文档进行打分重排"""
        if not docs:
            return []

        # 构造模型输入对: [[Query, Doc1], [Query, Doc2], ...]
        pairs = [[query, doc.page_content] for doc in docs]
        
        # 模型推理打分
        scores = self.reranker.predict(pairs)
        
        # 将分数绑定回文档
        doc_score_pairs = []
        for doc, score in zip(docs, scores):
            # 将 numpy.float32 转换为 float
            doc.metadata['rerank_score'] = float(score)
            doc_score_pairs.append((doc, score))
            logger.debug(f"重排序打分: {score:.4f} | 内容: {doc.page_content[:30]}...")

        # 按分数降序排列
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 取 Top K
        reranked_docs = [pair[0] for pair in doc_score_pairs[:top_k]]
        
        logger.info(f"重排序完成，最高分: {doc_score_pairs[0][1]:.4f}")
        return reranked_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 3) -> List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        # 先进行混合检索，获取更多候选
        docs = self.hybrid_search(query, top_k * 3)
        
        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        
        return filtered_docs