"""
RAG系统主程序
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
    LocalVisionModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.chat_history: List[Tuple[str, str]] = [] 

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("👁️ 正在加载本地视觉模型 (Qwen3-VL)...这可能需要一点时间")
        self.vision_module = LocalVisionModule(
            model_path=self.config.vision_model_path
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks, rerank_model_path=self.config.rerank_model_path, initial_k=self.config.initial_k)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        # === 1. 多轮对话上下文处理 (Contextualization) ===
        # 如果有历史记录，先尝试把 "它怎么做" 改成 "红烧肉怎么做"
        if self.chat_history:
            print("🔄 结合历史上下文分析...")
            standalone_query = self.generation_module.contextualize_query(question, self.chat_history)
            if standalone_query != question:
                print(f"🧩 补全后查询: {standalone_query}")
        else:
            standalone_query = question

        # 1. 查询路由
        route_type = self.generation_module.query_router(standalone_query)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            search_query = standalone_query
        else:
            print("🤖 优化检索关键词...")
            search_query = self.generation_module.query_rewrite(standalone_query)
        
        # 3. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(search_query)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=self.config.top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(search_query, top_k=self.config.top_k)

        # 显示检索到的子块信息
        if relevant_chunks: 
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:50].replace('\n', ' ').strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题
                    title_end = content_preview.find('\n') if '\n' in chunk.page_content[:100] else len(content_preview)
                    section_title = chunk.page_content[:title_end].strip('#').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            full_response = self.generation_module.generate_list_answer(standalone_query, relevant_docs)
            yield full_response
            # 记录历史
            self.chat_history.append((question, full_response))
            return
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

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

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                response_generator = self.generation_module.generate_step_by_step_answer_stream(
                    standalone_query, relevant_docs, history=self.chat_history # 传入历史
                )
            else:
                response_generator = self.generation_module.generate_basic_answer_stream(
                    standalone_query, relevant_docs, history=self.chat_history # 传入历史
                )

            # === 6. 收集流式输出并保存历史 ===
            full_response_buffer = ""
            for chunk in response_generator:
                full_response_buffer += chunk
                yield chunk
            
            # 循环结束后，将完整对话存入历史
            self.chat_history.append((question, full_response_buffer))

    def ask_with_image(self, image_path: str, user_question: str = ""):
        """
        处理图片+文字的混合查询
        
        Args:
            image_path: 图片路径
            user_question: 用户关于图片的具体问题（可选）
        """
        print(f"\n📸 正在本地分析图片: {image_path}")
        
        if not Path(image_path).exists():
            yield "错误：找不到图片文件。"
            return

        # 1. 视觉识别：获取菜品名称
        try:
            dish_name = self.vision_module.identify_dish(image_path)
            print(f"👀 识别结果: **{dish_name}**")
            yield f"🔍 已识别图片内容为：**{dish_name}**\n\n"
        except Exception as e:
            logger.error(f"识别出错: {e}")
            yield "抱歉，图片识别服务暂时不可用。"
            return

        # 2. 如果用户没有提问，默认生成基础介绍或做法
        if not user_question:
            # 默认行为：查询做法
            rag_query = f"{dish_name}的做法"
            print(f"🔄 用户未提问，默认查询: '{rag_query}'")
            yield "正在为您查找该菜品的制作方法...\n"
        else:
            # 3. 语义融合：将菜名注入到用户的问题中
            # 例如：用户问 "适合老人吃吗" -> 转换为 "红烧肉适合老人吃吗"
            # 这样 RAG 检索模块才能正确找到红烧肉的文档
            rag_query = f"{dish_name} {user_question}"
            print(f"🔄 结合图片信息，构建复合查询: '{rag_query}'")

        # 4. 调用标准的 RAG 流程进行检索和回答
        # 这里使用 yield from 将生成器的内容透传出去
        yield from self.ask_question(rag_query, stream=True)
    
    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)
        
        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
    
    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

        return answer
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        print("\n使用说明:")
        print("1. 文字提问: 直接输入问题 (例如: 红烧肉怎么做)")
        print("2. 图片提问: 输入 'image:图片路径 [你的问题]'")
        print("3. 输入 'clear' 或 'new' 清空对话历史") # 新增说明
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n👤 您的问题: ").strip()
                
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 新增：清空历史命令
                if user_input.lower() in ['clear', 'new', 'reset']:
                    self.clear_history()
                    continue
                
                if not user_input:
                    continue

                print("\n🤖 回答:")
                
                # --- 判断模式 ---
                if user_input.startswith("image:"):
                    # === 图片模式 ===
                    # 解析输入：去掉 'image:' 前缀
                    content = user_input[6:].strip()
                    
                    # 分离图片路径和问题文本
                    # 假设路径和问题之间用空格分隔
                    parts = content.split(' ', 1)
                    image_path = parts[0]
                    user_question = parts[1] if len(parts) > 1 else ""
                    
                    # 调用图片问答逻辑
                    for chunk in self.ask_with_image(image_path, user_question):
                        print(chunk, end="", flush=True)
                        
                else:
                    # === 纯文字模式 ===
                    # 默认使用流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                
                print("\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"处理过程中发生错误: {e}")
                print(f"\n❌ 处理问题时出错: {e}")
        
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
