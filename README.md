# 🥘 CookingRAG - 多模态智能食谱问答系统

> 基于 **LangChain** + **RAG** + **Qwen-VL** 的垂直领域 AI 助手。支持“识图搜菜”、混合检索与智能多轮对话。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![Moonshot](https://img.shields.io/badge/LLM-Moonshot%20AI-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 项目简介

**CuisineRAG** 是一个解决“今天吃什么”以及“这道菜怎么做”的智能系统。与传统的关键词搜索不同，它利用大模型（LLM）的理解能力和视觉模型（VLM）的感知能力，提供精准的烹饪指导。

**核心痛点解决：**
- 📸 **看到食材不知道做什么？** -> 支持直接上传图片查询。
- 🔍 **搜“红烧肉”出来全是广告？** -> 本地知识库 RAG，精准无广。
- 🧠 **不知道具体菜名？** -> 支持“推荐几个川菜”等模糊语义查询。

## ✨ 核心功能

*   **👁️ 多模态交互 (Multi-modal)**：集成 **Qwen-VL** 本地视觉模型，支持“识图搜菜”，从图片自动识别菜品并生成食谱。
*   **🔍 高级混合检索 (Hybrid Search)**：采用 **Vector (向量)** + **BM25 (关键词)** 双路召回，结合 **RRF** 算法融合排序，并通过 **Cross-Encoder** 进行二次精排，大幅减少幻觉。
*   **🧠 智能意图路由 (Query Routing)**：自动识别用户意图（列表推荐 / 详细做法 / 闲聊），动态切换回答策略。
*   **🔄 多轮对话增强**：具备上下文记忆，支持“它怎么做”之类的指代消解。
*   **📝 结构化输出**：强制模型按照 `介绍 -> 食材 -> 步骤 -> 技巧` 的标准格式输出，实操性强。

## 🛠️ 技术栈

- **框架**: LangChain, PyTorch
- **LLM**: Moonshot AI (Kimi-k2)
- **Embedding**: BAAI/bge-small-zh-v1.5
- **Rerank**: BAAI/bge-reranker-base
- **Vision**: Qwen3-VL-8B-Instruct (Local)
- **Vector DB**: FAISS

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/你的用户名/CuisineRAG.git
cd CuisineRAG
```

### 2. 环境准备
建议使用 Conda 创建虚拟环境：
```bash
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt
```

### 3. 配置环境变量
在项目根目录创建 `.env` 文件，并填入你的 Moonshot (Kimi) API Key：
```text
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. 下载模型
本项目使用本地视觉模型，请从 HuggingFace 或 ModelScope 下载 **Qwen3-VL-8B-Instruct** (或兼容版本)，并将其放在 `models/` 目录下（或者修改 `config.py` 中的路径）。

### 5. 准备数据
将你的食谱文本数据放入 `data/` 目录。

### 6. 运行系统
```bash
python main.py
```
*首次运行时，系统会自动扫描数据并构建向量索引，这可能需要几分钟时间。*

## 💡 使用指南

系统启动后进入交互模式：

**1. 文字提问**
直接输入问题即可：
```text
👤 您的问题: 红烧肉怎么做
🤖 回答: (生成详细的红烧肉制作步骤...)
```

**2. 图片提问**
使用 `image:` 前缀加上图片路径：
```text
👤 您的问题: image:./test_imgs/meat.jpg 这适合老人吃吗？
🤖 回答: 🔍 已识别图片内容为：**东坡肉**... 结合您的问题，东坡肉肥而不腻，但含糖油较高...
```

**3. 管理指令**
- 输入 `new` 或 `clear`：清空对话历史。
- 输入 `exit` 或 `quit`：退出系统。

## ⚠️ 注意事项

1. **显存要求**：运行 Qwen-VL 本地模型建议显存 >= 16GB。如果显存不足，建议在 `local_vision_module.py` 中启用 4-bit 量化加载。
2. **API 额度**：文本生成依赖 Moonshot API，请确保账户有余额。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
