# CLAUDE.md（中文版）

本文件为 Claude Code (claude.ai/code) 在此代码库中工作提供指导。

## 项目概述

**STORM** 是一个基于大语言模型的自动化知识管理系统，能够从零开始生成维基百科风格的百科文章。它包含两种模式：
- **STORM**：通过检索和多视角提问实现自动化文章生成
- **Co-STORM**：人机协作式知识整理系统

项目已发布为 PyPI 包 `knowledge-storm`（版本 1.1.1）。

## 常用命令

### 安装依赖

```bash
# 创建虚拟环境
conda create -n storm python=3.11
conda activate storm

# 安装依赖
pip install -r requirements.txt

# 或使用 uv（推荐，更快的安装）
uv pip install -r requirements.txt
```

### 开发环境设置

```bash
# 安装 pre-commit 钩子
pip install pre-commit
pre-commit install

# 格式化代码（提交前必须执行）
black knowledge_storm/
```

### 运行示例

**STORM 示例：**
```bash
# 使用 GPT 运行 STORM
python examples/storm_examples/run_storm_wiki_gpt.py \
    --output-dir ./output \
    --retriever bing \
    --do-research \
    --do-generate-outline \
    --do-generate-article \
    --do-polish-article

# 使用 Claude 运行 STORM
python examples/storm_examples/run_storm_wiki_claude.py \
    --output-dir ./output \
    --retriever bing

# 使用 VectorRM 运行 STORM
python examples/storm_examples/run_storm_wiki_gpt_with_VectorRM.py \
    --output-dir ./output
```

**Co-STORM 示例：**
```bash
python examples/costorm_examples/run_costorm_gpt.py \
    --output-dir ./output \
    --retriever bing
```

### Streamlit 演示界面

```bash
cd frontend/demo_light
pip install -r requirements.txt
streamlit run storm.py
```

### 测试

**注意：** 此项目目前缺乏全面的测试覆盖。没有专门的测试文件或测试目录。添加测试时：
- 使用 `pytest` 测试框架
- 创建符合 Python 惯例的 `tests/` 目录
- 为 `knowledge_storm/storm_wiki/modules/` 中的核心模块添加测试

## 架构

### 核心组件

```
knowledge_storm/
├── storm_wiki/              # STORM 引擎
│   ├── engine.py            # 主 STORM 编排（17,988 行）
│   └── modules/             # 核心模块
│       ├── article_generation.py      # 文章生成
│       ├── article_polish.py          # 文章润色
│       ├── knowledge_curation.py      # 知识整理
│       ├── outline_generation.py      # 大纲生成
│       ├── persona_generator.py       # 角色生成器
│       ├── retriever.py               # 检索器
│       └── storm_dataclass.py         # 数据类
├── collaborative_storm/     # Co-STORM 引擎
│   ├── engine.py            # 主 Co-STORM 编排（32,330 行）
│   └── modules/             # Co-STORM 特定模块
│       ├── co_storm_agents.py         # 协作智能体
│       ├── grounded_question_answering.py  # 基于证据的问答
│       └── warmstart_hierarchical_chat.py  # 预热分层聊天
├── interface.py             # 统一接口（21,021 行）
├── lm.py                    # 语言模型接口（42,725 行）
├── rm.py                    # 检索模型接口（47,638 行）
├── encoder.py               # 文本编码工具
├── utils.py                 # 辅助函数
├── dataclass.py             # 数据结构
└── logging_wrapper.py       # 日志配置
```

### STORM 工作流程（4 个阶段）

1. **知识整理（Knowledge Curation）** - 通过模拟对话收集信息
2. **大纲生成（Outline Generation）** - 创建结构化文章大纲
3. **文章生成（Article Generation）** - 基于大纲和参考资料生成完整文章
4. **文章润色（Article Polishing）** - 优化和改进文章质量

### Co-STORM 工作流程（3 个阶段）

1. **预热启动（Warm Start）** - 建立人类与 AI 的共享概念空间
2. **协作对话（Collaborative Discourse）** - 带轮次管理的多智能体对话
3. **动态思维导图（Dynamic Mind Map）** - 维护不断演进的概念图谱

### 语言模型支持

**通过 LiteLLM（`lm.py`）：**
- OpenAI（GPT-4、GPT-3.5）
- Anthropic（Claude）
- Azure OpenAI
- Mistral（通过 VLLM）
- 其他 100+ 模型

在 `secrets.toml` 中配置模型：
```toml
OPENAI_API_KEY="your_key"
ANTHROPIC_API_KEY="your_key"
```

### 检索系统支持

**多种搜索后端（`rm.py`）：**
- You.com（YouRM）
- Bing 搜索
- Brave 搜索
- Serper
- DuckDuckGo
- Tavily 搜索
- SearXNG
- Azure AI 搜索
- VectorRM（基于 Qdrant 的自定义向量数据库）

### 核心依赖

- **dspy**（v2.4.9）- 可编程 LLM 框架
- **litellm** - 统一 100+ LLM API 接口
- **langchain** - LLM 应用开发框架
- **qdrant-client** - 高性能向量数据库
- **streamlit** - Web UI 框架
- **sentence-transformers** - 文本嵌入

## 配置文件

### `pyproject.toml`
- 现代 Python 项目配置
- 需要 Python 3.11+
- 固定依赖版本以确保稳定性

### `requirements.txt`
- 12 个核心依赖，版本锁定
- 确保环境可重现

### `.pre-commit-config.yaml`
- Black 代码格式化工具
- 仅格式化 `knowledge_storm/` 目录
- 提交时自动运行

### `secrets.toml`（需自行创建）
存储 API 密钥和敏感信息（git 已忽略）：
```toml
OPENAI_API_KEY="sk-..."
BING_SEARCH_API_KEY="..."
ANTHROPIC_API_KEY="..."
```

## 重要文件

### 文档
- **README.md**（372 行）- 全面的项目文档、安装、API 使用说明
- **CONTRIBUTING.md**（48 行）- 贡献指南、开发设置
- **frontend/demo_light/README.md** - Streamlit 演示说明

### 示例
- `examples/storm_examples/` - 不同模型的 STORM 使用示例
- `examples/costorm_examples/` - Co-STORM 协作示例

### 前端
- `frontend/demo_light/storm.py` - 基于 Streamlit 的交互式演示
- 实时可视化生成过程

## 开发指南

### 代码风格
- **使用 Black 格式化**（pre-commit 钩子强制执行）
- 遵循 `knowledge_storm/` 模块中的现有模式
- 保持模块专注于单一职责

### 添加功能
1. 遵循 `storm_wiki/modules/` 或 `collaborative_storm/modules/` 中的模块化架构
2. 为新的抽象基类扩展 `interface.py`
3. 在 `examples/` 目录中添加示例脚本
4. 更新 `README.md` 记录新功能

### 已知问题
- **大文件**：一些核心文件超过 1000 行（如 `lm.py`、`rm.py`、`engine.py`）
- **无测试**：项目缺乏全面的测试覆盖
- **无 CI/CD**：没有 GitHub Actions 或类似的 CI 配置

### 扩展检索系统
要添加新的检索系统，扩展 `knowledge_storm/rm.py`：
- 实现 BaseRM 抽象类
- 在 `secrets.toml` 中添加配置
- 在示例中使用新的 `--retriever` 选项

### 扩展语言模型
要添加新的 LLM 提供商，扩展 `knowledge_storm/lm.py`：
- 实现 BaseLM 抽象类
- 在 `secrets.toml` 中配置
- 使用 LiteLLM 以保持标准 API 兼容性

## 数据集资源

项目包含数据集引用：
- **FreshWiki** - 用于评估的干净维基百科文章
- **WildSeek** - 网络抓取的知识库

参考 `README.md` 获取数据集访问和使用方法。

## PyPI 包

项目已作为 `knowledge-storm` 发布到 PyPI：
```bash
pip install knowledge-storm
```

这将安装包以便在其他项目中直接导入和使用。

## 重要说明

1. **Python 版本**：需要 Python 3.10+（推荐 3.11）
2. **API 密钥**：所有提供商都需要在 `secrets.toml` 中配置有效的 API 密钥
3. **速率限制**：批量操作时注意 API 速率限制
4. **输出目录**：始终为示例脚本指定 `--output-dir`
5. **内存使用**：大语言模型需要大量内存（推荐 8GB+）

## 中文用户特别说明

### 快速开始建议

1. **API 密钥获取**：
   - OpenAI：访问 https://platform.openai.com/api-keys
   - Anthropic：访问 https://console.anthropic.com/
   - Bing：访问 https://www.microsoft.com/en-us/bing/apis/bing-web-search-api

2. **推荐配置**：
   - 使用 Claude-3 或 GPT-4 获得最佳效果
   - Bing 搜索作为默认检索器（免费层足够试用）

3. **常见问题**：
   - 如果遇到 API 密钥错误，检查 `secrets.toml` 文件路径和格式
   - 如果生成速度慢，考虑使用更小的模型（如 GPT-3.5）
   - 如果内存不足，减少并发请求或使用更小的模型

4. **性能优化**：
   - 使用 `uv` 而不是 `pip` 可以显著加快依赖安装
   - 启用 `pre-commit` 可以确保代码质量
   - 定期运行 `black` 格式化代码

### 社区资源

- **项目主页**：https://github.com/stanford-oval/storm
- **论文**：在 `README.md` 中找到相关学术引用
- **问题反馈**：通过 GitHub Issues 提交问题

### 中文相关

虽然项目界面主要为英文，但您可以：
- 修改 `frontend/demo_light/storm.py` 添加中文界面支持
- 在 `examples/` 中创建中文提示词的示例
- 贡献中文翻译文档