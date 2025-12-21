"""
基于 DeepSeek 模型和 You.com 或 Bing 搜索引擎的 STORM Wiki 流程。
运行此脚本需要设置以下环境变量：
    - DEEPSEEK_API_KEY: DeepSeek API 密钥
    - DEEPSEEK_API_BASE: DeepSeek API 基础 URL（默认为 https://api.deepseek.com）
    - YDC_API_KEY: You.com API 密钥；BING_SEARCH_API_KEY: Bing 搜索 API 密钥，SERPER_API_KEY: Serper API 密钥，BRAVE_API_KEY: Brave API 密钥，或 TAVILY_API_KEY: Tavily API 密钥

输出结构如下：
args.output_dir/
    topic_name/  # 主题名称遵循下划线连接的命名规范，不包含空格和斜杠
        conversation_log.json           # 信息搜集对话日志
        raw_search_results.json         # 搜索引擎的原始搜索结果
        direct_gen_outline.txt          # 直接使用 LLM 参数化知识生成的大纲
        storm_gen_outline.txt           # 使用收集的信息优化后的大纲
        url_to_info.json                # 最终文章中使用的信息源
        storm_gen_article.txt           # 生成的最终文章
        storm_gen_article_polished.txt  # 润色后的最终文章（如果 args.do_polish_article 为 True）
"""

import os
import re
import logging
from argparse import ArgumentParser

from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import DeepSeekModel
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    BraveRM,
    SerperRM,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    SearXNG,
)
from knowledge_storm.utils import load_api_key


def sanitize_topic(topic):
    """
    清理主题名称以便用于文件名。
    删除或替换文件名中不允许的字符。
    """
    # 将空格替换为下划线
    topic = topic.replace(" ", "_")

    # 删除任何非字母数字、下划线或连字符的字符
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)

    # 确保清理后主题名称不为空
    if not topic:
        topic = "unnamed_topic"

    return topic


def main(args):
    # 从 secrets.toml 文件加载 API 密钥
    # 检查 secrets.toml 文件是否存在
    toml_file_path = "secrets.toml"
    if not os.path.exists(toml_file_path):
        # 如果当前目录没有找到，则尝试在项目根目录查找
        toml_file_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "secrets.toml"
        )
        toml_file_path = os.path.abspath(toml_file_path)

    # 加载API密钥到环境变量中
    load_api_key(toml_file_path=toml_file_path)
    lm_configs = STORMWikiLMConfigs()

    logger = logging.getLogger(__name__)

    # 确保 DEEPSEEK_API_KEY 已设置
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError(
            "未设置 DEEPSEEK_API_KEY 环境变量。请在 secrets.toml 文件中设置它。"
        )

    # 配置 DeepSeek 模型的通用参数
    deepseek_kwargs = {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "api_base": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # 对话模拟器语言模型：用于模拟专家与用户的对话
    conv_simulator_lm = DeepSeekModel(
        model=args.model, max_tokens=500, **deepseek_kwargs
    )

    # 问题生成语言模型：用于生成信息搜集问题
    question_asker_lm = DeepSeekModel(
        model=args.model, max_tokens=500, **deepseek_kwargs
    )

    # 大纲生成语言模型：用于生成文章大纲
    outline_gen_lm = DeepSeekModel(model=args.model, max_tokens=400, **deepseek_kwargs)

    # 文章生成语言模型：用于生成文章内容
    article_gen_lm = DeepSeekModel(model=args.model, max_tokens=700, **deepseek_kwargs)

    # 文章润色语言模型：用于润色和完善文章
    article_polish_lm = DeepSeekModel(
        model=args.model, max_tokens=4000, **deepseek_kwargs
    )

    # 配置 STORM 流程中各个阶段使用的语言模型
    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    # 配置 STORM Wiki 运行器的参数
    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,  # 输出路径
        max_conv_turn=args.max_conv_turn,  # 最大对话轮数（深度）
        max_perspective=args.max_perspective,  # 最大视角数 （宽度）
        search_top_k=args.search_top_k,  # 搜索结果数量   （搜索页面数量）
        max_thread_num=args.max_thread_num,  # 最大线程数 （并行展开数量）
    )

    # 根据用户选择的检索器类型初始化相应的检索模块
    rm = None
    match args.retriever:
        case "bing":
            rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "you":
            rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(
                k=engine_args.search_top_k, safe_search="On", region="us-en"
            )
        case "serper":
            rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=engine_args.search_top_k,
                include_raw_content=True,
            )
        case "searxng":
            rm = SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"), k=engine_args.search_top_k
            )
        case _:
            raise ValueError(
                f'无效的检索器: {args.retriever}。请选择 "bing"、"you"、"brave"、"duckduckgo"、"serper"、"tavily" 或 "searxng"'
            )

    # 初始化 STORM Wiki 运行器
    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    # 获取用户输入的主题
    topic = input("主题: ")
    # 清理主题名称，确保可以用作文件名
    sanitized_topic = sanitize_topic(topic)

    try:
        # 运行 STORM 流程，包括研究、大纲生成、文章生成和润色等阶段
        runner.run(
            topic=sanitized_topic,
            do_research=args.do_research,  # 模拟对话来研究主题
            do_generate_outline=args.do_generate_outline,  # 为主题生成大纲
            do_generate_article=args.do_generate_article,  # 为主题生成文章
            do_polish_article=args.do_polish_article,  # 润色文章
            remove_duplicate=args.remove_duplicate,  # 从文章中删除重复内容
        )
        # 运行后处理
        runner.post_run()
        # 输出运行摘要
        runner.summary()
    except Exception as e:
        logger.exception(f"发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    # 全局参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/deepseek",
        help="存储输出结果的目录。",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="使用的最大线程数。信息搜集和文章生成部分可以通过多线程加速。"
        "如果频繁遇到调用 LM API 时的'超出速率限制'错误，请考虑减少此值。",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "duckduckgo", "tavily"],
        default="tavily",  # 添加默认值
        help="用于检索信息的搜索引擎 API。",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepseek-chat"],
        default="deepseek-chat",
        help='使用的 DeepSeek 模型。"deepseek-chat" 用于常规任务',
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度参数。")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 采样参数。")
    # 流程阶段控制参数
    parser.add_argument(
        "--do-research",
        action="store_true",
        default=True,  # 添加默认值
        help="如果为 True，模拟对话来研究主题；否则，加载已有结果。",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        default=True,  # 添加默认值
        help="如果为 True，为主题生成大纲；否则，加载已有结果。",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        default=True,  # 添加默认值
        help="如果为 True，为主题生成文章；否则，加载已有结果。",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        default=True,  # 添加默认值
        help="如果为 True，通过添加摘要部分和（可选的）删除重复内容来润色文章。",
    )
    # 预写作阶段的超参数
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="对话式提问中的最大问题数量。",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="视角引导式提问中要考虑的最大视角数量。",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="每个搜索查询要考虑的前 k 个搜索结果。",
    )
    # 写作阶段的超参数
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=3,
        help="每个章节标题收集的前 k 个参考文献。",
    )
    parser.add_argument(
        "--remove-duplicate",
        action="store_true",
        help="如果为 True，从文章中删除重复内容。",
    )

    main(parser.parse_args())
