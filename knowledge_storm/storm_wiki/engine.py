import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional

import dspy

from .modules.article_generation import StormArticleGenerationModule
from .modules.article_polish import StormArticlePolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.outline_generation import StormOutlineGenerationModule
from .modules.persona_generator import StormPersonaGenerator
from .modules.storm_dataclass import StormInformationTable, StormArticle
from ..interface import Engine, LMConfigs, Retriever
from ..lm import LitellmModel
from ..utils import FileIOHelper, makeStringRed, truncate_filename

"""
STORM Wiki 引擎核心模块。

该模块实现了 STORM（Simulated Theory of mind Reasoning for Open-domain Multi-turn
Research）框架的主要编排逻辑，负责管理整个知识生成流程的四个核心阶段：

1. 知识策展 (Knowledge Curation): 通过多角度对话式提问收集相关信息
2. 大纲生成 (Outline Generation): 基于收集的信息生成结构化文章大纲
3. 文章生成 (Article Generation): 根据大纲和参考资料生成完整文章
4. 文章抛光 (Article Polishing): 优化文章质量，提升可读性和准确性

主要组件：
- STORMWikiLMConfigs: 统一管理不同阶段所需的大语言模型配置
- STORMWikiRunnerArguments: 控制 STORM 流程运行参数的数据类
- STORMWikiRunner: 执行完整 STORM 工作流程的主编排器

该引擎支持多种 LLM 提供商（OpenAI、Azure OpenAI 等），并通过模块化设计
实现了各阶段之间的松耦合，便于扩展和维护。
"""


class STORMWikiLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of STORM.

    Given that different parts in STORM framework have different complexity, we use different LLM configurations
    to achieve a balance between quality and efficiency. If no specific configuration is provided, we use the default
    setup in the paper.
    """

    def __init__(self):
        self.conv_simulator_lm = (
            None  # LLM used in conversation simulator except for question asking.
        )
        self.question_asker_lm = None  # LLM used in question asking.
        self.outline_gen_lm = None  # LLM used in outline generation.
        self.article_gen_lm = None  # LLM used in article generation.
        self.article_polish_lm = None  # LLM used in article polishing.

    def init_openai_model(
        self,
        openai_api_key: str,
        azure_api_key: str,
        openai_type: Literal["openai", "azure"],
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """Legacy: Corresponding to the original setup in the NAACL'24 paper."""
        azure_kwargs = {
            "api_key": azure_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": api_base,
            "api_version": api_version,
        }

        openai_kwargs = {
            "api_key": openai_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": None,
        }
        if openai_type and openai_type == "openai":
            self.conv_simulator_lm = LitellmModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = LitellmModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            # 1/12/2024: Update gpt-4 to gpt-4-1106-preview. (Currently keep the original setup when using azure.)
            self.outline_gen_lm = LitellmModel(
                model="gpt-4-0125-preview", max_tokens=400, **openai_kwargs
            )
            self.article_gen_lm = LitellmModel(
                model="gpt-4o-2024-05-13", max_tokens=700, **openai_kwargs
            )
            self.article_polish_lm = LitellmModel(
                model="gpt-4o-2024-05-13", max_tokens=4000, **openai_kwargs
            )
        elif openai_type and openai_type == "azure":
            self.conv_simulator_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=500,
                **azure_kwargs,
                model_type="chat",
            )
            # use combination of openai and azure-openai as azure-openai does not support gpt-4 in standard deployment
            self.outline_gen_lm = LitellmModel(
                model="azure/gpt-4o", max_tokens=400, **azure_kwargs, model_type="chat"
            )
            self.article_gen_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=700,
                **azure_kwargs,
                model_type="chat",
            )
            self.article_polish_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=4000,
                **azure_kwargs,
                model_type="chat",
            )
        else:
            logging.warning(
                "No valid OpenAI API provider is provided. Cannot use default LLM configurations."
            )

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model


@dataclass
class STORMWikiRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""

    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={
            "help": "Maximum number of questions in conversational question asking."
        },
    )
    max_perspective: int = field(
        default=3,
        metadata={
            "help": "Maximum number of perspectives to consider in perspective-guided question asking."
        },
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximum number of threads to use. "
            "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."
        },
    )


class STORMWikiRunner(Engine):
    """STORM Wiki 流程执行器。

    负责编排和执行完整的 STORM 知识生成流程，包括四个核心阶段：
    1. 知识策展：通过多角度对话收集信息
    2. 大纲生成：基于收集的信息生成文章结构
    3. 文章生成：根据大纲和参考资料生成完整文章
    4. 文章抛光：优化文章质量和可读性
    """

    def __init__(
        self, args: STORMWikiRunnerArguments, lm_configs: STORMWikiLMConfigs, rm
    ):
        """初始化 STORM Wiki 执行器。

        Args:
            args: STORM 运行参数配置
            lm_configs: 大语言模型配置
            rm: 检索模型实例
        """
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        # 初始化检索器，用于从搜索引擎或知识库检索信息
        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)

        # 初始化人物角色生成器，用于生成不同视角的提问者
        storm_persona_generator = StormPersonaGenerator(
            self.lm_configs.question_asker_lm
        )

        # 初始化知识策展模块：通过模拟多角色对话收集主题相关信息
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
        )

        # 初始化大纲生成模块：基于收集的信息生成结构化文章大纲
        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )

        # 初始化文章生成模块：根据大纲和参考资料生成完整文章内容
        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
        )

        # 初始化文章抛光模块：优化文章质量，添加摘要并去除重复内容
        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm,
        )

        # 检查语言模型配置是否完整
        self.lm_configs.init_check()
        # 应用装饰器（如日志记录、性能监控等）
        self.apply_decorators()

    def run_knowledge_curation_module(
        self,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None,
    ) -> StormInformationTable:
        """执行知识策展模块。

        通过模拟多角色对话进行主题研究，从不同视角提问并收集信息。
        这是 STORM 流程的第一阶段，为后续大纲生成和文章撰写奠定基础。

        Args:
            ground_truth_url: 参考文章的 URL，该 URL 将被排除在搜索结果之外
            callback_handler: 回调处理器，用于处理中间结果

        Returns:
            StormInformationTable: 包含收集到的信息和引用的信息表
        """
        # 核心流程：通过多角色对话收集主题相关信息
        (
            information_table,  # 信息表：存储收集到的所有信息和引用
            conversation_log,  # 对话日志：记录完整的对话过程
        ) = self.storm_knowledge_curation_module.research(
            topic=self.topic,
            ground_truth_url=ground_truth_url,
            callback_handler=callback_handler,
            max_perspective=self.args.max_perspective,  # 最大视角数
            disable_perspective=False,
            return_conversation_log=True,
        )

        # 保存对话日志到本地文件，便于后续分析和调试
        FileIOHelper.dump_json(
            conversation_log,
            os.path.join(self.article_output_dir, "conversation_log.json"),
        )
        # 保存原始搜索结果，包含 URL 和对应的信息
        information_table.dump_url_to_info(
            os.path.join(self.article_output_dir, "raw_search_results.json")
        )
        return information_table

    def run_outline_generation_module(
        self,
        information_table: StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """执行大纲生成模块。

        基于知识策展阶段收集的信息，生成结构化的文章大纲。
        这是 STORM 流程的第二阶段，为文章生成提供清晰的结构框架。

        Args:
            information_table: 知识策展阶段收集的信息表
            callback_handler: 回调处理器，用于处理中间结果

        Returns:
            StormArticle: 包含大纲的文章对象
        """
        # 核心流程：生成 STORM 优化的大纲和直接生成的大纲
        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,  # 同时返回草稿大纲用于对比
            callback_handler=callback_handler,
        )

        # 保存 STORM 生成的优化大纲（使用对话信息优化）
        outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "storm_gen_outline.txt")
        )
        # 保存直接生成的草稿大纲（不使用对话信息）
        draft_outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "direct_gen_outline.txt")
        )
        return outline

    def run_article_generation_module(
        self,
        outline: StormArticle,
        information_table=StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """执行文章生成模块。

        根据大纲和收集的参考资料，生成完整的文章内容。
        这是 STORM 流程的第三阶段，将结构化大纲填充为具体内容。

        Args:
            outline: 文章大纲
            information_table: 知识策展阶段收集的信息表
            callback_handler: 回调处理器，用于处理中间结果

        Returns:
            StormArticle: 生成的文章草稿
        """
        # 核心流程：基于大纲和信息表生成文章内容
        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,  # 使用大纲作为结构指导
            callback_handler=callback_handler,
        )

        # 保存生成的文章草稿（纯文本格式）
        draft_article.dump_article_as_plain_text(
            os.path.join(self.article_output_dir, "storm_gen_article.txt")
        )
        # 保存文章引用的参考资料信息
        draft_article.dump_reference_to_file(
            os.path.join(self.article_output_dir, "url_to_info.json")
        )
        return draft_article

    def run_article_polishing_module(
        self, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """执行文章抛光模块。

        优化文章质量，包括添加摘要、改善可读性，可选择性地去除重复内容。
        这是 STORM 流程的第四阶段，也是最后的优化阶段。

        Args:
            draft_article: 待抛光的文章草稿
            remove_duplicate: 是否去除重复内容

        Returns:
            StormArticle: 抛光后的最终文章
        """
        # 核心流程：对文章进行抛光优化
        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate,  # 可选：去除重复内容
        )

        # 保存抛光后的最终文章
        FileIOHelper.write_str(
            polished_article.to_string(),
            os.path.join(self.article_output_dir, "storm_gen_article_polished.txt"),
        )
        return polished_article

    def post_run(self):
        """执行后处理操作。

        在完成主要流程后执行的清理和记录工作，包括：
        1. 保存运行配置信息
        2. 保存大语言模型调用历史记录
        """
        # 记录本次运行的配置信息
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(
            config_log, os.path.join(self.article_output_dir, "run_config.json")
        )

        # 收集并保存所有大语言模型的调用历史
        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        with open(
            os.path.join(self.article_output_dir, "llm_call_history.jsonl"), "w"
        ) as f:
            for call in llm_call_history:
                if "kwargs" in call:
                    call.pop("kwargs")  # 所有 kwargs 已统一保存到 run_config.json 中
                f.write(json.dumps(call) + "\n")

    def _load_information_table_from_local_fs(self, information_table_local_path):
        """从本地文件系统加载信息表。

        Args:
            information_table_local_path: 信息表文件的本地路径

        Returns:
            StormInformationTable: 加载的信息表对象
        """
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} not exists. Please set --do-research argument to prepare the conversation_log.json for this topic."
        )
        return StormInformationTable.from_conversation_log_file(
            information_table_local_path
        )

    def _load_outline_from_local_fs(self, topic, outline_local_path):
        """从本地文件系统加载大纲。

        Args:
            topic: 文章主题
            outline_local_path: 大纲文件的本地路径

        Returns:
            StormArticle: 包含大纲的文章对象
        """
        assert os.path.exists(outline_local_path), makeStringRed(
            f"{outline_local_path} not exists. Please set --do-generate-outline argument to prepare the storm_gen_outline.txt for this topic."
        )
        return StormArticle.from_outline_file(topic=topic, file_path=outline_local_path)

    def _load_draft_article_from_local_fs(
        self, topic, draft_article_path, url_to_info_path
    ):
        """从本地文件系统加载文章草稿。

        Args:
            topic: 文章主题
            draft_article_path: 草稿文章文件路径
            url_to_info_path: 引用信息文件路径

        Returns:
            StormArticle: 加载的文章草稿对象
        """
        assert os.path.exists(draft_article_path), makeStringRed(
            f"{draft_article_path} not exists. Please set --do-generate-article argument to prepare the storm_gen_article.txt for this topic."
        )
        assert os.path.exists(url_to_info_path), makeStringRed(
            f"{url_to_info_path} not exists. Please set --do-generate-article argument to prepare the url_to_info.json for this topic."
        )
        # 加载文章文本内容
        article_text = FileIOHelper.load_str(draft_article_path)
        # 加载引用的参考资料信息
        references = FileIOHelper.load_json(url_to_info_path)
        return StormArticle.from_string(
            topic_name=topic, article_text=article_text, references=references
        )

    def run(
        self,
        topic: str,
        ground_truth_url: str = "",
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = False,
        callback_handler: BaseCallbackHandler = BaseCallbackHandler(),
    ):
        """运行完整的 STORM 流程。

        这是主入口函数，按顺序执行 STORM 的四个核心阶段。
        支持灵活控制执行哪些阶段，可以从中间阶段恢复执行。

        Args:
            topic: 要研究的主题
            ground_truth_url: 参考文章的 URL，该 URL 将被排除在搜索结果之外
            do_research: 是否执行知识策展阶段；如果为 False，需要输出目录中已存在 conversation_log.json 和 raw_search_results.json
            do_generate_outline: 是否执行大纲生成阶段；如果为 False，需要输出目录中已存在 storm_gen_outline.txt
            do_generate_article: 是否执行文章生成阶段；如果为 False，需要输出目录中已存在 storm_gen_article.txt
            do_polish_article: 是否执行文章抛光阶段，包括添加摘要和（可选）去除重复内容
            remove_duplicate: 是否在抛光阶段去除重复内容
            callback_handler: 回调处理器，用于处理各阶段的中间结果
        """
        # 确保至少指定了一个执行阶段
        assert (
            do_research
            or do_generate_outline
            or do_generate_article
            or do_polish_article
        ), makeStringRed(
            "No action is specified. Please set at least one of --do-research, --do-generate-outline, --do-generate-article, --do-polish-article"
        )

        # 初始化主题和输出目录
        self.topic = topic
        # 将主题转换为合法的目录名（替换空格和斜杠）
        self.article_dir_name = truncate_filename(
            topic.replace(" ", "_").replace("/", "_")
        )
        # 创建文章输出目录
        self.article_output_dir = os.path.join(
            self.args.output_dir, self.article_dir_name
        )
        os.makedirs(self.article_output_dir, exist_ok=True)

        # ========== 阶段 1: 知识策展 ==========
        # 通过多角色对话收集主题相关信息
        information_table: StormInformationTable = None
        if do_research:
            information_table = self.run_knowledge_curation_module(
                ground_truth_url=ground_truth_url, callback_handler=callback_handler
            )

        # ========== 阶段 2: 大纲生成 ==========
        # 基于收集的信息生成结构化文章大纲
        outline: StormArticle = None
        if do_generate_outline:
            # 如果没有执行知识策展，从本地加载信息表
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            outline = self.run_outline_generation_module(
                information_table=information_table, callback_handler=callback_handler
            )

        # ========== 阶段 3: 文章生成 ==========
        # 根据大纲和参考资料生成完整文章内容
        draft_article: StormArticle = None
        if do_generate_article:
            # 如果没有执行知识策展，从本地加载信息表
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            # 如果没有执行大纲生成，从本地加载大纲
            if outline is None:
                outline = self._load_outline_from_local_fs(
                    topic=topic,
                    outline_local_path=os.path.join(
                        self.article_output_dir, "storm_gen_outline.txt"
                    ),
                )
            draft_article = self.run_article_generation_module(
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler,
            )

        # ========== 阶段 4: 文章抛光 ==========
        # 优化文章质量，添加摘要并去除重复内容
        if do_polish_article:
            # 如果没有执行文章生成，从本地加载文章草稿
            if draft_article is None:
                draft_article_path = os.path.join(
                    self.article_output_dir, "storm_gen_article.txt"
                )
                url_to_info_path = os.path.join(
                    self.article_output_dir, "url_to_info.json"
                )
                draft_article = self._load_draft_article_from_local_fs(
                    topic=topic,
                    draft_article_path=draft_article_path,
                    url_to_info_path=url_to_info_path,
                )
            # 执行文章抛光，生成最终版本
            self.run_article_polishing_module(
                draft_article=draft_article, remove_duplicate=remove_duplicate
            )
