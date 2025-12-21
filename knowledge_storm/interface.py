"""
此文件定义了 STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) 项目的核心接口和抽象基类。
它充当系统的骨架，规定了数据结构（如 Information, Article）和模块契约（如 KnowledgeCurationModule, ArticleGenerationModule）。

主要功能包括：
1. 定义基础数据类：用于存储检索信息、文章结构和章节内容。
2. 定义模块接口：规范知识搜集、大纲生成、文章写作和润色的标准方法。
3. 定义引擎和配置：提供执行引擎和语言模型配置的抽象基类。
4. 提供 Agent 接口：为 Co-STORM 等协作场景定义 Agent 的行为规范。
"""

import concurrent.futures
import dspy
import functools
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from .utils import ArticleTextProcessing

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .logging_wrapper import LoggingWrapper


class InformationTable(ABC):
    """
    The InformationTable class serves as data class to store the information
    collected during KnowledgeCuration stage.

    Create subclass to incorporate more information as needed. For example,
    in STORM paper https://arxiv.org/pdf/2402.14207.pdf, additional information
    would be perspective guided dialogue history.
    """

    def __init__(self):
        pass

    @abstractmethod
    def retrieve_information(**kwargs):
        pass


class Information:
    """表示详细信息的类。

    该类继承自Information，包含唯一标识符(URL)，并扩展了描述、片段和标题等属性，
    用于存储STORM系统中的信息对象。

    Attributes:
        url (str): 信息的唯一URL标识符
        description (str): 简要描述
        snippets (list): 文本片段或摘录列表
        title (str): 信息的标题或标头
        meta (dict): 元数据信息，如问题和查询等
        citation_uuid (int): 引用的唯一标识符
    """

    def __init__(self, url, description, snippets, title, meta=None):
        """使用详细属性初始化Information对象。

        Args:
            url (str): 作为信息标识符的唯一URL
            description (str): 详细描述
            snippets (list): 文本片段或摘录列表
            title (str): 信息的标题或标头
            meta (dict, optional): 元数据信息，默认为None
        """
        self.description = description
        self.snippets = snippets
        self.title = title
        self.url = url
        self.meta = meta if meta is not None else {}
        self.citation_uuid = -1  # 引用标识符，默认为-1

    def __hash__(self):
        """计算对象的哈希值（已废弃，请使用下面的MD5哈希版本）。

        Returns:
            int: 基于URL和片段的哈希值
        """
        return hash(
            (
                self.url,
                tuple(sorted(self.snippets)),
            )
        )

    def __eq__(self, other):
        """判断两个Information对象是否相等。

        比较URL、片段集合和元数据字符串是否都相同。

        Args:
            other: 要比较的另一个对象

        Returns:
            bool: 如果两个对象相等返回True，否则返回False
        """
        if not isinstance(other, Information):
            return False
        return (
            self.url == other.url
            and set(self.snippets) == set(other.snippets)
            and self._meta_str() == other._meta_str()
        )

    def __hash__(self):
        """计算对象的哈希值（基于MD5）。

        使用URL、排序后的片段和元数据字符串生成MD5哈希值。

        Returns:
            int: 16进制MD5哈希值转换为整数
        """
        return int(
            self._md5_hash((self.url, tuple(sorted(self.snippets)), self._meta_str())),
            16,
        )

    def _meta_str(self):
        """生成元数据信息的字符串表示。

        提取元数据中的问题和查询字段，格式化为字符串。

        Returns:
            str: 包含问题和查询的格式化字符串
        """
        return f"Question: {self.meta.get('question', '')}, Query: {self.meta.get('query', '')}"

    def _md5_hash(self, value):
        """为给定值生成MD5哈希。

        将输入值转换为字符串后计算MD5哈希值。对于字典、列表、元组等复杂类型，
        先转换为JSON字符串。

        Args:
            value: 要哈希的值，可以是任意类型

        Returns:
            str: 十六进制格式的MD5哈希值
        """
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        return hashlib.md5(str(value).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, info_dict):
        """从字典创建Information对象。

        用法示例: info = Information.from_dict(storm_info_dict)

        Args:
            info_dict (dict): 包含'url'、'description'、'snippets'和'title'等键的字典，
                            对应对象的属性

        Returns:
            Information: Information类的实例
        """
        info = cls(
            url=info_dict["url"],
            description=info_dict["description"],
            snippets=info_dict["snippets"],
            title=info_dict["title"],
            meta=info_dict.get("meta", None),
        )
        # 设置引用标识符
        info.citation_uuid = int(info_dict.get("citation_uuid", -1))
        return info

    def to_dict(self):
        """将Information对象转换为字典格式。

        Returns:
            dict: 包含所有对象属性的字典
        """
        return {
            "url": self.url,
            "description": self.description,
            "snippets": self.snippets,
            "title": self.title,
            "meta": self.meta,
            "citation_uuid": self.citation_uuid,
        }


class ArticleSectionNode:
    """
    The ArticleSectionNode is the dataclass for handling the section of the article.
    The content storage, section writing preferences are defined in this node.
    """

    def __init__(self, section_name: str, content=None):
        """
        section_name: section heading in string format. E.g. Introduction, History, etc.
        content: content of the section. Up to you for design choice of the data structure.
        """
        self.section_name = section_name
        self.content = content
        self.children = []
        self.preference = None

    def add_child(self, new_child_node, insert_to_front=False):
        if insert_to_front:
            self.children.insert(0, new_child_node)
        else:
            self.children.append(new_child_node)

    def remove_child(self, child):
        self.children.remove(child)


class Article(ABC):
    def __init__(self, topic_name):
        self.root = ArticleSectionNode(topic_name)

    def find_section(
        self, node: ArticleSectionNode, name: str
    ) -> Optional[ArticleSectionNode]:
        """
        Return the node of the section given the section name.

        Args:
            node: the node as the root to find.
            name: the name of node as section name

        Return:
            reference of the node or None if section name has no match
        """
        if node.section_name == name:
            return node
        for child in node.children:
            result = self.find_section(child, name)
            if result:
                return result
        return None

    @abstractmethod
    def to_string(self) -> str:
        """
        Export Article object into string representation.
        """

    def get_outline_tree(self):
        """
        Generates a hierarchical tree structure representing the outline of the document.

        Returns:
            Dict[str, Dict]: A nested dictionary representing the hierarchical structure of the document's outline.
                             Each key is a section name, and the value is another dictionary representing the child sections,
                             recursively forming the tree structure of the document's outline. If a section has no subsections,
                             its value is an empty dictionary.

        Example:
            Assuming a document with a structure like:
            - Introduction
                - Background
                - Objective
            - Methods
                - Data Collection
                - Analysis
            The method would return:
            {
                'Introduction': {
                    'Background': {},
                    'Objective': {}
                },
                'Methods': {
                    'Data Collection': {},
                    'Analysis': {}
                }
            }
        """

        def build_tree(node) -> Dict[str, Dict]:
            tree = {}
            for child in node.children:
                tree[child.section_name] = build_tree(child)
            return tree if tree else {}

        return build_tree(self.root)

    def get_first_level_section_names(self) -> List[str]:
        """
        Get first level section names
        """
        return [i.section_name for i in self.root.children]

    @classmethod
    @abstractmethod
    def from_string(cls, topic_name: str, article_text: str):
        """
        Create an instance of the Article object from a string
        """
        pass

    def prune_empty_nodes(self, node=None):
        if node is None:
            node = self.root

        node.children[:] = [
            child for child in node.children if self.prune_empty_nodes(child)
        ]

        if (node.content is None or node.content == "") and not node.children:
            return None
        else:
            return node


class Retriever:
    """
    检索器模块的抽象基类，提供基于查询检索信息的模板。

    该类应被扩展以实现特定的检索功能。
    用户可以通过实现retrieve方法来设计自己的检索器模块。
    每个部分使用的检索模型/搜索引擎应在属性名中使用'_rm'后缀进行声明。
    """

    def __init__(self, rm: dspy.Retrieve, max_thread: int = 1):
        """初始化检索器

        Args:
            rm: DSPy检索模型实例
            max_thread: 最大并发线程数，默认为1
        """
        self.max_thread = max_thread  # 最大线程数
        self.rm = rm  # 检索模型

    def collect_and_reset_rm_usage(self):
        """收集并重置检索模型的使用统计

        Returns:
            字典，键为模型名称，值为查询次数
        """
        combined_usage = []
        # 如果检索模型有使用统计方法，则收集统计信息
        if hasattr(getattr(self, "rm"), "get_usage_and_reset"):
            combined_usage.append(getattr(self, "rm").get_usage_and_reset())

        # 合并所有模型的使用统计
        name_to_usage = {}
        for usage in combined_usage:
            for model_name, query_cnt in usage.items():
                if model_name not in name_to_usage:
                    name_to_usage[model_name] = query_cnt
                else:
                    name_to_usage[model_name] += query_cnt

        return name_to_usage

    def retrieve(
        self, query: Union[str, List[str]], exclude_urls: List[str] = []
    ) -> List[Information]:
        """检索与查询相关的信息

        Args:
            query: 单个查询字符串或查询字符串列表
            exclude_urls: 需要排除的URL列表，默认为空列表

        Returns:
            Information对象列表，包含检索到的信息
        """
        # 将单个查询转换为列表格式
        queries = query if isinstance(query, list) else [query]
        to_return = []  # 存储所有检索结果

        def process_query(q):
            """处理单个查询

            Args:
                q: 查询字符串

            Returns:
                Information对象列表
            """
            # 使用检索模型执行查询
            retrieved_data_list = self.rm(
                query_or_queries=[q], exclude_urls=exclude_urls
            )
            local_to_return = []
            for data in retrieved_data_list:
                for i in range(len(data["snippets"])):
                    # STORM生成带引用的文章。我们不考虑多跳引用。
                    # 移除源文本中的引用以避免混淆。
                    data["snippets"][i] = ArticleTextProcessing.remove_citations(
                        data["snippets"][i]
                    )
                # 将检索数据转换为Information对象
                storm_info = Information.from_dict(data)
                storm_info.meta["query"] = q  # 保存原始查询
                local_to_return.append(storm_info)
            return local_to_return

        # 使用线程池并发处理所有查询
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_thread
        ) as executor:
            results = list(executor.map(process_query, queries))

        # 合并所有查询的结果
        for result in results:
            to_return.extend(result)

        return to_return


class KnowledgeCurationModule(ABC):
    """
    The interface for knowledge curation stage. Given topic, return collected information.
    """

    def __init__(self, retriever: Retriever):
        """
        Store args and finish initialization.
        """
        self.retriever = retriever

    @abstractmethod
    def research(self, topic) -> InformationTable:
        """
        Curate information and knowledge for the given topic

        Args:
            topic: topic of interest in natural language.

        Returns:
            collected_information: collected information in InformationTable type.
        """
        pass


class OutlineGenerationModule(ABC):
    """
    The interface for outline generation stage. Given topic, collected information from knowledge
    curation stage, generate outline for the article.
    """

    @abstractmethod
    def generate_outline(
        self, topic: str, information_table: InformationTable, **kwargs
    ) -> Article:
        """
        Generate outline for the article. Required arguments include:
            topic: the topic of interest
            information_table: knowledge curation data generated from KnowledgeCurationModule

        More arguments could be
            1. draft outline
            2. user provided outline

        Returns:
            article_outline of type ArticleOutline
        """
        pass


class ArticleGenerationModule(ABC):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage,
    """

    @abstractmethod
    def generate_article(
        self,
        topic: str,
        information_table: InformationTable,
        article_with_outline: Article,
        **kwargs,
    ) -> Article:
        """
        Generate article. Required arguments include:
            topic: the topic of interest
            information_table: knowledge curation data generated from KnowledgeCurationModule
            article_with_outline: article with specified outline from OutlineGenerationModule
        """
        pass


class ArticlePolishingModule(ABC):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage,
    """

    @abstractmethod
    def polish_article(self, topic: str, draft_article: Article, **kwargs) -> Article:
        """
        Polish article. Required arguments include:
            topic: the topic of interest
            draft_article: draft article from ArticleGenerationModule.
        """
        pass


def log_execution_time(func):
    """Decorator to log the execution time of a function."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        self.time[func.__name__] = execution_time
        return result

    return wrapper


class LMConfigs(ABC):
    """Abstract base class for language model configurations of the knowledge curation engine.

    The language model used for each part should be declared with a suffix '_lm' in the attribute name.
    """

    def __init__(self):
        pass

    def init_check(self):
        for attr_name in self.__dict__:
            if "_lm" in attr_name and getattr(self, attr_name) is None:
                logging.warning(
                    f"Language model for {attr_name} is not initialized. Please call set_{attr_name}()"
                )

    def collect_and_reset_lm_history(self):
        history = []
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(getattr(self, attr_name), "history"):
                history.extend(getattr(self, attr_name).history)
                getattr(self, attr_name).history = []

        return history

    def collect_and_reset_lm_usage(self):
        combined_usage = []
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(
                getattr(self, attr_name), "get_usage_and_reset"
            ):
                combined_usage.append(getattr(self, attr_name).get_usage_and_reset())

        model_name_to_usage = {}
        for usage in combined_usage:
            for model_name, tokens in usage.items():
                if model_name not in model_name_to_usage:
                    model_name_to_usage[model_name] = tokens
                else:
                    model_name_to_usage[model_name]["prompt_tokens"] += tokens[
                        "prompt_tokens"
                    ]
                    model_name_to_usage[model_name]["completion_tokens"] += tokens[
                        "completion_tokens"
                    ]

        return model_name_to_usage

    def log(self):
        return OrderedDict(
            {
                attr_name: getattr(self, attr_name).kwargs
                for attr_name in self.__dict__
                if "_lm" in attr_name and hasattr(getattr(self, attr_name), "kwargs")
            }
        )


class Engine(ABC):
    """引擎基类，用于管理知识生成流程的执行和资源统计"""

    def __init__(self, lm_configs: LMConfigs):
        """初始化引擎

        Args:
            lm_configs: 语言模型配置对象
        """
        self.lm_configs = lm_configs
        self.time = {}  # 存储各个函数的执行时间
        self.lm_cost = {}  # 语言模型的成本，通过输入/输出token数量衡量
        self.rm_cost = {}  # 检索器的成本，通过查询次数衡量

    def log_execution_time_and_lm_rm_usage(self, func):
        """装饰器：记录函数的执行时间、语言模型使用情况和检索模型使用情况

        Args:
            func: 需要被装饰的函数

        Returns:
            装饰后的函数
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # 记录开始时间
            result = func(*args, **kwargs)  # 执行原函数
            end_time = time.time()  # 记录结束时间
            execution_time = end_time - start_time  # 计算执行时长
            self.time[func.__name__] = execution_time  # 保存执行时间
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            # 收集并重置语言模型的使用统计
            self.lm_cost[func.__name__] = self.lm_configs.collect_and_reset_lm_usage()
            # 如果存在检索器，收集并重置检索模型的使用统计
            if hasattr(self, "retriever"):
                self.rm_cost[func.__name__] = (
                    self.retriever.collect_and_reset_rm_usage()
                )
            return result

        return wrapper

    def apply_decorators(self):
        """为需要装饰的方法应用装饰器

        自动查找所有以 'run_' 开头的方法，并为它们应用执行时间和资源使用统计装饰器
        """
        # 查找所有以 'run_' 开头的可调用方法
        methods_to_decorate = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and method_name.startswith("run_")
        ]
        # 为每个方法应用装饰器
        for method_name in methods_to_decorate:
            original_method = getattr(self, method_name)  # 获取原始方法
            decorated_method = self.log_execution_time_and_lm_rm_usage(
                original_method
            )  # 应用装饰器
            setattr(self, method_name, decorated_method)  # 替换为装饰后的方法

    @abstractmethod
    def run_knowledge_curation_module(self, **kwargs) -> Optional[InformationTable]:
        """运行知识整理模块（抽象方法，需在子类中实现）

        Returns:
            信息表对象，如果不适用则返回None
        """
        pass

    @abstractmethod
    def run_outline_generation_module(self, **kwarg) -> Article:
        """运行大纲生成模块（抽象方法，需在子类中实现）

        Returns:
            包含大纲的文章对象
        """
        pass

    @abstractmethod
    def run_article_generation_module(self, **kwarg) -> Article:
        """运行文章生成模块（抽象方法，需在子类中实现）

        Returns:
            生成的文章对象
        """
        pass

    @abstractmethod
    def run_article_polishing_module(self, **kwarg) -> Article:
        """运行文章润色模块（抽象方法，需在子类中实现）

        Returns:
            润色后的文章对象
        """
        pass

    @abstractmethod
    def run(self, **kwargs):
        """运行完整的处理流程（抽象方法，需在子类中实现）"""
        pass

    def summary(self):
        """打印执行摘要，包括执行时间、语言模型token使用量和检索模型查询次数"""
        print("***** 执行时间 *****")
        for k, v in self.time.items():
            print(f"{k}: {v:.4f} 秒")

        print("***** 语言模型Token使用量: *****")
        for k, v in self.lm_cost.items():
            print(f"{k}")
            for model_name, tokens in v.items():
                print(f"    {model_name}: {tokens}")

        print("***** 检索模型查询次数: *****")
        for k, v in self.rm_cost.items():
            print(f"{k}: {v}")

    def reset(self):
        """重置所有统计信息（执行时间、语言模型成本、检索模型成本）"""
        self.time = {}
        self.lm_cost = {}
        self.rm_cost = {}


class Agent(ABC):
    """
    Interface for STORM and Co-STORM LLM agent

    This class must be implemented by any subclass of `Agent` to define how the agent generates an utterance.
    The generated utterance can be influenced by the conversation history, knowledge base, and any additional parameters passed via `kwargs`.
    The implementation should align with the specific role and perspective of the agent, as defined by the agent's topic, role name, and role description.

    Args:
        knowledge_base (KnowledgeBase): The current knowledge base (e.g., mind map in Co-STORM) that contains the accumulated information relevant to the conversation.
        conversation_history (List[ConversationTurn]): A list of past conversation turns, providing context for generating the next utterance.
                                                       The agent can refer to this history to maintain continuity and relevance in the conversation.
        logging_wrapper (LoggingWrapper): A wrapper used for logging important events during the utterance generation process.
        **kwargs: Additional arguments that can be passed to the method for more specialized utterance generation behavior depending on the agent's specific implementation.

    Returns:
        ConversationTurn: A new conversation turn generated by the agent, containing the agent's response, including the role, utterance type, and relevant information from the knowledge base.

    Notes:
        - Subclasses of `Agent` should define the exact strategy for generating the utterance, which could involve interacting with a language model, retrieving relevant knowledge, or following specific conversational policies.
        - The agent's role, perspective, and the knowledge base content will influence how the utterance is formulated.
    """

    from .dataclass import KnowledgeBase, ConversationTurn

    def __init__(self, topic: str, role_name: str, role_description: str):
        self.topic = topic
        self.role_name = role_name
        self.role_description = role_description

    def get_role_description(self):
        if self.role_description:
            return f"{self.role_name}: {self.role_description}"
        return self.role_name

    @abstractmethod
    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
        logging_wrapper: "LoggingWrapper",
        **kwargs,
    ):
        pass
