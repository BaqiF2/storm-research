import copy
import re
from collections import OrderedDict
from typing import Union, Optional, Any, List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ...interface import Information, InformationTable, Article, ArticleSectionNode
from ...utils import ArticleTextProcessing, FileIOHelper


class DialogueTurn:
    """对话轮次类，表示一次完整的对话交互。

    该类封装了一次对话轮次中的所有信息，包括代理的回复、用户的输入、
    搜索查询以及搜索结果。主要用于在知识整理阶段记录和存储对话历史。

    Attributes:
        agent_utterance: 代理的回复内容
        user_utterance: 用户的输入内容
        search_queries: 执行的搜索查询列表
        search_results: 搜索结果列表，每个元素为Information对象
    """

    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        search_queries: Optional[List[str]] = None,
        search_results: Optional[List[Union[Information, Dict]]] = None,
    ):
        """初始化对话轮次。

        Args:
            agent_utterance: 代理的回复内容，可选
            user_utterance: 用户的输入内容，可选
            search_queries: 执行的搜索查询列表，可选
            search_results: 搜索结果列表，可以是Information对象或字典格式，可选
        """
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.search_queries = search_queries
        self.search_results = search_results

        # 将字典格式的搜索结果转换为Information对象
        if self.search_results:
            for idx in range(len(self.search_results)):
                if type(self.search_results[idx]) == dict:
                    self.search_results[idx] = Information.from_dict(
                        self.search_results[idx]
                    )

    def log(self):
        """生成包含对话轮次所有信息的日志对象。

        将对话轮次中的所有信息转换为有序字典格式，便于序列化和存储。
        搜索结果会被转换为字典格式。

        Returns:
            OrderedDict: 包含agent_utterance、user_utterance、search_queries
                        和search_results的有序字典
        """
        return OrderedDict(
            {
                "agent_utterance": self.agent_utterance,
                "user_utterance": self.user_utterance,
                "search_queries": self.search_queries,
                "search_results": [data.to_dict() for data in self.search_results],
            }
        )


class StormInformationTable(InformationTable):
    """
    信息表类，用于存储知识整理(KnowledgeCuration)阶段收集的信息。

    可以根据需要创建子类以包含更多信息。例如，在 STORM 论文
    https://arxiv.org/pdf/2402.14207.pdf 中，额外的信息可能包括
    视角引导的对话历史。

    Attributes:
        conversations: 对话列表，每个元素为(角色, 对话轮次列表)的元组
        url_to_info: URL到信息对象的映射字典
    """

    def __init__(self, conversations=List[Tuple[str, List[DialogueTurn]]]):
        """初始化信息表。

        Args:
            conversations: 对话列表，格式为[(角色名, [对话轮次])]
        """
        super().__init__()
        self.conversations = conversations
        self.url_to_info: Dict[str, Information] = (
            StormInformationTable.construct_url_to_info(self.conversations)
        )

    @staticmethod
    def construct_url_to_info(
        conversations: List[Tuple[str, List[DialogueTurn]]]
    ) -> Dict[str, Information]:
        """从对话中构建URL到信息对象的映射。

        遍历所有对话轮次，提取搜索结果中的信息，按URL聚合相同来源的片段，
        并去除重复的片段。

        Args:
            conversations: 对话列表，格式为[(角色名, [对话轮次])]

        Returns:
            Dict[str, Information]: URL到信息对象的映射字典
        """
        url_to_info = {}

        # 遍历所有对话，提取搜索结果信息
        for persona, conv in conversations:
            for turn in conv:
                for storm_info in turn.search_results:
                    if storm_info.url in url_to_info:
                        # URL已存在，追加新片段
                        url_to_info[storm_info.url].snippets.extend(storm_info.snippets)
                    else:
                        # 新URL，创建新条目
                        url_to_info[storm_info.url] = storm_info
        # 去除每个URL下的重复片段
        for url in url_to_info:
            url_to_info[url].snippets = list(set(url_to_info[url].snippets))
        return url_to_info

    @staticmethod
    def construct_log_dict(
        conversations: List[Tuple[str, List[DialogueTurn]]]
    ) -> List[Dict[str, Union[str, Any]]]:
        """将对话转换为日志字典格式。

        Args:
            conversations: 对话列表，格式为[(角色名, [对话轮次])]

        Returns:
            List[Dict]: 对话日志列表，每个元素包含perspective和dlg_turns字段
        """
        conversation_log = []
        for persona, conv in conversations:
            conversation_log.append(
                {"perspective": persona, "dlg_turns": [turn.log() for turn in conv]}
            )
        return conversation_log

    def dump_url_to_info(self, path):
        """将URL到信息的映射导出为JSON文件。

        Args:
            path: 输出文件路径
        """
        url_to_info = copy.deepcopy(self.url_to_info)
        # 将信息对象转换为字典格式
        for url in url_to_info:
            url_to_info[url] = url_to_info[url].to_dict()
        FileIOHelper.dump_json(url_to_info, path)

    @classmethod
    def from_conversation_log_file(cls, path):
        """从对话日志文件加载信息表。

        Args:
            path: 对话日志文件路径

        Returns:
            StormInformationTable: 从日志重建的信息表实例
        """
        conversation_log_data = FileIOHelper.load_json(path)
        conversations = []
        for item in conversation_log_data:
            # 重建对话轮次
            dialogue_turns = [DialogueTurn(**turn) for turn in item["dlg_turns"]]
            persona = item["perspective"]
            conversations.append((persona, dialogue_turns))
        return cls(conversations)

    def prepare_table_for_retrieval(self):
        """准备信息表用于检索。

        初始化句子编码器，收集所有URL和文本片段，并对片段进行编码。
        编码后的片段将用于后续的相似度检索。
        """
        # 初始化句子编码器
        self.encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.collected_urls = []
        self.collected_snippets = []
        # 收集所有URL和对应的文本片段
        for url, information in self.url_to_info.items():
            for snippet in information.snippets:
                self.collected_urls.append(url)
                self.collected_snippets.append(snippet)
        # 对所有片段进行编码
        self.encoded_snippets = self.encoder.encode(self.collected_snippets)

    def retrieve_information(
        self, queries: Union[List[str], str], search_top_k
    ) -> List[Information]:
        """根据查询检索相关信息。

        使用余弦相似度计算查询与文本片段的相似度，返回最相关的信息。

        Args:
            queries: 查询字符串或查询列表
            search_top_k: 每个查询返回的top-k片段数量

        Returns:
            List[Information]: 检索到的信息对象列表
        """
        selected_urls = []
        selected_snippets = []
        # 确保queries是列表格式
        if type(queries) is str:
            queries = [queries]
        # 对每个查询进行检索
        for query in queries:
            # 编码查询
            encoded_query = self.encoder.encode(query)
            # 计算余弦相似度
            sim = cosine_similarity([encoded_query], self.encoded_snippets)[0]
            # 按相似度排序
            sorted_indices = np.argsort(sim)
            # 选择top-k最相似的片段
            for i in sorted_indices[-search_top_k:][::-1]:
                selected_urls.append(self.collected_urls[i])
                selected_snippets.append(self.collected_snippets[i])

        # 按URL聚合选中的片段
        url_to_snippets = {}
        for url, snippet in zip(selected_urls, selected_snippets):
            if url not in url_to_snippets:
                url_to_snippets[url] = set()
            url_to_snippets[url].add(snippet)

        # 构建返回的信息对象
        selected_url_to_info = {}
        for url in url_to_snippets:
            selected_url_to_info[url] = copy.deepcopy(self.url_to_info[url])
            selected_url_to_info[url].snippets = list(url_to_snippets[url])

        return list(selected_url_to_info.values())


class StormArticle(Article):
    def __init__(self, topic_name):
        super().__init__(topic_name=topic_name)
        self.reference = {"url_to_unified_index": {}, "url_to_info": {}}

    def find_section(
        self, node: ArticleSectionNode, name: str
    ) -> Optional[ArticleSectionNode]:
        """
        根据章节名称查找并返回对应的节点。

        Args:
            node: 作为查找根节点的起始节点。
            name: 节点名称，即章节名称

        Return:
            节点的引用，如果章节名称没有匹配则返回None
        """
        if node.section_name == name:
            return node
        for child in node.children:
            result = self.find_section(child, name)
            if result:
                return result
        return None

    def _merge_new_info_to_references(
        self, new_info_list: List[Information], index_to_keep=None
    ) -> Dict[int, int]:
        """
        将新的storm信息合并到现有引用中，并更新引用索引映射。

        Args:
        new_info_list (List[Information]): 表示新storm信息的字典列表。
        index_to_keep (List[int]): 要保留的new_info_list的索引列表。如果为None，则保留所有。

        Returns:
        Dict[int, int]: 一个字典，将输入列表中每个storm信息的索引映射到引用中的统一引用索引。
        """
        # 关键路径：合并新的引用信息，去重并更新索引映射
        citation_idx_mapping = {}
        for idx, storm_info in enumerate(new_info_list):
            if index_to_keep is not None and idx not in index_to_keep:
                continue
            url = storm_info.url
            if url not in self.reference["url_to_unified_index"]:
                self.reference["url_to_unified_index"][url] = (
                    len(self.reference["url_to_unified_index"]) + 1
                )  # The citation index starts from 1.
                self.reference["url_to_info"][url] = storm_info
            else:
                existing_snippets = self.reference["url_to_info"][url].snippets
                existing_snippets.extend(storm_info.snippets)
                self.reference["url_to_info"][url].snippets = list(
                    set(existing_snippets)
                )
            citation_idx_mapping[idx + 1] = self.reference["url_to_unified_index"][
                url
            ]  # The citation index starts from 1.
        return citation_idx_mapping

    def insert_or_create_section(
        self,
        article_dict: Dict[str, Dict],
        parent_section_name: str = None,
        trim_children=False,
    ):
        """
        根据字典结构插入或创建文章章节。

        Args:
            article_dict: 包含章节信息的字典结构
            parent_section_name: 父章节名称，默认为None（使用根节点）
            trim_children: 是否裁剪不在article_dict中的子章节，默认为False
        """
        # 关键路径：查找父节点，如果未指定则使用根节点
        parent_node = (
            self.root
            if parent_section_name is None
            else self.find_section(self.root, parent_section_name)
        )

        # 关键路径：根据trim_children标志决定是否裁剪子节点
        if trim_children:
            section_names = set(article_dict.keys())
            for child in parent_node.children[:]:
                if child.section_name not in section_names:
                    parent_node.remove_child(child)

        # 关键路径：遍历所有章节，插入或更新节点
        for section_name, content_dict in article_dict.items():
            # 在父节点中查找当前章节节点
            current_section_node = self.find_section(parent_node, section_name)
            if current_section_node is None:
                # 如果章节不存在，则创建新节点
                current_section_node = ArticleSectionNode(
                    section_name=section_name, content=content_dict["content"].strip()
                )
                # 关键路径：如果是根节点下的summary章节，插入到前面
                insert_to_front = (
                    parent_node.section_name == self.root.section_name
                    and current_section_node.section_name == "summary"
                )
                parent_node.add_child(
                    current_section_node, insert_to_front=insert_to_front
                )
            else:
                # 如果章节已存在，则更新内容
                current_section_node.content = content_dict["content"].strip()

            # 递归处理子章节
            self.insert_or_create_section(
                article_dict=content_dict["subsections"],
                parent_section_name=section_name,
                trim_children=True,
            )

    def update_section(
        self,
        current_section_content: str,
        current_section_info_list: List[Information],
        parent_section_name: Optional[str] = None,
    ) -> Optional[ArticleSectionNode]:
        """
        向文章添加或更新章节。

        Args:
            current_section_content: 新章节的标题名称，字符串格式。
            current_section_info_list: 章节内容的信息列表。
            parent_section_name: 要添加新章节的父章节名称。默认为根节点。
            current_section_content: 可选的章节内容。

        Returns:
            如果成功创建/更新，则返回当前章节的ArticleSectionNode。否则返回None。
        """
        # 关键路径：处理引用信息，更新引用索引
        if current_section_info_list is not None:
            # 从内容中提取引用编号
            references = set(
                [int(x) for x in re.findall(r"\[(\d+)\]", current_section_content)]
            )
            # 对于超出最大引用数的引用编号，删除该引用
            if len(references) > 0:
                max_ref_num = max(references)
                if max_ref_num > len(current_section_info_list):
                    for i in range(len(current_section_info_list), max_ref_num + 1):
                        current_section_content = current_section_content.replace(
                            f"[{i}]", ""
                        )
                        if i in references:
                            references.remove(i)
            # 对于未使用的引用，从current_section_info_list中裁剪掉
            index_to_keep = [i - 1 for i in references]
            citation_mapping = self._merge_new_info_to_references(
                current_section_info_list, index_to_keep
            )
            # 更新内容中的引用索引
            current_section_content = ArticleTextProcessing.update_citation_index(
                current_section_content, citation_mapping
            )

        # 关键路径：解析内容并插入章节
        if parent_section_name is None:
            parent_section_name = self.root.section_name
        article_dict = ArticleTextProcessing.parse_article_into_dict(
            current_section_content
        )
        self.insert_or_create_section(
            article_dict=article_dict,
            parent_section_name=parent_section_name,
            trim_children=False,
        )

    def get_outline_as_list(
        self,
        root_section_name: Optional[str] = None,
        add_hashtags: bool = False,
        include_root: bool = True,
    ) -> List[str]:
        """
        获取文章大纲为列表形式。

        Args:
            root_section_name: 获取指定章节名称的子树中所有章节名称，采用前序遍历顺序。
                          例如：
                            #root
                            ##section1
                            ###section1.1
                            ###section1.2
                            ##section2
                          article.get_outline_as_list("section1") 返回 [section1, section1.1, section1.2, section2]
            add_hashtags: 是否添加井号标记（#）用于表示层级，默认为False
            include_root: 是否包含根节点，默认为True

        Returns:
            章节和子章节名称的列表。
        """
        # 关键路径：根据root_section_name查找节点
        if root_section_name is None:
            section_node = self.root
        else:
            section_node = self.find_section(self.root, root_section_name)
            # 如果查找的不是根节点，则调整include_root标志
            include_root = include_root or section_node != self.root.section_name
        if section_node is None:
            return []
        result = []

        def preorder_traverse(node, level):
            # 关键路径：构建前缀符号（#）表示层级
            prefix = "#" * level if add_hashtags else ""  # 如果排除根节点，则调整层级
            result.append(
                f"{prefix} {node.section_name}".strip()
                if add_hashtags
                else node.section_name
            )
            # 递归遍历子节点
            for child in node.children:
                preorder_traverse(child, level + 1)

        # 关键路径：根据是否包含根节点和添加井号标记调整初始层级
        if include_root:
            preorder_traverse(section_node, level=1)
        else:
            # 不包含根节点时，只遍历子节点
            for child in section_node.children:
                preorder_traverse(child, level=1)
        return result

    def to_string(self) -> str:
        """
        将文章转换为字符串格式。

        Returns:
            包含章节和子章节名称及其内容的字符串列表。
        """
        result = []

        def preorder_traverse(node, level):
            # 关键路径：构建章节标题前缀
            prefix = "#" * level
            result.append(f"{prefix} {node.section_name}".strip())
            result.append(node.content)
            # 递归遍历子节点
            for child in node.children:
                preorder_traverse(child, level + 1)

        # 关键路径：从根节点的子节点开始遍历
        for child in self.root.children:
            preorder_traverse(child, level=1)
        # 清理空行并合并
        result = [i.strip() for i in result if i is not None and i.strip()]
        return "\n\n".join(result)

    def reorder_reference_index(self):
        """
        重新排序引用索引，使引用编号按照文章中出现的顺序重新排列。
        """
        # 关键路径：前序遍历文章，获取引用在文章中出现的顺序
        ref_indices = []

        def pre_order_find_index(node):
            if node is not None:
                if node.content is not None and node.content:
                    ref_indices.extend(
                        ArticleTextProcessing.parse_citation_indices(node.content)
                    )
                for child in node.children:
                    pre_order_find_index(child)

        pre_order_find_index(self.root)
        # 构建索引映射：旧索引 -> 新索引
        ref_index_mapping = {}
        for ref_index in ref_indices:
            if ref_index not in ref_index_mapping:
                ref_index_mapping[ref_index] = len(ref_index_mapping) + 1

        # 关键路径：更新内容中的引用索引
        def pre_order_update_index(node):
            if node is not None:
                if node.content is not None and node.content:
                    node.content = ArticleTextProcessing.update_citation_index(
                        node.content, ref_index_mapping
                    )
                for child in node.children:
                    pre_order_update_index(child)

        pre_order_update_index(self.root)
        # 关键路径：更新引用字典中的索引
        for url in list(self.reference["url_to_unified_index"]):
            pre_index = self.reference["url_to_unified_index"][url]
            if pre_index not in ref_index_mapping:
                # 如果旧索引不在映射中，则删除该引用
                del self.reference["url_to_unified_index"][url]
            else:
                # 更新为新的索引
                new_index = ref_index_mapping[pre_index]
                self.reference["url_to_unified_index"][url] = new_index

    def get_outline_tree(self):
        """
        获取文章大纲的树形结构。
        """

        def build_tree(node) -> Dict[str, Dict]:
            tree = {}
            # 递归构建子树
            for child in node.children:
                tree[child.section_name] = build_tree(child)
            return tree if tree else {}

        return build_tree(self.root)

    def get_first_level_section_names(self) -> List[str]:
        """
        获取一级章节名称列表。
        """
        # 关键路径：返回根节点的所有直接子节点名称
        return [i.section_name for i in self.root.children]

    @classmethod
    def from_outline_file(cls, topic: str, file_path: str):
        """
        从大纲文件创建StormArticle类实例。

        Args:
            topic: 文章主题
            file_path: 大纲文件路径

        Returns:
            StormArticle实例
        """
        outline_str = FileIOHelper.load_str(file_path)
        return StormArticle.from_outline_str(topic=topic, outline_str=outline_str)

    @classmethod
    def from_outline_str(cls, topic: str, outline_str: str):
        """
        从大纲字符串创建StormArticle类实例。

        Args:
            topic: 文章主题
            outline_str: 大纲字符串

        Returns:
            StormArticle实例
        """
        lines = []
        try:
            # 关键路径：解析大纲字符串，去除空行
            lines = outline_str.split("\n")
            lines = [line.strip() for line in lines if line.strip()]
        except:
            pass

        instance = cls(topic)
        if lines:
            # 关键路径：检查第一行是否为主题行，调整层级
            adjust_level = lines[0].startswith("#") and lines[0].replace(
                "#", ""
            ).strip().lower() == topic.lower().replace("_", " ")
            if adjust_level:
                lines = lines[1:]
            # 关键路径：使用栈结构维护层级关系
            node_stack = [(0, instance.root)]  # 栈用于跟踪（层级，节点）

            for line in lines:
                # 计算当前行的层级和章节名称
                level = line.count("#") - adjust_level
                section_name = line.replace("#", "").strip()

                if section_name == topic:
                    continue

                new_node = ArticleSectionNode(section_name)

                # 关键路径：维护栈，确保当前节点父级关系正确
                while node_stack and level <= node_stack[-1][0]:
                    node_stack.pop()

                # 添加新节点到当前父节点
                node_stack[-1][1].add_child(new_node)
                node_stack.append((level, new_node))
        return instance

    def dump_outline_to_file(self, file_path):
        """
        将文章大纲导出到文件。

        Args:
            file_path: 输出文件路径
        """
        # 关键路径：生成带井号标记的大纲，不包含根节点
        outline = self.get_outline_as_list(add_hashtags=True, include_root=False)
        FileIOHelper.write_str("\n".join(outline), file_path)

    def dump_reference_to_file(self, file_path):
        """
        将引用信息导出到文件。

        Args:
            file_path: 输出文件路径
        """
        # 关键路径：深度复制引用并转换为字典格式
        reference = copy.deepcopy(self.reference)
        for url in reference["url_to_info"]:
            reference["url_to_info"][url] = reference["url_to_info"][url].to_dict()
        FileIOHelper.dump_json(reference, file_path)

    def dump_article_as_plain_text(self, file_path):
        """
        将文章导出为纯文本文件。

        Args:
            file_path: 输出文件路径
        """
        # 关键路径：将文章转换为字符串并写入文件
        text = self.to_string()
        FileIOHelper.write_str(text, file_path)

    @classmethod
    def from_string(cls, topic_name: str, article_text: str, references: dict):
        """
        从字符串和引用信息创建StormArticle实例。

        Args:
            topic_name: 文章主题名称
            article_text: 文章文本内容
            references: 引用信息字典

        Returns:
            StormArticle实例
        """
        # 关键路径：解析文本并创建章节结构
        article_dict = ArticleTextProcessing.parse_article_into_dict(article_text)
        article = cls(topic_name=topic_name)
        article.insert_or_create_section(article_dict=article_dict)
        # 关键路径：将引用字典转换为Information对象
        for url in list(references["url_to_info"]):
            references["url_to_info"][url] = Information.from_dict(
                references["url_to_info"][url]
            )
        article.reference = references
        return article

    def post_processing(self):
        """
        文章后处理：清理空节点并重新排序引用索引。
        """
        # 关键路径：首先清理空节点，然后重新排序引用
        self.prune_empty_nodes()
        self.reorder_reference_index()
