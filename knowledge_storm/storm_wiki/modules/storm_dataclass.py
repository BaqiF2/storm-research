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

    def _merge_new_info_to_references(
        self, new_info_list: List[Information], index_to_keep=None
    ) -> Dict[int, int]:
        """
        Merges new storm information into existing references and updates the citation index mapping.

        Args:
        new_info_list (List[Information]): A list of dictionaries representing new storm information.
        index_to_keep (List[int]): A list of index of the new_info_list to keep. If none, keep all.

        Returns:
        Dict[int, int]: A dictionary mapping the index of each storm information piece in the input list
                        to its unified citation index in the references.
        """
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
        parent_node = (
            self.root
            if parent_section_name is None
            else self.find_section(self.root, parent_section_name)
        )

        if trim_children:
            section_names = set(article_dict.keys())
            for child in parent_node.children[:]:
                if child.section_name not in section_names:
                    parent_node.remove_child(child)

        for section_name, content_dict in article_dict.items():
            current_section_node = self.find_section(parent_node, section_name)
            if current_section_node is None:
                current_section_node = ArticleSectionNode(
                    section_name=section_name, content=content_dict["content"].strip()
                )
                insert_to_front = (
                    parent_node.section_name == self.root.section_name
                    and current_section_node.section_name == "summary"
                )
                parent_node.add_child(
                    current_section_node, insert_to_front=insert_to_front
                )
            else:
                current_section_node.content = content_dict["content"].strip()

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
        Add new section to the article.

        Args:
            current_section_name: new section heading name in string format.
            parent_section_name: under which parent section to add the new one. Default to root.
            current_section_content: optional section content.

        Returns:
            the ArticleSectionNode for current section if successfully created / updated. Otherwise none.
        """

        if current_section_info_list is not None:
            references = set(
                [int(x) for x in re.findall(r"\[(\d+)\]", current_section_content)]
            )
            # for any reference number greater than max number of references, delete the reference
            if len(references) > 0:
                max_ref_num = max(references)
                if max_ref_num > len(current_section_info_list):
                    for i in range(len(current_section_info_list), max_ref_num + 1):
                        current_section_content = current_section_content.replace(
                            f"[{i}]", ""
                        )
                        if i in references:
                            references.remove(i)
            # for any reference that is not used, trim it from current_section_info_list
            index_to_keep = [i - 1 for i in references]
            citation_mapping = self._merge_new_info_to_references(
                current_section_info_list, index_to_keep
            )
            current_section_content = ArticleTextProcessing.update_citation_index(
                current_section_content, citation_mapping
            )

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
        Get outline of the article as a list.

        Args:
            section_name: get all section names in pre-order travel ordering in the subtree of section_name.
                          For example:
                            #root
                            ##section1
                            ###section1.1
                            ###section1.2
                            ##section2
                          article.get_outline_as_list("section1") returns [section1, section1.1, section1.2, section2]

        Returns:
            list of section and subsection names.
        """
        if root_section_name is None:
            section_node = self.root
        else:
            section_node = self.find_section(self.root, root_section_name)
            include_root = include_root or section_node != self.root.section_name
        if section_node is None:
            return []
        result = []

        def preorder_traverse(node, level):
            prefix = (
                "#" * level if add_hashtags else ""
            )  # Adjust level if excluding root
            result.append(
                f"{prefix} {node.section_name}".strip()
                if add_hashtags
                else node.section_name
            )
            for child in node.children:
                preorder_traverse(child, level + 1)

        # Adjust the initial level based on whether root is included and hashtags are added
        if include_root:
            preorder_traverse(section_node, level=1)
        else:
            for child in section_node.children:
                preorder_traverse(child, level=1)
        return result

    def to_string(self) -> str:
        """
        Get outline of the article as a list.

        Returns:
            list of section and subsection names.
        """
        result = []

        def preorder_traverse(node, level):
            prefix = "#" * level
            result.append(f"{prefix} {node.section_name}".strip())
            result.append(node.content)
            for child in node.children:
                preorder_traverse(child, level + 1)

        # Adjust the initial level based on whether root is included and hashtags are added
        for child in self.root.children:
            preorder_traverse(child, level=1)
        result = [i.strip() for i in result if i is not None and i.strip()]
        return "\n\n".join(result)

    def reorder_reference_index(self):
        # pre-order traversal to get order of references appear in the article
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
        # constrcut index mapping
        ref_index_mapping = {}
        for ref_index in ref_indices:
            if ref_index not in ref_index_mapping:
                ref_index_mapping[ref_index] = len(ref_index_mapping) + 1

        # update content
        def pre_order_update_index(node):
            if node is not None:
                if node.content is not None and node.content:
                    node.content = ArticleTextProcessing.update_citation_index(
                        node.content, ref_index_mapping
                    )
                for child in node.children:
                    pre_order_update_index(child)

        pre_order_update_index(self.root)
        # update reference
        for url in list(self.reference["url_to_unified_index"]):
            pre_index = self.reference["url_to_unified_index"][url]
            if pre_index not in ref_index_mapping:
                del self.reference["url_to_unified_index"][url]
            else:
                new_index = ref_index_mapping[pre_index]
                self.reference["url_to_unified_index"][url] = new_index

    def get_outline_tree(self):
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
    def from_outline_file(cls, topic: str, file_path: str):
        """
        Create StormArticle class instance from outline file.
        """
        outline_str = FileIOHelper.load_str(file_path)
        return StormArticle.from_outline_str(topic=topic, outline_str=outline_str)

    @classmethod
    def from_outline_str(cls, topic: str, outline_str: str):
        """
        Create StormArticle class instance from outline only string.
        """
        lines = []
        try:
            lines = outline_str.split("\n")
            lines = [line.strip() for line in lines if line.strip()]
        except:
            pass

        instance = cls(topic)
        if lines:
            a = lines[0].startswith("#") and lines[0].replace("#", "").strip().lower()
            b = topic.lower().replace("_", " ")
            adjust_level = lines[0].startswith("#") and lines[0].replace(
                "#", ""
            ).strip().lower() == topic.lower().replace("_", " ")
            if adjust_level:
                lines = lines[1:]
            node_stack = [(0, instance.root)]  # Stack to keep track of (level, node)

            for line in lines:
                level = line.count("#") - adjust_level
                section_name = line.replace("#", "").strip()

                if section_name == topic:
                    continue

                new_node = ArticleSectionNode(section_name)

                while node_stack and level <= node_stack[-1][0]:
                    node_stack.pop()

                node_stack[-1][1].add_child(new_node)
                node_stack.append((level, new_node))
        return instance

    def dump_outline_to_file(self, file_path):
        outline = self.get_outline_as_list(add_hashtags=True, include_root=False)
        FileIOHelper.write_str("\n".join(outline), file_path)

    def dump_reference_to_file(self, file_path):
        reference = copy.deepcopy(self.reference)
        for url in reference["url_to_info"]:
            reference["url_to_info"][url] = reference["url_to_info"][url].to_dict()
        FileIOHelper.dump_json(reference, file_path)

    def dump_article_as_plain_text(self, file_path):
        text = self.to_string()
        FileIOHelper.write_str(text, file_path)

    @classmethod
    def from_string(cls, topic_name: str, article_text: str, references: dict):
        article_dict = ArticleTextProcessing.parse_article_into_dict(article_text)
        article = cls(topic_name=topic_name)
        article.insert_or_create_section(article_dict=article_dict)
        for url in list(references["url_to_info"]):
            references["url_to_info"][url] = Information.from_dict(
                references["url_to_info"][url]
            )
        article.reference = references
        return article

    def post_processing(self):
        self.prune_empty_nodes()
        self.reorder_reference_index()
