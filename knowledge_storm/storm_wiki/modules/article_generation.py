import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing


class StormArticleGenerationModule(ArticleGenerationModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage,
    """

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)

    def generate_section(
        self, topic, section_name, information_table, section_outline, section_query
    ):
        collected_info: List[Information] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": collected_info,
        }

    def generate_article(
        self,
        topic: str,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        根据信息表和文章大纲生成主题文章。

        参数:
            topic (str): 文章主题。
            information_table (StormInformationTable): 包含收集信息的信息表。
            article_with_outline (StormArticle): 带有指定大纲的文章。
            callback_handler (BaseCallbackHandler): 可选的回调处理器，用于在文章生成过程的各个阶段触发自定义回调。默认为None。
        """
        # 准备信息表以供检索使用
        information_table.prepare_table_for_retrieval()

        # 如果没有提供文章大纲，创建一个新的文章对象
        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        # 获取需要编写的一级章节列表
        sections_to_write = article_with_outline.get_first_level_section_names()

        section_output_dict_collection = []
        # 关键路径：处理没有大纲的情况
        if len(sections_to_write) == 0:
            logging.error(
                f"No outline for {topic}. Will directly search with the topic."
            )
            # 直接使用主题作为章节进行生成
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic],
            )
            section_output_dict_collection = [section_output_dict]
        else:
            # 关键路径：使用线程池并行生成各章节
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    # 跳过独立的介绍章节
                    if section_title.lower().strip() == "introduction":
                        continue
                    # 跳过独立的结论章节
                    if section_title.lower().strip().startswith(
                        "conclusion"
                    ) or section_title.lower().strip().startswith("summary"):
                        continue
                    # 获取章节查询关键词（不带标签）
                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    # 获取章节查询关键词（带标签）
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True
                    )
                    # 构建章节大纲
                    section_outline = "\n".join(queries_with_hashtags)
                    # 提交并行任务
                    future_to_sec_title[
                        executor.submit(
                            self.generate_section,
                            topic,
                            section_title,
                            information_table,
                            section_outline,
                            section_query,
                        )
                    ] = section_title

                # 关键路径：收集所有并行任务的结果
                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())

        # 深拷贝文章对象以避免修改原始对象
        article = copy.deepcopy(article_with_outline)
        # 关键路径：更新文章各章节内容
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                parent_section_name=topic,
                current_section_content=section_output_dict["section_content"],
                current_section_info_list=section_output_dict["collected_info"],
            )
        # 执行文章后处理
        article.post_processing()
        return article


class ConvToSection(dspy.Module):
    """使用从信息搜索对话中收集的信息来编写章节。"""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(
        self, topic: str, outline: str, section: str, collected_info: List[Information]
    ):
        info = ""
        # 格式化收集的信息，为每个信息项添加编号
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets)
            info += "\n\n"

        # 限制信息长度，保留换行符
        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        # 使用语言模型生成章节内容
        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output
            )

        return dspy.Prediction(section=section)

    """根据收集的信息编写维基百科章节。

    写作格式要求：
        1. 使用"#" 标题 表示章节标题，"##" 标题 表示小节标题，"###" 标题 表示子小节标题，以此类推。
        2. 在行内使用[1], [2], ..., [n]引用（例如，"美国首都是华盛顿特区[1][3]"）。您不需要在末尾包含References或Sources部分来列出源。
    """


class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    """

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str,
    )
