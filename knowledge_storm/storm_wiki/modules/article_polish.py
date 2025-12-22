import copy
from typing import Union

import dspy

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing


class StormArticlePolishingModule(ArticlePolishingModule):
    """
    文章润色阶段的接口。基于主题、知识策展阶段收集的信息和大纲生成阶段生成的轮廓来润色文章。
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm, polish_engine=self.article_polish_lm
        )

    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        润色文章，生成摘要部分并清理重复内容。

        参数:
            topic (str): 文章主题。
            draft_article (StormArticle): 草稿文章。
            remove_duplicate (bool): 是否使用额外的LM调用来移除文章中的重复内容。
        """
        # 关键路径：将文章对象转换为文本格式
        article_text = draft_article.to_string()

        # 关键路径：调用润色模块处理文章
        # - 生成摘要部分（lead section）
        # - 可选择是否清理重复内容
        polish_result = self.polish_page(
            topic=topic, draft_page=article_text, polish_whole_page=remove_duplicate
        )

        # 关键路径：构建摘要部分格式
        # 添加"# summary"标题标识摘要部分
        lead_section = f"# summary\n{polish_result.lead_section}"

        # 关键路径：组合摘要和正文
        # 使用双换行符分隔摘要和正文，形成完整文章
        polished_article = "\n\n".join([lead_section, polish_result.page])

        # 关键路径：解析文章文本为结构化字典
        # 将文本格式转换为可操作的数据结构
        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article
        )

        # 关键路径：深拷贝原始文章对象
        # 避免修改原始草稿对象
        polished_article = copy.deepcopy(draft_article)

        # 关键路径：插入或创建章节内容
        # 将润色后的内容整合到文章对象中
        polished_article.insert_or_create_section(article_dict=polished_article_dict)

        # 关键路径：执行最终后处理
        # 清理格式、统一引用、检查完整性
        polished_article.post_processing()
        return polished_article

    """为给定的维基百科页面编写摘要部分，指导原则如下：
    1. 摘要应独立存在，作为文章主题的简洁概述。它应识别主题、建立上下文、解释主题的重要性，并总结最重要的要点，包括任何突出的争议。
    2. 摘要部分应简洁，包含不超过四个精心组织的段落。
    3. 摘要部分应适当仔细标注来源。在必要时添加内联引用（例如，"华盛顿特区是美国的首都[1][3]"。）
    """


class WriteLeadSection(dspy.Signature):
    """Write a lead section for the given Wikipedia page with the following guidelines:
    1. The lead should stand on its own as a concise overview of the article's topic. It should identify the topic, establish context, explain why the topic is notable, and summarize the most important points, including any prominent controversies.
    2. The lead section should be concise and contain no more than four well-composed paragraphs.
    3. The lead section should be carefully sourced as appropriate. Add inline citations (e.g., "Washington, D.C., is the capital of the United States.[1][3].") where necessary.
    """

    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    draft_page = dspy.InputField(prefix="The draft page:\n", format=str)
    lead_section = dspy.OutputField(prefix="Write the lead section:\n", format=str)


"""
你是一款忠实的文字编辑工具，擅长在文章中查找重复的内容并将其删除，以确保文章中没有重复的内容。你不会删除文章中任何非重复的部分。
你会妥善保留行内引用以及文章结构（用“#”、“##”等表示）。请为以下文章完成你的工作。
"""


class PolishPage(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following article."""

    draft_page = dspy.InputField(prefix="The draft article:\n", format=str)
    page = dspy.OutputField(prefix="Your revised article:\n", format=str)


class PolishPageModule(dspy.Module):
    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # 关键路径：生成摘要部分
        # 注意：将show_guidelines设置为false以使生成对不同LM家族更加稳健
        with dspy.settings.context(lm=self.write_lead_engine, show_guidelines=False):
            lead_section = self.write_lead(
                topic=topic, draft_page=draft_page
            ).lead_section
            # 清理输出格式，移除前缀提示文本
            if "The lead section:" in lead_section:
                lead_section = lead_section.split("The lead section:")[1].strip()

        # 关键路径：根据标志决定是否润色整页
        if polish_whole_page:
            # 关键路径：执行页面润色，删除重复内容
            # 注意：将show_guidelines设置为false以使生成对不同LM家族更加稳健
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                page = self.polish_page(draft_page=draft_page).page
        else:
            # 如果不需要润色，返回原始页面
            page = draft_page

        # 返回润色结果：摘要部分和页面内容
        return dspy.Prediction(lead_section=lead_section, page=page)
