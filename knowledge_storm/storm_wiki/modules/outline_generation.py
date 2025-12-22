from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing


class StormOutlineGenerationModule(OutlineGenerationModule):
    """
    大纲生成阶段的接口类。根据指定主题和知识策展阶段收集的信息，为文章生成大纲。
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm)

    def generate_outline(
        self,
        topic: str,
        information_table: StormInformationTable,
        old_outline: Optional[StormArticle] = None,
        callback_handler: BaseCallbackHandler = None,
        return_draft_outline=False,
    ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        """
        根据指定主题和知识策展阶段收集的信息，为文章生成大纲。此方法可选择性地同时返回最终文章大纲和草稿大纲。

        Args:
            topic (str): 文章的主题。
            information_table (StormInformationTable): 包含已收集信息的信息表。
            old_outline (Optional[StormArticle]): 可选的先前版本文章大纲，可用于参考或比较。默认为None。
            callback_handler (BaseCallbackHandler): 可选的回调处理器，用于在大纲生成过程的各个阶段触发自定义回调，
                例如当信息组织开始时。默认为None。
            return_draft_outline (bool): 指示方法是否应同时返回最终文章大纲和草稿大纲的标志。如果为False，
                则仅返回最终文章大纲。默认为False。

        Returns:
            Union[StormArticle, Tuple[StormArticle, StormArticle]]: 根据`return_draft_outline`的值，此方法返回单个包含最终大纲的
                `StormArticle`对象，或包含两个`StormArticle`对象的元组（第一个包含最终大纲，第二个包含草稿大纲）。
        """
        # 关键路径：触发信息组织开始的回调
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        # 关键路径：将所有对话轮次连接成单一列表
        concatenated_dialogue_turns = sum(
            [conv for (_, conv) in information_table.conversations], []
        )
        # 关键路径：调用内部大纲生成逻辑
        result = self.write_outline(
            topic=topic,
            dlg_history=concatenated_dialogue_turns,
            callback_handler=callback_handler,
        )
        # 关键路径：从大纲字符串创建文章对象
        article_with_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.outline
        )
        article_with_draft_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.old_outline
        )
        # 关键路径：根据返回标志决定返回单个对象或元组
        if not return_draft_outline:
            return article_with_outline_only
        return article_with_outline_only, article_with_draft_outline_only


class WriteOutline(dspy.Module):
    """为维基百科页面生成大纲的模块。"""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv)
        self.engine = engine

    def forward(
        self,
        topic: str,
        dlg_history,
        old_outline: Optional[str] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        # 关键路径：过滤对话历史，去除无关的"topic you"内容
        trimmed_dlg_history = []
        for turn in dlg_history:
            if (
                "topic you" in turn.agent_utterance.lower()
                or "topic you" in turn.user_utterance.lower()
            ):
                continue
            trimmed_dlg_history.append(turn)
        # 关键路径：将过滤后的对话转换为标准化格式
        conv = "\n".join(
            [
                f"Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}"
                for turn in trimmed_dlg_history
            ]
        )
        # 关键路径：文本预处理 - 去除引用并限制字数
        conv = ArticleTextProcessing.remove_citations(conv)
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 5000)

        with dspy.settings.context(lm=self.engine):
            # 关键路径：如果没有提供旧大纲，则先根据主题生成初始草稿大纲
            if old_outline is None:
                old_outline = ArticleTextProcessing.clean_up_outline(
                    self.draft_page_outline(topic=topic).outline
                )
                if callback_handler:
                    callback_handler.on_direct_outline_generation_end(
                        outline=old_outline
                    )
            # 关键路径：基于主题和对话历史和旧大纲生成改进的大纲
            outline = ArticleTextProcessing.clean_up_outline(
                self.write_page_outline(
                    topic=topic, old_outline=old_outline, conv=conv
                ).outline
            )
            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=outline)

        return dspy.Prediction(outline=outline, old_outline=old_outline)

    """为维基百科页面编写大纲。

    写作格式要求：
    1. 使用"#" 标题 表示章节标题，"##" 标题 表示子章节标题，"###" 标题 表示子子章节标题，以此类推。
    2. 不要包含其他信息。
    3. 不要在大纲中包含主题名称本身。
    """


class WritePageOutline(dspy.Signature):
    """Write an outline for a Wikipedia page.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    # 想要撰写的主题：
    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    # 编写维基百科页面大纲：
    outline = dspy.OutputField(prefix="Write the Wikipedia page outline:\n", format=str)


class NaiveOutlineGen(dspy.Module):
    """直接使用LLM的参数化知识生成大纲的模块。"""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline)

    def forward(self, topic: str):
        outline = self.write_outline(topic=topic).outline

        return dspy.Prediction(outline=outline)

    """改进维基百科页面的大纲。你已经有一个涵盖一般信息的草稿大纲。现在你希望基于从信息寻求对话中学到的信息来改进它，使其更具信息量。

    写作格式要求：
    1. 使用"#" 标题 表示章节标题，"##" 标题 表示子章节标题，"###" 标题 表示子子章节标题，以此类推。
    2. 不要包含其他信息。
    3. 不要在大纲中包含主题名称本身。
    """


class WritePageOutlineFromConv(dspy.Signature):
    """Improve an outline for a Wikipedia page. You already have a draft outline that covers the general information. Now you want to improve it based on the information learned from an information-seeking conversation to make it more informative.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    # 想要撰写的主题：
    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    # 对话历史记录：指在通信设备或应用程序中保存的过去的对话记录。
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    # 当前概要
    old_outline = dspy.OutputField(prefix="Current outline:\n", format=str)
    # 编写维基百科页面大纲（使用“# 标题”来表示章节标题，“## 标题”来表示子章节标题，……）
    outline = dspy.OutputField(
        prefix='Write the Wikipedia page outline (Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, ...):\n',
        format=str,
    )
