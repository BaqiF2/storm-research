import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict

import dspy

from .callback import BaseCallbackHandler
from .persona_generator import StormPersonaGenerator
from .storm_dataclass import DialogueTurn, StormInformationTable
from ...interface import KnowledgeCurationModule, Retriever, Information
from ...utils import ArticleTextProcessing

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx

    streamlit_connection = True
except ImportError as err:
    streamlit_connection = False

script_dir = os.path.dirname(os.path.abspath(__file__))


class ConvSimulator(dspy.Module):
    """模拟具有特定人设的维基百科作者与专家之间的对话。"""

    def __init__(
        self,
        topic_expert_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retriever: Retriever,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_turn: int,
    ):
        super().__init__()
        # 初始化维基百科作者（负责提问）
        self.wiki_writer = WikiWriter(engine=question_asker_engine)
        # 初始化主题专家（负责回答问题）
        self.topic_expert = TopicExpert(
            engine=topic_expert_engine,
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=retriever,
        )
        # 设置最大对话轮数
        self.max_turn = max_turn

    def forward(
        self,
        topic: str,
        persona: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
    ):
        """
        执行对话模拟的主流程。

        参数:
            topic: 要研究的主题
            persona: 维基百科作者的人设
            ground_truth_url: 真实答案的URL，将从搜索中排除以避免在评估中泄露真实答案
        """
        # 初始化对话历史记录
        dlg_history: List[DialogueTurn] = []
        # 开始多轮对话
        for _ in range(self.max_turn):
            # 维基百科作者根据主题、人设和历史对话生成问题
            user_utterance = self.wiki_writer(
                topic=topic, persona=persona, dialogue_turns=dlg_history
            ).question
            # 如果生成的问题为空，记录错误并退出对话
            if user_utterance == "":
                logging.error("Simulated Wikipedia writer utterance is empty.")
                break
            # 如果作者表示感谢，说明对话结束
            if user_utterance.startswith("Thank you so much for your help!"):
                break
            # 主题专家根据问题生成回答
            expert_output = self.topic_expert(
                topic=topic, question=user_utterance, ground_truth_url=ground_truth_url
            )
            # 创建对话轮次记录
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,  # 专家的回答
                user_utterance=user_utterance,  # 作者的问题
                search_queries=expert_output.queries,  # 搜索查询
                search_results=expert_output.searched_results,  # 搜索结果
            )
            # 将对话轮次添加到历史记录
            dlg_history.append(dlg_turn)
            # 触发回调处理器，通知对话轮次结束
            callback_handler.on_dialogue_turn_end(dlg_turn=dlg_turn)

        # 返回完整的对话历史
        return dspy.Prediction(dlg_history=dlg_history)


class WikiWriter(dspy.Module):
    """在对话设置中基于视角引导的问题提问。

    提出的问题将用于启动下一轮信息搜索。
    Perspective-guided question asking in conversational setup.

    The asked question will be used to start a next round of information seeking."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """初始化WikiWriter。

        Args:
            engine: 用于问题生成的语言模型引擎
        """
        super().__init__()
        # 带角色的问题提问器
        self.ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        # 不带角色的问题提问器
        self.ask_question = dspy.ChainOfThought(AskQuestion)
        self.engine = engine

    def forward(
        self,
        topic: str,
        persona: str,
        dialogue_turns: List[DialogueTurn],
        draft_page=None,
    ):
        """
        基于对话历史和角色**生成下一个问题**。

        Args:
            topic: 讨论的主题
            persona: 提问者的角色/视角
            dialogue_turns: 对话轮次列表
            draft_page: 草稿页面(可选)

        Returns:
            包含生成问题的dspy.Prediction对象
        """
        conv = []
        # 对于较早的对话轮次,省略专家回答以节省空间
        for turn in dialogue_turns[:-4]:
            conv.append(
                f"You: {turn.user_utterance}\nExpert: Omit the answer here due to space limit."
            )
        # 保留最近4轮对话的完整内容
        for turn in dialogue_turns[-4:]:
            # 移除文本中原有的引用标记。引用标记假定为方括号内的数字格式
            conv.append(
                f"You: {turn.user_utterance}\nExpert: {ArticleTextProcessing.remove_citations(turn.agent_utterance)}"
            )
        # 将对话历史拼接成字符串
        conv = "\n".join(conv)
        conv = conv.strip() or "N/A"
        # 限制对话历史的词数,同时保留换行符
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)

        with dspy.settings.context(lm=self.engine):
            # 如果提供了角色信息,使用带角色的问题生成器
            if persona is not None and len(persona.strip()) > 0:
                question = self.ask_question_with_persona(
                    topic=topic, persona=persona, conv=conv
                ).question
            else:
                # 否则使用普通的问题生成器
                question = self.ask_question(
                    topic=topic, persona=persona, conv=conv
                ).question

        return dspy.Prediction(question=question)


# 你是一位经验丰富的维基百科撰稿人。
# 你正在与一位专家交流，以获取你想要贡献的主题的相关信息。提出好的问题以获取与该主题更相关的有用信息。
# 当你没有更多问题要问时，说“非常感谢您的帮助！”来结束对话。
# 请每次只提出一个问题，不要重复之前问过的问题。你的问题应与你想要撰写的主题相关。
class AskQuestion(dspy.Signature):
    """You are an experienced Wikipedia writer. You are chatting with an expert to get information for the topic you want to contribute. Ask good questions to get more useful information relevant to the topic.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write.
    """

    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


# 你是一位经验丰富的维基百科撰稿人，并且想要编辑某个特定页面。除了作为维基百科撰稿人的身份外，你在研究主题时也有特定的关注点。
# 现在，你正在与一位专家交流以获取信息。提出好的问题以获取更有用的信息。
# 当你没有更多问题要问时，说“非常感谢您的帮助！”来结束对话。
# 请每次只提出一个问题，不要重复你之前问过的问题。你的问题应该与你想要撰写的主题相关。
class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced Wikipedia writer and want to edit a specific page. Besides your identity as a Wikipedia writer, you have specific focus when researching the topic.
    Now, you are chatting with an expert to get information. Ask good questions to get more useful information.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write.
    """

    # 您想要撰写的主题
    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    # 你的身份除了是维基百科的撰稿人之外：
    persona = dspy.InputField(
        prefix="Your persona besides being a Wikipedia writer: ", format=str
    )
    # 对话历史记录
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


# 您想通过谷歌搜索来回答这个问题。请在搜索框中输入您要使用的查询内容。
# 请按照以下格式填写您将使用的查询：
# - 查询 1
# - 查询 2
# - ...
# - 查询 n
class QuestionToQuery(dspy.Signature):
    """You want to answer the question using Google search. What do you type in the search box?
    Write the queries you will use in the following format:
    - query 1
    - query 2
    ...
    - query n"""

    # 您正在讨论的主题是：
    topic = dspy.InputField(prefix="Topic you are discussing about: ", format=str)
    # 您想要回答的问题
    question = dspy.InputField(prefix="Question you want to answer: ", format=str)
    queries = dspy.OutputField(format=str)


# 你是一位善于有效利用信息的专家。
# 你正在与一位想要为某个你熟悉的主题撰写维基百科条目的维基百科撰稿人进行交流。
# 你已经收集到了相关的信息，现在将利用这些信息来给出回应。
# 请让你的回复尽可能详尽，确保每一句话都有所依据，都能从收集到的信息中得到支持。
# 如果[收集到的信息]与[主题]或[问题]没有直接关系，请根据现有信息提供最相关的答案。
# 如果无法制定出合适的回答，就回复“基于现有的信息，我无法回答这个问题”，并解释任何限制或不足之处。
class AnswerQuestion(dspy.Signature):
    """You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants to write a Wikipedia page on topic you know. You have gathered the related information and will now use the information to form a response.
    Make your response as informative as possible, ensuring that every sentence is supported by the gathered information. If the [gathered information] is not directly related to the [topic] or [question], provide the most relevant answer based on the available information. If no appropriate answer can be formulated, respond with, “I cannot answer this question based on the available information,” and explain any limitations or gaps.
    """

    # 您正在讨论的主题是：
    topic = dspy.InputField(prefix="Topic you are discussing about:", format=str)
    conv = dspy.InputField(prefix="Question:\n", format=str)
    # 收集到的信息
    info = dspy.InputField(prefix="Gathered information:\n", format=str)
    # 现在请给出您的回答。（请尽量使用多种不同的来源，并且不要出现幻想内容。）
    answer = dspy.OutputField(
        prefix="Now give your response. (Try to use as many different sources as possible and add do not hallucinate.)\n",
        format=str,
    )


class TopicExpert(dspy.Module):
    """使用基于搜索的检索和答案生成来回答问题。此模块执行以下步骤:
    1. 从问题生成查询语句。
    2. 使用查询语句搜索信息。
    3. 过滤掉不可靠的来源。
    4. 使用检索到的信息生成答案。
    """

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries: int,
        search_top_k: int,
        retriever: Retriever,
    ):
        """
        初始化TopicExpert。

        Args:
            engine: 用于生成查询和答案的语言模型引擎
            max_search_queries: 每个问题最多生成的搜索查询数量
            search_top_k: 每次搜索返回的top-k结果数
            retriever: 用于检索信息的检索器
        """
        super().__init__()
        # 查询生成器:将问题转换为搜索查询
        self.generate_queries = dspy.Predict(QuestionToQuery)
        # 搜索器
        self.retriever = retriever
        # 答案生成器:基于检索到的信息生成答案
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, topic: str, question: str, ground_truth_url: str):
        """
        基于搜索检索和信息整合生成问题的答案。

        Args:
            topic: 讨论的主题
            question: 需要回答的问题
            ground_truth_url: 需要排除的真实URL(避免循环引用)

        Returns:
            包含查询列表、搜索结果和生成答案的dspy.Prediction对象
        """
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            # 识别阶段:将问题分解为多个搜索查询
            queries = self.generate_queries(topic=topic, question=question).queries
            # 清理查询文本:移除连字符、引号等特殊字符
            queries = [
                q.replace("-", "").strip().strip('"').strip('"').strip()
                for q in queries.split("\n")
            ]
            # 限制查询数量不超过最大值
            queries = queries[: self.max_search_queries]
            # 搜索阶段:使用查询检索相关信息
            # Search
            searched_results: List[Information] = self.retriever.retrieve(
                list(set(queries)), exclude_urls=[ground_truth_url]
            )
            if len(searched_results) > 0:
                # 评估阶段:简化处理,直接使用每个结果的top-1片段
                info = ""
                # 遍历搜索结果,提取每个结果的第一个片段
                for n, r in enumerate(searched_results):
                    info += "\n".join(f"[{n + 1}]: {s}" for s in r.snippets[:1])
                    info += "\n\n"

                # 限制信息的词数,同时保留换行格式
                info = ArticleTextProcessing.limit_word_count_preserve_newline(
                    info, 1000
                )

                try:
                    # 使用检索到的信息生成答案
                    answer = self.answer_question(
                        topic=topic, conv=question, info=info
                    ).answer
                    # 移除不完整的句子和引用
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )
                except Exception as e:
                    logging.error(f"Error occurs when generating answer: {e}")
                    answer = "Sorry, I cannot answer this question. Please ask another question."
            else:
                # 当没有找到信息时,专家不应该产生幻觉,直接说明无法回答
                # When no information is found, the expert shouldn't hallucinate.
                answer = "Sorry, I cannot find information for this question. Please ask another question."

        return dspy.Prediction(
            queries=queries, searched_results=searched_results, answer=answer
        )


class StormKnowledgeCurationModule(KnowledgeCurationModule):
    """
    知识整理阶段的接口。给定主题，返回收集到的信息。
    The interface for knowledge curation stage. Given topic, return collected information.
    """

    def __init__(
        self,
        retriever: Retriever,
        persona_generator: Optional[StormPersonaGenerator],
        conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_conv_turn: int,
        max_thread_num: int,
    ):
        """
        存储参数并完成初始化。

        Args:
            retriever: 检索器，用于搜索相关信息
            persona_generator: 角色生成器，用于生成不同视角的角色
            conv_simulator_lm: 对话模拟器使用的语言模型
            question_asker_lm: 问题提问者使用的语言模型
            max_search_queries_per_turn: 每轮对话最大搜索查询数
            search_top_k: 搜索返回的前K个结果
            max_conv_turn: 最大对话轮数
            max_thread_num: 最大线程数
        """
        self.retriever = retriever
        self.persona_generator = persona_generator
        self.conv_simulator_lm = conv_simulator_lm
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.retriever = retriever
        self.conv_simulator = ConvSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_turn=max_conv_turn,
        )

    def _get_considered_personas(self, topic: str, max_num_persona) -> List[str]:
        """获取要考虑的角色列表"""
        return self.persona_generator.generate_persona(
            topic=topic, max_num_persona=max_num_persona
        )

    def _run_conversation(
        self,
        conv_simulator,
        topic,
        ground_truth_url,
        considered_personas,
        callback_handler: BaseCallbackHandler,
    ) -> List[Tuple[str, List[DialogueTurn]]]:
        """
        并发执行多个对话模拟，每个对话使用不同的角色，并收集它们的对话历史。
        每个对话的对话历史在存储前会被清理。

        参数:
            conv_simulator (callable): 用于模拟对话的函数。必须接受四个参数：
                `topic`、`ground_truth_url`、`persona` 和 `callback_handler`，
                并返回一个具有 `dlg_history` 属性的对象。
            topic (str): 对话模拟的主题。
            ground_truth_url (str): 与对话主题相关的真实数据的URL。
            considered_personas (list): 将用于进行对话模拟的角色列表。
                每个角色会单独传递给 `conv_simulator`。
            callback_handler (callable): 传递给 `conv_simulator` 的回调函数。
                它应该处理模拟过程中的任何回调或事件。

        返回:
            list of tuples: 一个列表，其中每个元组包含一个角色及其对应的
            已清理的对话历史（`dlg_history`）。
        """

        conversations = []

        def run_conv(persona):
            """运行单个角色的对话模拟"""
            return conv_simulator(
                topic=topic,
                ground_truth_url=ground_truth_url,
                persona=persona,
                callback_handler=callback_handler,
            )

        # 确定工作线程数：取最大线程数和角色数的最小值
        max_workers = min(self.max_thread_num, len(considered_personas))

        # 使用线程池并发执行对话模拟
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 为每个角色提交对话任务
            future_to_persona = {
                executor.submit(run_conv, persona): persona
                for persona in considered_personas
            }

            if streamlit_connection:
                # 确保与Streamlit前端连接时日志上下文正确
                for t in executor._threads:
                    add_script_run_ctx(t)

            # 收集完成的对话结果
            for future in as_completed(future_to_persona):
                persona = future_to_persona[future]
                conv = future.result()
                # 清理引用并添加到对话列表
                conversations.append(
                    (persona, ArticleTextProcessing.clean_up_citation(conv).dlg_history)
                )

        return conversations

    def research(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
        max_perspective: int = 0,
        disable_perspective: bool = True,
        return_conversation_log=False,
    ) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]:
        """
        为给定主题整理信息和知识

        Args:
            topic: 自然语言表达的感兴趣的主题
            ground_truth_url: 真实数据的URL，在搜索时会被排除以避免评估中的数据泄露
            callback_handler: 回调处理器，用于处理研究过程中的各种事件
            max_perspective: 最大视角数量（角色数）
            disable_perspective: 是否禁用多视角（如果为True，则不生成角色）
            return_conversation_log: 是否返回对话日志

        Returns:
            collected_information: StormInformationTable类型的收集到的信息
            或者如果return_conversation_log为True，返回(信息表, 对话日志字典)的元组
        """

        callback_handler.on_identify_perspective_start()
        considered_personas = []
        if disable_perspective:
            # 如果禁用多视角，使用空字符串作为默认角色
            considered_personas = [""]
        else:
            # 生成多个不同视角的角色
            considered_personas = self._get_considered_personas(
                topic=topic, max_num_persona=max_perspective
            )
        callback_handler.on_identify_perspective_end(perspectives=considered_personas)

        # 运行对话收集信息
        callback_handler.on_information_gathering_start()
        conversations = self._run_conversation(
            conv_simulator=self.conv_simulator,
            topic=topic,
            ground_truth_url=ground_truth_url,
            considered_personas=considered_personas,
            callback_handler=callback_handler,
        )

        # 构建信息表
        information_table = StormInformationTable(conversations)
        callback_handler.on_information_gathering_end()
        # 根据需要返回信息表和对话日志
        if return_conversation_log:
            return information_table, StormInformationTable.construct_log_dict(
                conversations
            )
        return information_table
