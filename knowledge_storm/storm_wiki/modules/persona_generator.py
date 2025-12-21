import logging
import os
import re
from typing import Union, List

import dspy
import requests
from bs4 import BeautifulSoup


def get_wiki_page_title_and_toc(url):
    """Get the main title and table of contents from an url of a Wikipedia page."""

    proxies = {}
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

    # 添加 headers 模拟真实浏览器访问
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    response = requests.get(
        url, headers=headers, proxies=proxies if proxies else None, timeout=10
    )
    response.raise_for_status()  # 检查 HTTP 状态码
    soup = BeautifulSoup(response.content, "html.parser")

    # Get the main title from the first h1 tag
    h1_tag = soup.find("h1")
    if h1_tag is None:
        # 记录更详细的调试信息
        logging.warning(
            f"No h1 tag found for {url}. Response status: {response.status_code}. Content preview: {str(response.content[:200])}"
        )
        raise ValueError(
            f"No h1 tag found in the page, possibly failed to fetch content from {url}"
        )

    main_title = h1_tag.text.replace("[edit]", "").strip().replace("\xa0", " ")

    toc = ""
    levels = []
    excluded_sections = {
        "Contents",
        "See also",
        "Notes",
        "References",
        "External links",
    }

    # Start processing from h2 to exclude the main title from TOC
    for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        level = int(
            header.name[1]
        )  # Extract the numeric part of the header tag (e.g., '2' from 'h2')
        section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")
        if section_title in excluded_sections:
            continue

        while levels and level <= levels[-1]:
            levels.pop()
        levels.append(level)

        indentation = "  " * (len(levels) - 1)
        toc += f"{indentation}{section_title}\n"

    return main_title, toc.strip()
    #
    # 我正在为下面提到的主题撰写维基百科页面。请识别并推荐一些密切相关主题的维基百科页面。
    # 我正在寻找能够提供与该主题相关的有趣方面见解的示例，或者帮助我理解类似主题的维基百科页面中
    # 通常包含的典型内容和结构的示例。请在单独的行中列出URL。
class FindRelatedTopic(dspy.Signature):
    """I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.
    Please list the urls in separate lines."""
    topic = dspy.InputField(prefix="Topic of interest:", format=str)  # 感兴趣的主题
    related_topics = dspy.OutputField(format=str)  # 相关主题列表
#
#     您需要选择一组维基百科编辑者，他们将共同合作创建一篇关于该主题的综合性文章。
#     他们每个人都代表与该主题相关的不同视角、角色或关联。您可以使用其他相关主题的维基百科页面
#     作为灵感来源。对于每个编辑者，添加他们将关注的内容的描述。
#     请按以下格式给出答案：1. 编辑者1的简短摘要：描述\n2. 编辑者2的简短摘要：描述\n...
class GenPersona(dspy.Signature):
    """You need to select a group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic. You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.
    Give your answer in the following format: 1. short summary of editor 1: description\n2. short summary of editor 2: description\n...
    """
    # 感兴趣的主题
    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    # 相关主题的维基页面概述，以供获取灵感：
    examples = dspy.InputField(
        prefix="Wiki page outlines of related topics for inspiration:\n", format=str
    )
    personas = dspy.OutputField(format=str)  # 生成的角色列表


class CreateWriterWithPersona(dspy.Module):
    """
    通过阅读相关主题的维基百科页面来发现研究主题的不同视角。
    Discover different perspectives of researching the topic by reading Wikipedia pages of related topics.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """
        初始化创建带有角色的撰写者模块。

        Args:
            engine: 用于生成内容的语言模型引擎
        """
        super().__init__()
        # 查找相关主题的模块
        self.find_related_topic = dspy.ChainOfThought(FindRelatedTopic)
        # 生成角色的模块
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.engine = engine

    def forward(self, topic: str, draft=None):
        """
        根据主题生成不同的撰写者角色。

        Args:
            topic: 要研究的主题
            draft: 可选的草稿内容

        Returns:
            dspy.Prediction: 包含生成的角色列表、原始角色输出和相关主题的预测结果
        """
        with dspy.settings.context(lm=self.engine):
            # 从相关主题的维基页面获取章节名称作为灵感来源
            related_topics = self.find_related_topic(topic=topic).related_topics
            # 提取URL列表
            urls = []
            for s in related_topics.split("\n"):
                if "http" in s:
                    urls.append(s[s.find("http") :])
            # 从维基页面收集示例（标题和目录）
            examples = []
            for url in urls:
                try:
                    title, toc = get_wiki_page_title_and_toc(url)
                    examples.append(f"Title: {title}\nTable of Contents: {toc}")
                except Exception as e:
                    logging.error(f"Error occurs when processing {url}: {e}")
                    continue
            # 如果没有找到示例，使用占位符
            if len(examples) == 0:
                examples.append("N/A")
            # 基于示例生成角色
            gen_persona_output = self.gen_persona(
                topic=topic, examples="\n----------\n".join(examples)
            ).personas

        # 解析生成的角色列表
        personas = []
        for s in gen_persona_output.split("\n"):
            # 匹配格式如 "1. 角色描述" 的内容
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                personas.append(match.group(1))

        sorted_personas = personas

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=sorted_personas,
            related_topics=related_topics,
        )


class StormPersonaGenerator:
    """
    基于给定主题创建角色的生成器类。
    A generator class for creating personas based on a given topic.

    该类使用底层引擎生成针对指定主题定制的角色。
    生成器与 `CreateWriterWithPersona` 实例集成，以创建多样化的角色，
    包括默认的"基础事实撰写者"角色。
    This class uses an underlying engine to generate personas tailored to the specified topic.
    The generator integrates with a `CreateWriterWithPersona` instance to create diverse personas,
    including a default 'Basic fact writer' persona.

    属性 (Attributes):
        create_writer_with_persona (CreateWriterWithPersona): 负责基于提供的引擎和主题
            生成角色的实例。
            An instance responsible for generating personas based on the provided engine and topic.

    参数 (Args):
        engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): 用于生成角色的底层引擎。
            必须是 `dspy.dsp.LM` 或 `dspy.dsp.HFModel` 的实例。
            The underlying engine used for generating personas.
            It must be an instance of either `dspy.dsp.LM` or `dspy.dsp.HFModel`.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """初始化角色生成器"""
        self.create_writer_with_persona = CreateWriterWithPersona(engine=engine)

    def generate_persona(self, topic: str, max_num_persona: int = 3) -> List[str]:
        """
        基于提供的主题生成角色列表，最多生成指定数量的角色。
        Generates a list of personas based on the provided topic, up to a maximum number specified.

        该方法首先使用底层的 `create_writer_with_persona` 实例创建角色，
        然后在返回之前将默认的"基础事实撰写者"角色添加到列表开头。
        返回的角色数量限制为 `max_num_persona`，不包括默认角色。
        This method first creates personas using the underlying `create_writer_with_persona` instance
        and then prepends a default 'Basic fact writer' persona to the list before returning it.
        The number of personas returned is limited to `max_num_persona`, excluding the default persona.

        参数 (Args):
            topic (str): 要生成角色的主题。
                The topic for which personas are to be generated.
            max_num_persona (int): 要生成的最大角色数量，不包括默认的"基础事实撰写者"角色。
                The maximum number of personas to generate, excluding the default 'Basic fact writer' persona.

        返回 (Returns):
            List[str]: 角色描述列表，包括默认的"基础事实撰写者"角色，
                以及基于主题生成的最多 `max_num_persona` 个额外角色。
                A list of persona descriptions, including the default 'Basic fact writer' persona
                and up to `max_num_persona` additional personas generated based on the topic.
        """
        # 使用底层实例生成角色
        personas = self.create_writer_with_persona(topic=topic)
        # 定义默认的基础事实撰写者角色
        default_persona = "Basic fact writer: Basic fact writer focusing on broadly covering the basic facts about the topic."
        # 将默认角色添加到生成的角色列表开头，并限制总数量
        considered_personas = [default_persona] + personas.personas[:max_num_persona]
        return considered_personas
