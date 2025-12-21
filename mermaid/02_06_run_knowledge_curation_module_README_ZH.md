  🎯 这个方法在做什么？（核心业务）

  简单说：这是一个"AI记者"在工作
  - 就像一个记者要写一篇深度报道，需要先广泛收集资料
  - 它不是简单搜索，而是模拟多个不同视角的专家进行对话，从每个角度深入挖掘信息

  📞 完整调用链条解析

  第一层：总指挥 (run_knowledge_curation_module)

  # 这就像编辑部派记者出去采访
  information_table, conversation_log = self.storm_knowledge_curation_module.research(...)
  作用：调用知识策展模块，开始收集资料

  第二层：采访总监 (research())

  这个方法扮演采访总监的角色：

  1. 🎭 组建采访团队（生成角色）
    - 如果不禁用多视角：调用persona_generator生成多个不同专业背景的角色
    - 例如：要写"人工智能"文章，可能生成：技术专家、教育工作者、伦理学家、投资人等视角
  2. 📋 并发派记者出去采访（_run_conversation）
  # 记者们同时出发，各自采访
  conversations = self._run_conversation(...)
    - 每个角色派一个记者去深入采访
    - 多线程并发：假设有5个角色，5个记者同时工作

  第三层：记者深入采访 (ConvSimulator.forward)

  每个记者的工作流程（多轮对话）：
  for _ in range(max_turn):  # 最多对话10-20轮
      问题 = WikiWriter生成问题()     # 记者提问
      答案 = TopicExpert回答问题()    # 专家回答
      记录对话()                      # 记录这一轮

  🔍 每轮对话包含两个关键动作：

  动作1：记者提问 (WikiWriter.forward)
  # 记者根据对话历史生成下一个问题
  conv = 准备对话历史(保留最近4轮完整内容, 早期对话省略答案, 限制2500词)
  question = 生成问题(topic, persona, conv)
  - 记者会回顾之前的对话，避免问重复问题
  - 根据自己的"角色"（视角）问专业问题

  动作2：专家回答 (TopicExpert.forward)
  # 专家需要先搜索资料再回答
  1. 查询生成器将问题拆分成搜索关键词
  2. 搜索引擎查找相关资料
  3. 答案生成器基于搜索结果生成详细回答

  第四层：专家的搜索回答流程

  Step 1：问题拆解 (QuestionToQuery)
  # "人工智能如何改变教育？"
  # 拆解为多个搜索查询：
  # - "AI education applications"
  # - "artificial intelligence classroom technology"
  # - "machine learning personalized learning"

  Step 2：信息检索 (Retriever.retrieve)
  - 调用Bing/Google等搜索API
  - 返回top-k个相关网页
  - 排除ground_truth_url：避免直接抄袭参考文章

  Step 3：答案生成 (AnswerQuestion)
  # 基于搜索到的多个信息源，生成综合答案
  answer = 生成答案(topic, question, 搜索结果)
  # 确保每句话都有依据，不胡编乱造

  第五层：信息整合

  记者采访回来后：
  1. 清理对话：去掉引用标记等格式
  2. 构建信息表：
  information_table = StormInformationTable(conversations)
    - 把所有对话内容整理成结构化数据
    - 每个信息都有来源URL，方便后续引用
  3. 保存档案：
    - conversation_log.json：完整对话记录（便于分析AI是怎么想的）
    - raw_search_results.json：搜索到的原始资料

  🎨 整体业务逻辑（用比喻）

  就像一个团队写研究报告：
  1. 主编（run_knowledge_curation）：布置任务，说"我们要写关于X的文章"
  2. 采访总监（research）：组建多个记者团队，说"从技术、商业、社会等角度去采访"
  3. 专业记者（ConvSimulator）：每个记者模拟特定角色深入采访
  4. 记者提问（WikiWriter）：根据对话历史问出好问题
  5. 专家搜索（TopicExpert）：专家会先查资料再回答，确保准确

  最终产出：
  - 一份结构化的信息表（包含所有收集到的知识和来源）
  - 一份对话日志（记录整个思考过程）