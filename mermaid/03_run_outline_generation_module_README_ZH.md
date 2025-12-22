  🎯 这个阶段在做什么？（核心业务）

  简单说：这是一个"文章架构师"在工作
  - 就像写书前，编辑会先搭建章节目录结构
  - 它基于前面收集的大量资料，提炼出清晰的逻辑脉络
  - 产出：一个有层次、有逻辑的文章大纲（就像建房子的框架）

  📞 完整调用链条解析

  第一层：总指挥（STORM.run方法）

  # 这就像项目经理决定："现在进入第二阶段"
  if do_generate_outline:
      outline = self.run_outline_generation_module(...)
    作用：检查是否需要执行大纲生成阶段

  第二层：资料管理员（_load_information_table_from_local_fs）

  如果没有执行知识策展阶段：
  # "资料仓库在哪里？去把之前的采访资料搬出来"
  information_table = self._load_information_table_from_local_fs(
      os.path.join(self.article_output_dir, "conversation_log.json")
  )

  这个步骤的作用：
  - 检查本地是否有conversation_log.json文件
  - 如果没有，报错提示需要先执行--do-research
  - 如果有，从文件重建StormInformationTable对象

  第三层：大纲生成专家（run_outline_generation_module）

  # "根据这些资料，设计文章结构"
  outline, draft_outline = self.storm_outline_generation_module.generate_outline(
      topic=self.topic,
      information_table=information_table,
      return_draft_outline=True,  # 同时返回草稿大纲
      callback_handler=callback_handler,
  )

  这个方法调用StormOutlineGenerationModule模块执行核心逻辑：

  步骤1：合并对话历史
  - 将所有角色的对话内容拼接成完整的对话历史
  - 为后续生成提供完整的上下文

  步骤2：生成两个版本的大纲
  - 草稿版本：直接基于主题生成初步大纲
  - 优化版本：结合对话历史信息，优化大纲结构

  步骤3：构建文章对象
  - 将文本大纲转换为结构化的StormArticle对象
  - 方便后续文章生成阶段使用

  第四层：文件存储系统

  # "把设计图纸（大纲）保存起来"
  outline.dump_outline_to_file(
      os.path.join(self.article_output_dir, "storm_gen_outline.txt")
  )
  draft_outline.dump_outline_to_file(
      os.path.join(self.article_output_dir, "storm_gen_outline_draft.txt")
  )

  保存两个文件：
  - storm_gen_outline.txt：优化后的大纲（主要使用）
  - storm_gen_outline_draft.txt：原始草稿（对比参考）

  第五层：回调通知系统

  # "报告总部：第二阶段完成！"
  callback_handler.on_outline_generation_end(...)
  - 通知前端界面：大纲已生成
  - 提供进度反馈
  - 传递生成的大纲内容用于实时显示

  🔍 核心业务逻辑解析

  为什么需要两个版本的大纲？

  1. 草稿大纲（draft_outline）
     - 生成方式：直接根据主题生成，没有利用对话信息
     - 作用：作为对照组，验证STORM多角色对话的价值
     - 特点：结构可能较简单，信息可能不够深入

  2. 优化大纲（outline）
     - 生成方式：结合对话历史中的深度信息
     - 作用：作为正式的文章结构指导
     - 特点：更全面、更深入、更有针对性

  断点续传机制

  # "如果跳过了知识策展，从哪里找资料？"
  if information_table is None:
      information_table = self._load_information_table_from_local_fs(...)

  这个机制很重要：
  - 支持灵活的工作流：可以只执行大纲生成，不执行前面的策展
  - 前提：本地已经保存了conversation_log.json文件
  - 优势：避免重复执行耗时的知识策展阶段

  🎨 整体业务逻辑（用比喻）

  就像写一本书的过程：

  1. 总编（STORM.run）：决定"现在开始设计目录"
  2. 资料员（_load_information_table）：从仓库搬出之前收集的所有材料
  3. 架构师（run_outline_generation_module）：仔细阅读材料，设计章节目录
  4. 设计师（StormOutlineGenerationModule）：画两张设计图
     - 一张是灵感突现的草图
     - 一张是深思熟虑的正式图
  5. 档案员（dump_outline_to_file）：把设计图存档，方便后续使用
  6. 秘书（callback_handler）：通知大家"目录设计完成"

  最终产出：
  - 一个结构化的文章大纲（StormArticle对象）
  - 两个大纲文件（优化版和草稿版）
  - 为第三阶段（文章生成）提供清晰的结构指导
