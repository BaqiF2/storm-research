🎯 这个阶段在做什么？（核心业务）

简单说：这是一个"文字编辑"在工作
- 就像对写好的文章进行最后一轮精修，添加摘要和删除冗余
- 它基于第三阶段生成的草稿文章，进行质量优化
- 产出：一篇格式规范、无重复内容、带有摘要的完整文章

📞 完整调用链条解析

第一层：总指挥（STORM.run方法）

# 这就像主编决定："现在进入最后阶段，进行文章润色"
if do_polish_article:
    self.run_article_polishing_module(...)
作用：检查是否需要执行文章润色阶段

第二层：草稿加载管理员（_load_draft_article_from_local_fs）

如果没有执行文章生成阶段：
# "之前的文章草稿在哪里？去把成品仓库搬出来"
if draft_article is None:
    draft_article_path = os.path.join(
        self.article_output_dir, "storm_gen_article.txt"
    )
    url_to_info_path = os.path.join(
        self.article_output_dir, "url_to_info.json"
    )
    draft_article = self._load_draft_article_from_local_fs(...)

这个步骤的作用：
- 检查本地是否有storm_gen_article.txt文件（文章内容）
- 检查本地是否有url_to_info.json文件（引用映射）
- 如果文件不存在，报错提示需要先执行--do-generate-article
- 如果文件存在，从文件重建完整的StormArticle对象（包含文章内容和引用信息）

第三层：文章润色专家（run_article_polishing_module）

# "对草稿进行最终润色处理"
self.run_article_polishing_module(
    draft_article=draft_article, remove_duplicate=remove_duplicate
)

这个方法调用StormArticlePolishingModule模块执行核心逻辑：

步骤1：文章文本化
- 调用draft_article.to_string()
- 将结构化的StormArticle对象转换为纯文本格式
- 方便后续语言模型处理

步骤2：双重处理策略
- 同时生成摘要部分（lead section）和处理正文内容
- 使用两个不同的语言模型：
  - write_lead_engine：专门负责摘要生成（通常使用与文章生成相同的模型）
  - polish_engine：专门负责内容润色（可以使用更小的模型以节省成本）

步骤3：摘要生成（关键功能）
- 调用WriteLeadSection Signature
- 生成原则：
  - 独立成篇：摘要应能独立概括整个文章
  - 识别主题：明确定义文章讨论的核心内容
  - 建立上下文：提供必要的背景信息
  - 解释重要性：说明主题为什么值得关注
  - 总结要点：提炼最重要的信息点
  - 限制长度：不超过4个精心组织的段落
  - 添加引用：在必要时添加内联引用[1][3]

步骤4：重复内容清理（可选）
根据remove_duplicate标志决定是否执行：
- 如果为True：调用PolishPage进行重复内容检测和删除
- 如果为False：跳过清理，保持原始内容

重复清理的具体过程：
- 检测：扫描文章找出重复的信息片段
- 删除：移除重复内容，保留最相关的一个版本
- 保留：保持所有内联引用和章节结构（#、##等标记）
- 不删除：任何非重复的内容

步骤5：内容整合
- 构建摘要格式：f"# summary\n{polish_result.lead_section}"
- 组合摘要和正文：使用双换行符("\n\n")分隔
- 解析为结构化数据：调用ArticleTextProcessing.parse_article_into_dict()
- 深拷贝原始对象：避免修改原始草稿
- 插入章节内容：调用insert_or_create_section()

步骤6：最终后处理
- 调用article.post_processing()
- 清理格式：统一空格、换行等
- 统一引用样式：确保所有引用格式一致
- 检查完整性：验证文章结构是否完整

第四层：摘要生成引擎（WriteLeadSection）

摘要部分的生成过程：

步骤1：上下文设置
- 使用write_lead_engine（通常是文章生成使用的同一个LM）
- 设置show_guidelines=False提高兼容性

步骤2：提示构建
- 输入参数：topic（主题）+ draft_page（草稿页面）
- 基于模板生成摘要内容

步骤3：输出清理
- 检测是否包含提示前缀："The lead section:"
- 如果有，提取实际摘要内容
- 移除多余空白和格式标记

第五层：重复清理引擎（PolishPage）

重复内容清理的详细过程：

步骤1：重复检测
- 分析整篇文章的内容
- 识别在多个地方出现的相同或高度相似的信息

步骤2：智能删除
- 保留首次出现的完整内容
- 删除后续的重复信息
- 保留所有相关的内联引用

步骤3：结构保护
- 保持原有的章节结构（#、##、###等）
- 保留所有内联引用格式
- 维护文章的逻辑顺序

第六层：回调通知系统

# "报告总部：文章润色完成！"
callback_handler.on_article_polishing_end(...)
- 通知前端界面：文章已润色完成
- 提供进度反馈
- 传递润色后的完整文章内容

🔍 核心业务逻辑解析

为什么需要摘要部分？

维基百科标准：
- 摘要（lead section）是维基百科文章的必需部分
- 位于文章开头，独立成篇
- 为读者提供快速了解文章核心内容的入口

信息浓缩：
- 提炼文章最重要的信息
- 帮助读者决定是否继续阅读全文
- 提高信息获取效率

为什么提供remove_duplicate选项？

性能与质量的权衡：
- True：使用额外一次LM调用清理重复内容（更高质量，但更耗时）
- False：跳过清理，直接使用原始内容（更快，但可能有重复）

应用场景：
- True：最终交付版本，需要最高质量
- False：快速预览或迭代开发阶段

断点续传机制

# "如果跳过了文章生成，从哪里找草稿？"
if draft_article is None:
    # 从storm_gen_article.txt加载文章
    # 从url_to_info.json加载引用信息

这个机制很重要：
- 支持灵活的工作流：可以只执行文章润色，不执行前面的生成
- 前提：本地已经保存了storm_gen_article.txt和url_to_info.json
- 优势：避免重复执行耗时的文章生成阶段

双重文件策略

为什么需要两个文件？
- storm_gen_article.txt：存储文章的主要文本内容
- url_to_info.json：存储引用URL到信息对象的映射关系

好处：
- 分离关注点：内容与元数据分离
- 易于调试：可以独立查看和修改
- 空间效率：避免重复存储相同信息

🎨 整体业务逻辑（用比喻）

就像出版一本书的最后环节：

1. 总编（STORM.run）：决定"现在开始最终润色"
2. 档案员（_load_draft_article_from_local_fs）：从成品仓库取出打印好的草稿
   - 一个文件是正文内容
   - 另一个文件是引用来源清单
3. 文案编辑（WriteLeadSection）：在封面内页写内容摘要
   - 独立成篇，能让读者快速了解全书内容
   - 提炼最关键的要点
   - 标注重要引用来源
4. 校对员（PolishPage）：仔细检查全书删除重复内容
   - 确保没有重复段落
   - 保留所有引用标记
   - 保持章节结构完整
5. 排版师（polish_article）：整合摘要和正文
   - 将摘要放在开头
   - 统一格式和引用样式
   - 最终质量检查
6. 秘书（callback_handler）：通知大家"书籍已准备好出版"

最终产出：
- 一个完整的StormArticle对象（包含：摘要、润色后的章节内容、引用列表）
- 符合维基百科格式标准的文章
- 支持断点续传：可以跳过前置阶段，直接从保存的文件加载中间结果
- 可选的重复内容清理，提高文章质量
