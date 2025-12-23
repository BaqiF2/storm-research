#### 🎯 这个阶段在做什么？（核心业务）

简单说：这是一个"文章写手"在工作
- 就像根据编辑提供的章节目录和采访资料，撰写完整的文章
- 它基于前面两个阶段的成果：信息表 + 大纲
- 产出：一篇结构完整、内容丰富的维基百科式文章

#### 📞 完整调用链条解析

第一层：总指挥（STORM.run方法）

这就像项目经理决定："现在进入第三阶段"
if do_generate_article:
    draft_article = self.run_article_generation_module(...)
作用：检查是否需要执行文章生成阶段

第二层：资料管理员（_load_information_table_from_local_fs）

如果没有执行知识策展阶段：
"之前的采访记录在哪里？去把资料库搬出来"
if information_table is None:
    information_table = self._load_information_table_from_local_fs(
        os.path.join(self.article_output_dir, "conversation_log.json")
    )

这个步骤的作用：
- 检查本地是否有conversation_log.json文件
- 如果没有，报错提示需要先执行--do-research
- 如果有，从文件重建StormInformationTable对象

第三层：大纲管理员（_load_outline_from_local_fs）

如果没有执行大纲生成阶段：

"目录设计图在哪里？去把结构图找出来"
if outline is None:
    outline = self._load_outline_from_local_fs(
        topic=topic,
        outline_local_path=os.path.join(
            self.article_output_dir, "storm_gen_outline.txt"
        ),
    )

这个步骤的作用：
- 检查本地是否有storm_gen_outline.txt文件
- 如果没有，报错提示需要先执行--do-generate-outline
- 如果有，从文件重建StormArticle对象（包含大纲结构）

第四层：文章生成专家（run_article_generation_module）

"根据资料和目录，开始写文章"
draft_article = self.run_article_generation_module(
    outline=outline,
    information_table=information_table,
    callback_handler=callback_handler,
)

这个方法调用StormArticleGenerationModule模块执行核心逻辑：

步骤1：准备信息表
- 调用information_table.prepare_table_for_retrieval()
- 将收集的信息转换为可检索的格式

步骤2：获取章节列表
- 从大纲中提取一级章节：sections_to_write = article_with_outline.get_first_level_section_names()

步骤3：并行生成章节（关键优化点）
- 使用ThreadPoolExecutor创建线程池
- 为每个章节分配独立的生成任务
- 跳过特殊章节：introduction（通常自动生成）和conclusion/summary

步骤4：构建章节查询
- 基于大纲结构生成查询关键词
- 使用两种格式：有标签（用于构建大纲）和无标签（用于检索）

步骤5：收集并行结果
- 使用as_completed()等待所有任务完成
- 按完成顺序收集各章节的生成结果

步骤6：整合文章内容
- 深拷贝原始大纲对象
- 逐个更新各章节的内容
- 整合引用信息和章节内容

步骤7：后处理
- 调用article.post_processing()
- 清理格式、统一引用样式、检查完整性

第五层：并行生成引擎（ConvToSection）

每个章节的生成过程：

步骤1：信息检索
- 根据章节查询从信息表检索相关内容
- 限制返回数量：retrieve_top_k（默认5条）

步骤2：信息格式化
- 为每个信息片段添加编号[1], [2], [3]...
- 限制总长度：1500字符（保留换行符）

步骤3：语言模型生成
- 使用DSPy框架调用大语言模型
- 基于提示模板WriteSection生成章节内容
- 应用格式化要求：# 标题、## 小节标题、内联引用[1][2]

步骤4：内容清理
- 调用ArticleTextProcessing.clean_up_section()
- 移除多余空格、统一格式、标准化引用

第六层：回调通知系统

"报告总部：文章已生成完成！"
callback_handler.on_article_generation_end(...)
- 通知前端界面：文章已生成
- 提供进度反馈
- 传递生成的完整文章内容用于实时显示

#### 🔍 核心业务逻辑解析

为什么需要并行生成？

性能优化：
- 每个章节可以独立生成
- 使用多线程充分利用CPU资源
- 显著减少总体生成时间（从线性时间到近似并行时间）

为什么跳过introduction和conclusion？

智能处理：
- introduction：通常基于整体内容自动生成，不需要单独处理
- conclusion/summary：可以在文章完成后统一生成，保持一致性

信息整合策略

"如何将收集的信息转化为文章内容？"
1. 检索相关：根据章节查询从信息表检索相关内容
2. 格式化：为每个信息片段编号，建立引用关系
3. 生成：基于格式化的信息和章节要求生成内容
4. 整合：将生成的内容和引用信息整合到文章对象中

#### 🎨 整体业务逻辑（用比喻）

就像写一本书的过程：

1. 总编（STORM.run）：决定"现在开始撰写正文"
2. 资料员（_load_information_table）：从仓库搬出之前收集的所有材料
3. 档案员（_load_outline_from_local_fs）：从档案室取出章节目录设计图
4. 写作团队（ThreadPoolExecutor）：分配多个写手同时工作
   - 每个写手负责一个章节
   - 大家基于同一份材料和目录各自撰写
5. 编辑（ConvToSection）：每个写手的工作流程
   - 阅读相关材料
   - 撰写章节内容
   - 标注引用来源
6. 统筹（run_article_generation_module）：协调所有写手的工作
   - 收集各章节稿件
   - 整合成完整书籍
   - 统一格式和风格
7. 秘书（callback_handler）：通知大家"书稿完成"

最终产出：
- 一个完整的StormArticle对象（包含：主题、大纲、各章节内容、引用列表）
- 为第四阶段（文章润色）提供完整的待优化内容
- 支持断点续传：可以跳过前置阶段，直接从保存的文件加载中间结果
