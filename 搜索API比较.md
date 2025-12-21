# STORM 搜索平台对比

本文档介绍了 STORM 项目中支持的各个搜索平台及其特点，帮助用户选择最适合的搜索服务。

## 搜索平台概述

STORM 支持 7 种主流搜索平台：

1. **Bing Search** - 微软的搜索引擎
2. **You.com** - AI 驱动的搜索平台
3. **Brave Search** - 注重隐私的搜索引擎
4. **DuckDuckGo** - 隐私导向的搜索引擎
5. **Serper** - Google 搜索 API 的替代方案
6. **Tavily Search** - 专为 AI 应用设计的搜索 API
7. **SearXNG** - 开源元搜索引擎

## 详细对比

### 1. Bing Search

**特点：**
- 微软官方搜索引擎，基于 Bing 搜索结果
- 搜索结果质量高，索引覆盖广泛
- 支持多种语言和地区
- 提供丰富的搜索结果类型（网页、图片、视频等）
- API 响应速度较快，稳定性好

**优势：**
- ✅ 搜索结果质量优秀
- ✅ API 文档完善，易于集成
- ✅ 微软官方支持，维护稳定
- ✅ 支持高级搜索语法

**劣势：**
- ❌ 需要 Bing Search API Key
- ❌ 有 API 调用限制和配额
- ❌ 隐私保护相对较弱

**适用场景：**
- 高质量搜索结果需求
- 商业项目中的搜索功能
- 需要稳定可靠的搜索服务

---

### 2. You.com

**特点：**
- AI 驱动的搜索引擎
- 结合了传统搜索和 AI 能力
- 提供个性化的搜索结果
- 支持对话式搜索

**优势：**
- ✅ AI 增强的搜索体验
- ✅ 个性化推荐能力强
- ✅ 支持多种搜索模式
- ✅ 搜索结果经过 AI 优化

**劣势：**
- ❌ 需要 YDC API Key
- ❌ 相对较新的平台
- ❌ API 生态还在发展中

**适用场景：**
- AI 增强的搜索需求
- 需要个性化搜索结果
- 对话式搜索应用

---

### 3. Brave Search

**特点：**
- 注重隐私保护的搜索引擎
- 不跟踪用户行为
- 独立索引，摆脱大厂依赖
- 基于 Chromium 浏览器生态

**优势：**
- ✅ 强隐私保护，无用户追踪
- ✅ 独立索引，结果公正
- ✅ 开源透明
- ✅ 搜索结果质量持续提升

**劣势：**
- ❌ 需要 Brave Search API Key
- ❌ 索引规模相对较小
- ❌ 某些地区访问可能受限

**适用场景：**
- 隐私优先的应用
- 注重数据保护的项目
- 开源友好的解决方案

---

### 4. DuckDuckGo

**特点：**
- 隐私导向的搜索引擎
- 不收集或存储用户信息
- 提供简洁的搜索结果
- 支持"零点击"信息获取

**优势：**
- ✅ 完全免费使用
- ✅ 强隐私保护
- ✅ 无需 API Key
- ✅ 简单的 API 调用方式

**劣势：**
- ❌ 搜索结果相对简洁
- ❌ 高级搜索功能有限
- ❌ 某些查询结果可能不够全面

**适用场景：**
- 快速简单的信息检索
- 隐私保护要求高的项目
- 预算有限的个人项目

---

### 5. Serper

**特点：**
- Google 搜索的 API 包装器
- 提供类似 Google 的搜索结果
- 简单易用的 API 接口
- 专为开发者设计

**优势：**
- ✅ Google 质量搜索结果
- ✅ API 简单易用
- ✅ 响应速度快
- ✅ 文档详细

**劣势：**
- ❌ 需要 Serper API Key
- ❌ 非官方 API（第三方服务）
- ❌ 受 Google 政策影响

**适用场景：**
- 需要 Google 质量结果
- 快速集成搜索功能
- 原型开发和测试

---

### 6. Tavily Search

**特点：**
- 专为 AI 和 LLM 应用设计的搜索 API
- 提供结构化的搜索结果
- 包含原始内容（`include_raw_content=True`）
- 优化用于 AI 处理

**优势：**
- ✅ 专为 AI 应用优化
- ✅ 提供结构化数据
- ✅ 包含原始网页内容
- ✅ 适合研究和信息提取

**劣势：**
- ❌ 需要 Tavily API Key
- ❌ 相对较新，生态较小
- ❌ 服务稳定性有待验证

**适用场景：**
- AI/LLM 应用开发
- 研究和信息收集
- 需要原始内容的场景

---

### 7. SearXNG

**特点：**
- 开源元搜索引擎
- 聚合多个搜索引擎结果
- 完全可自部署
- 隐私保护能力强

**优势：**
- ✅ 完全开源，可自部署
- ✅ 聚合多个搜索引擎
- ✅ 强隐私保护
- ✅ 可自定义搜索策略

**劣势：**
- ❌ 需要 SearXNG API Key 或自建服务
- ❌ 配置复杂度较高
- ❌ 依赖第三方搜索引擎

**适用场景：**
- 开源项目
- 需要自部署解决方案
- 隐私优先的企业环境

## 选择建议

### 按需求选择

| 需求 | 推荐平台 | 原因 |
|------|----------|------|
| **高质量搜索结果** | Bing Search, Serper | 结果质量优秀，API 稳定 |
| **隐私保护** | Brave Search, DuckDuckGo, SearXNG | 不追踪用户，隐私优先 |
| **免费使用** | DuckDuckGo | 无需 API Key，完全免费 |
| **AI 应用开发** | Tavily Search, You.com | 专为 AI 优化，支持结构化结果 |
| **快速集成** | DuckDuckGo, Serper | API 简单，文档完善 |
| **企业项目** | Bing Search, SearXNG | 稳定可靠，支持自部署 |

### 按预算选择

- **零成本**: DuckDuckGo
- **低成本**: DuckDuckGo, SearXNG（自部署）
- **中等成本**: Serper, Tavily Search
- **高成本**: Bing Search, Brave Search, You.com

### 按开发复杂度选择

- **最简单**: DuckDuckGo（无需配置）
- **简单**: Serper, Tavily Search（标准 API）
- **中等**: Bing Search, You.com（需要配置）
- **复杂**: SearXNG（可能需要自部署）

## 环境配置

### API Key 获取

各平台 API Key 获取方式：

1. **Bing Search**: [Azure Portal](https://portal.azure.com/)
2. **You.com**: [You.com API](https://you.com/api)
3. **Brave Search**: [Brave Search API](https://api.search.brave.com/)
4. **Serper**: [Serper API](https://serper.dev/)
5. **Tavily Search**: [Tavily API](https://tavily.com/)
6. **SearXNG**: [SearXNG GitHub](https://github.com/searxng/searxng)

### 配置文件

在项目根目录创建 `secrets.toml`：

```toml
BING_SEARCH_API_KEY="your_bing_key"
YDC_API_KEY="your_youcom_key"
BRAVE_API_KEY="your_brave_key"
SERPER_API_KEY="your_serper_key"
TAVILY_API_KEY="your_tavily_key"
SEARXNG_API_KEY="your_searxng_key"
```

## 总结

每个搜索平台都有其独特的优势：

- **DuckDuckGo** 是零配置、免费使用的首选
- **Bing Search** 提供最稳定的商业级搜索体验
- **Brave Search** 是隐私保护的最佳选择
- **Tavily Search** 是 AI 应用开发的专用工具
- **Serper** 是快速获取 Google 质量结果的简单方案
- **You.com** 为 AI 驱动的搜索提供了新选择
- **SearXNG** 为开源和自部署需求提供了灵活方案

选择时请根据具体需求、预算和技术栈来综合考虑。