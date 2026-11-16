# Modular RAG MCP Server

> 一个可插拔、可观测的模块化 RAG（检索增强生成）服务框架，通过 MCP（Model Context Protocol）协议对外暴露工具接口，支持 Copilot / Claude 等 AI 助手直接调用。同时也是一份专为**大模型相关岗位学习与面试求职**设计的实战项目与配套教学资源。

---

## 📖 目录

- [项目概述](#-项目概述)
- [分支说明](#-分支说明)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [MCP 配置示例](#-mcp-配置示例)
- [Dashboard 使用指南](#-dashboard-使用指南)
- [运行测试](#-运行测试)
- [运行排错 FAQ](#-运行排错-faq)
- [谁适合用这个项目 & 怎么用](#-谁适合用这个项目--怎么用)
- [简历参考](#-简历参考)
- [常见问题](#-常见问题)
- [后续安排](#-后续安排)

---

## 🏗️ 项目概述

### 这个项目是什么

本项目将 RAG 面试中最常见的核心环节——**检索（Hybrid Search + Rerank）**、**多模态视觉处理（Image Captioning）**、**RAG 评估（Ragas + Custom）**、**生成（LLM Response）**——以及当下热门的应用协议 **MCP（Model Context Protocol）** 串联为一个完整的、可运行的工程项目。

**项目的一大亮点是极易适配到你自己的业务中**。得益于全链路可插拔架构，你可以快速将它结合到自己已有的项目里，无论你的背景和需求如何，都能找到适合自己的使用方式。具体的使用策略会在后文 [谁适合用这个项目 & 怎么用](#-谁适合用这个项目--怎么用) 中详细展开。

### 不只是项目，更是一整套思路

**比这个项目本身更有价值的，是它背后蕴含的一整套工程化思路**：

- 如何编写 **DEV_SPEC**（开发规格文档）来驱动开发
- 如何用 **Skill** 基于 Spec 自动完成代码编写
- 如何用 **Skill** 进行自动化测试、打包、环境配置
- 如何基于可插拔架构进行扩展（比如扩展到 Agent）

**学会了思路，你可以自己做全新的项目和扩展**。以上每一步的具体做法、设计思路，在笔记中都有对应的视频讲解，建议配合观看。

### 核心能力一览

| 模块 | 能力 | 说明 |
|------|------|------|
| **Ingestion Pipeline** | PDF → Markdown → Chunk → Transform → Embedding → Upsert | 全链路数据摄取，支持多模态图片描述（Image Captioning） |
| **Hybrid Search** | Dense (向量) + Sparse (BM25) + RRF Fusion + Rerank | 粗排召回 + 精排重排的两段式检索架构 |
| **MCP Server** | 标准 MCP 协议暴露 Tools | `query_knowledge_hub`、`list_collections`、`get_document_summary` |
| **Dashboard** | Streamlit 六页面管理平台 | 系统总览 / 数据浏览 / Ingestion 管理 / 摄取追踪 / 查询追踪 / 评估面板 |
| **Evaluation** | Ragas + Custom 评估体系 | 支持 golden test set 回归测试，拒绝"凭感觉"调优 |
| **Observability** | 全链路白盒化追踪 | Ingestion 与 Query 两条链路的每一个中间状态透明可见 |
| **Skill 驱动全流程** | 从编写到测试、打包、配置一键完成 | auto-coder / qa-tester / package / setup 等 Skill 覆盖完整开发生命周期（笔记中每个 Skill 的使用和设计思路均有讲解，请参考配套视频） |

### 技术亮点

**🔌 全链路可插拔架构**：LLM / Embedding / Reranker / Splitter / VectorStore / Evaluator 每一个核心环节均定义了抽象接口，支持"乐高积木式"替换，通过配置文件一键切换后端，零代码修改。

**🔍 混合检索 + 重排**：BM25 稀疏检索解决专有名词精确匹配 + Dense Embedding 解决同义词语义匹配，RRF 融合后可选 Cross-Encoder / LLM Rerank 精排，平衡查全率与查准率。

**🖼️ 多模态图像处理**：采用 Image-to-Text 策略，利用 Vision LLM 自动生成图片描述并缝合进 Chunk，复用纯文本 RAG 链路即可实现"搜文字出图"。

**📡 MCP 生态集成**：遵循 Model Context Protocol 标准，可直接对接 GitHub Copilot、Claude Desktop 等 MCP Client，零前端开发，一次开发处处可用。

**📊 可视化管理 + 自动化评估**：Streamlit Dashboard 提供完整的数据管理与链路追踪能力，集成 Ragas 等评估框架，建立基于数据的迭代反馈回路。

**🧪 三层测试体系**：Unit / Integration / E2E 分层测试，覆盖独立模块逻辑、模块间交互、完整链路（MCP Client / Dashboard）。

**🤖 Skill 驱动全流程**：内置 auto-coder（自动编码）、qa-tester（自动测试）、package（清理打包）、setup（一键配置）等 Agent Skill，覆盖从代码编写到测试、打包、部署的完整开发生命周期。每个 Skill 的使用方法和设计思路在笔记的项目部分均有讲解视频，可参考学习。

> 📖 详细架构设计、模块说明和任务排期请参阅 [DEV_SPEC.md](DEV_SPEC.md)

---

## 📂 分支说明

本项目提供三个分支，面向不同使用场景，请根据自身需求选择：

### `main` — 最干净的完整代码

- 始终只有 **1 个 commit**，包含项目的最新完整代码
- **适合人群**：
  - 想要快速体验项目完整功能的同学
  - 时间紧迫，想要快速拿到一个项目去面试、跳过中间开发过程的同学
  - 想要直接在该项目基础上做二次扩展的同学
- **使用方式**：克隆后直接运行 Setup Skill 即可体验

### `dev` — 保留完整开发记录

- 代码与 `main` 完全一致，但保留了完整的 commit 历史
- 记录了从零开始逐步构建的每一步过程，包含大量中间节点
- **适合人群**：想了解项目是如何一步步从零搭建起来的同学，可以通过 commit 历史回溯开发思路

### `clean-start` — 干净起点，从零开始

- 仅包含工程骨架（Agent Skills + DEV_SPEC），所有任务进度清零
- 保留了完整的 Skill 配置，可以使用 Agent 辅助开发
- **适合人群**：
  - 时间充分、想要从头开发的同学（**强烈建议**）
  - 想要体验完整工作流的同学：写 Spec → 拆任务 → 写代码 → 写测试 → 迭代优化
  - 甚至可以基于自己的理解重新设计架构，用自己的思路实现，深度理解每一个模块
  - 使用我们讲的所有对应思路（Spec 驱动开发、测试先行、可插拔架构等）来完成整个项目
- **核心理念**：整个项目的代码编写是 **让 AI 基于 DEV_SPEC 来自动完成的**，你自己不需要手写代码。AI 通过 Skill 读取 Spec 中的任务定义、架构设计和接口规范，自动生成符合规格的代码。这个思路请参考笔记对应视频讲解：**5.1 项目 Skills 使用：如何让 AI 使用 Skill 遵循 DEV_SPEC 完成代码**。

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd Modular-RAG-MCP-Server
```

### 2. 一键配置（Setup Skill）

本项目提供了 **Setup Skill** 一键完成所有环境配置，包括：Provider 选择 → API Key 配置 → 依赖安装 → 配置文件生成 → Dashboard 启动。

在 VS Code 中打开项目，通过 Copilot / Claude 对话框输入：

```
setup
```

Agent 会自动引导你完成全部配置流程。

> 💡 如果不熟悉 Skill 的使用方式，请观看配套笔记中的 **Setup Skill 使用讲解视频**。

### 3. 本地开发环境（`clean-start` 分支）

在 `clean-start` 分支上，代码由 Agent 基于 [DEV_SPEC.md](DEV_SPEC.md) 逐步生成。环境准备只需三步：

```bash
# 1. 创建虚拟环境（uv 会自动下载并选用 Python 3.12）
uv venv .venv --python 3.12

# 2. 安装项目与开发依赖（editable 安装，包名直接可用：core / libs / ingestion / mcp_server / observability）
uv pip install -e ".[dev]"

# 3. 运行入口与测试
uv run python main.py
uv run pytest -q
```

代码采用 `src/` 布局，顶层包通过 editable 安装暴露：

```
src/
├── core/          # 核心层：数据类型、配置、查询引擎、响应构建、追踪
├── libs/          # 可插拔层：LLM / Embedding / Splitter / VectorStore / Reranker / Evaluator
├── ingestion/     # 摄取层：Pipeline、切分、增强、编码、存储
├── mcp_server/    # MCP Server 层：协议处理与 Tools
└── observability/ # 可观测层：结构化日志、Dashboard、评估
```

配置文件在 `config/settings.yaml`，Prompt 模板在 `config/prompts/`，运行时数据写入 `data/`、`cache/`、`logs/`（均不纳入版本管理）。

### 4. 跑通第一次摄取与查询

配置好之后，四条命令就能把整条链路跑一遍：

```bash
# 1) 摄取文档（把 PDF 放进 data/documents/，或用 --path 指定任意目录）
uv run python scripts/ingest.py --path data/documents --collection default

# 2) 提问（--verbose 会打印 dense / sparse / 融合 / 重排各阶段的结果）
uv run python scripts/query.py --query "你的问题" --verbose

# 3) 跑一次评估（golden test set 回归，输出 hit_rate / mrr）
uv run python scripts/evaluate.py

# 4) 打开可视化面板
uv run streamlit run src/observability/dashboard/app.py
```

> ⚠️ `scripts/evaluate.py` 默认跑的是仓库自带的 `tests/fixtures/golden_test_set.json`，
> 它对应的是**固定的回归语料**（`docs/*.md`，见 `tests/e2e/test_recall.py`）。
> 你摄取自己的文档后再跑它，命中的自然是 0——这不是检索坏了，而是语料不同。
> 想评估自己的语料，就用 `--test-set` 指向按你的文档写的 golden test set；
> 只想确认评估链路能跑通，用 `--no-search`（跳过真实检索，走占位结果）。

启动 MCP Server（给 Copilot / Claude 等客户端用）：

```bash
uv run python main.py                      # 默认读 config/settings.yaml
uv run python main.py --config <你的配置>   # 换一个配置文件
uv run python main.py --mode plain          # 不走 MCP SDK，直接讲 JSON-RPC
```

> 💡 `main.py` 只在 **stdout** 上输出 MCP 协议消息，所有日志都走 **stderr**——这是 stdio 传输的要求，不要往 stdout 打印任何别的东西。

---

## 🧩 配置说明

`config/settings.yaml` 是唯一的配置来源：**切换 Provider 不需要改一行代码**，改 YAML 即可。

| 分组 | 字段 | 说明 | 默认值 |
|------|------|------|--------|
| **llm** | `provider` | `openai` / `azure` / `ollama` / `deepseek` / `qwen` / `gemini` | `openai` |
| | `model` / `deployment_name` | 模型名；Azure 用部署名 | - |
| | `azure_endpoint` / `api_version` | Azure 必填 | - |
| | `api_key` | 留空则读环境变量 `OPENAI_API_KEY` 等 | `""` |
| | `base_url` | 自建 / 本地服务地址（Ollama 填 `http://localhost:11434`） | `""` |
| | `temperature` / `max_tokens` | 生成参数 | `0.0` / `4096` |
| **embedding** | `provider` / `model` | 向量模型；**换了模型必须同时改 `dimensions`** | `openai` / `text-embedding-ada-002` |
| | `dimensions` | 向量维度，**必须与模型实际输出一致**，否则写入或检索会报错 | `1536` |
| **vision_llm** | `enabled` | 是否给图片生成描述（搜文字出图） | `false` |
| | `provider` / `model` / `max_image_size` | 视觉模型与图片边长上限 | `openai` / `gpt-4o` / `2048` |
| **vector_store** | `provider` | `chroma`（本地）/ `qdrant` / `pinecone` | `chroma` |
| | `persist_directory` | 向量库落盘目录 | `./data/db/chroma` |
| | `collection_name` | 默认集合名 | `knowledge_hub` |
| **retrieval** | `dense_top_k` / `sparse_top_k` | 两条粗排路线各取多少 | `20` / `20` |
| | `fusion_top_k` | RRF 融合后保留多少 | `10` |
| | `rrf_k` | RRF 平滑常数，越小头部权重差距越大 | `60` |
| **rerank** | `enabled` / `provider` | `none` / `cross_encoder` / `llm` | `false` / `none` |
| | `model` / `top_k` | 重排模型与精排条数 | - / `5` |
| **evaluation** | `enabled` / `provider` | `custom`（检索指标）/ `ragas`（生成质量） | `false` / `custom` |
| | `metrics` | 参与评估的指标 | `hit_rate`, `mrr` |
| **observability** | `log_level` / `trace_enabled` / `trace_file` | 日志级别与追踪文件位置 | `INFO` / `true` / `./logs/traces.jsonl` |
| | `structured_logging` | 是否输出 JSON Lines 结构化日志 | `true` |
| **ingestion** | `chunk_size` / `chunk_overlap` | 切块大小与重叠；**改了必须重新摄取** | `1000` / `200` |
| | `splitter` | `recursive` / `semantic` / `fixed_length` | `recursive` |
| | `batch_size` | 编码批大小，太大容易显存/内存爆 | `100` |
| | `chunk_refiner.use_llm` / `metadata_enricher.use_llm` | 是否用 LLM 精炼与增强元数据（更准但更慢更贵） | `false` |
| **dashboard** | `enabled` / `port` | 面板开关与端口 | `true` / `8501` |
| | `traces_dir` / `auto_refresh` / `refresh_interval` | 追踪目录与自动刷新 | `./logs` / `true` / `5` |

**几条容易踩的坑：**

- **API Key 留空即可**，代码会自动读环境变量：`OPENAI_API_KEY`、`AZURE_API_KEY`、`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`GEMINI_API_KEY` 等。这样配置文件可以安全地提交，密钥不进版本库。
- **`embedding.dimensions` 必须和模型对得上**。换了 embedding 模型却没改维度，写入和检索都会失败——这是最常见的配置错误。
- **所有相对路径都相对仓库根目录**，不是相对当前工作目录。所以从任何地方启动 Server 读写的是同一份数据。
- **改完 `chunk_size` / `chunk_overlap` / `splitter` 要重新摄取**（`--force`），否则旧的分块还在库里。

---

## 🔌 MCP 配置示例

本项目就是一个标准的 **MCP Server**（stdio 传输），任何支持 MCP 的客户端都能接。

### GitHub Copilot（VS Code）

在项目根目录建 `.vscode/mcp.json`：

```json
{
  "servers": {
    "modular-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["${workspaceFolder}/main.py"],
      "env": {
        "SMART_KNOWLEDGE_HUB_CONFIG": "${workspaceFolder}/config/settings.yaml"
      }
    }
  }
}
```

### Claude Desktop

编辑 `claude_desktop_config.json`（Windows：`%APPDATA%\Claude\claude_desktop_config.json`；macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "/absolute/path/to/MODULAR-RAG-MCP-SERVER/.venv/bin/python",
      "args": ["/absolute/path/to/MODULAR-RAG-MCP-SERVER/main.py"],
      "env": {
        "SMART_KNOWLEDGE_HUB_CONFIG": "/absolute/path/to/MODULAR-RAG-MCP-SERVER/config/settings.yaml"
      }
    }
  }
}
```

### Cursor / Claude Code 等其它客户端

原理完全一样，只是配置文件的键名不同：

| 客户端 | 配置文件 | 键名 |
|--------|----------|------|
| GitHub Copilot (VS Code) | `.vscode/mcp.json` | `servers` |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` |
| Claude Code | `~/.claude.json` 或项目内 `.mcp.json` | `mcpServers` |

**注意：**

- **路径必须是绝对路径**（Claude Desktop 尤其如此，它的工作目录不是你的项目目录）。`${workspaceFolder}` 只有 VS Code 支持。
- **`command` 要指向装了依赖的解释器**：用虚拟环境里的 `python`，而不是系统 `python`。
- **配置文件有两种指定方式**：`env` 里的 `SMART_KNOWLEDGE_HUB_CONFIG` 环境变量，或启动参数 `--config <路径>`。两者都生效，参数优先。
- **接上之后能看到三个工具**：`query_knowledge_hub`（检索问答，返回带引用的答案）、`list_collections`（列出集合）、`get_document_summary`（取单篇文档的摘要与标签）。

---

## 📊 Dashboard 使用指南

```bash
uv run streamlit run src/observability/dashboard/app.py
# 端口被占用时：uv run streamlit run src/observability/dashboard/app.py --server.port 8502
```

浏览器打开 `http://localhost:8501`。面板共六个页面（用 `st.navigation` 注册，侧边栏切换）：

| 页面 | 能看到什么 |
|------|-----------|
| **System overview** | 每个组件当前用的 Provider 与模型；集合里的文档数 / 分块数 / 图片数 |
| **Data Browser** | 摄取进来的文档，展开可看它被切成的每个块、块的元数据与图片 |
| **Ingestion** | 上传 PDF 触发摄取并看进度条；管理已有文档，可一键删除 |
| **Ingestion Traces** | 每次摄取运行的每个阶段耗时（瀑布图），定位慢在哪一步 |
| **Query Traces** | 每个问题怎么被回答的：dense 与 sparse 各自召回了什么、RRF 融合结果、重排改变了什么顺序，并可对单次回答触发 **Ragas Evaluate** |
| **Evaluation** | 选 golden test set 与评估器，一键跑评估并看 hit_rate / mrr 等指标；也能看到上一次的结果 |

界面示意（文字版，非截图）：

```
┌─ 侧边栏 ─────────┐  ┌─ System overview ───────────────────────────┐
│ ▸ Overview       │  │ Components                                  │
│ ▸ Data Browser   │  │  ┌─────────┐┌─────────┐┌─────────┐          │
│ ▸ Ingestion      │  │  │ LLM     ││Embedding││VectorSt.│  ...      │
│ ▸ Ingestion Trace│  │  │ openai  ││openai   ││chroma   │          │
│ ▸ Query Traces   │  │  └─────────┘└─────────┘└─────────┘          │
│ ▸ Evaluation     │  │ Data statistics                            │
└──────────────────┘  │  Documents 12   Chunks 348   Images 7       │
                       └────────────────────────────────────────────┘
```

> 💡 追踪数据来自 `logs/traces.jsonl`（JSON Lines），摄取与查询两条链路各写各的记录，Dashboard 直接读这个文件。

---

## 🧪 运行测试

三层测试，命令一样，只是路径和标记不同：

```bash
# 全部（约 2 分钟）
uv run pytest -q

# 按层跑
uv run pytest -q tests/unit          # 单元测试：纯逻辑，不碰网络
uv run pytest -q tests/integration   # 集成测试：模块之间、真实向量库
uv run pytest -q tests/e2e            # 端到端：MCP Client、Dashboard、摄取与检索链路

# 按标记跑（e2e 最慢，日常开发可以先跑快的）
uv run pytest -q -m "not e2e"
uv run pytest -q -m unit

# 代码风格检查（只跑 check，不做格式化）
uv run ruff check src tests conftest.py main.py
```

**不需要 API Key、不需要联网**：端到端测试用本地起的一个小服务顶替 embedding 提供方（`tests/e2e/stub_embedding_server.py`），所以 `uv run pytest -q` 在任何机器上都能跑完。

| 目录 | 测什么 |
|------|--------|
| `tests/unit` | 每个模块的边界行为：工厂注册、切分器、编码器、重排器、评估器、配置校验 |
| `tests/integration` | 模块之间：切分 → 编码 → 入库 → 检索的往返，MCP 协议处理，摄取流水线 |
| `tests/e2e` | 完整链路：以子进程启动 MCP Server 并像客户端一样调用；用 Streamlit 的 `AppTest` 跑遍六个页面；摄取后检索能否命中 golden test set |

---

## 🐛 运行排错 FAQ

**1. 报 401 / AuthenticationError，提示 API Key 无效？**

配置文件里的 `api_key` 留空是**故意的**，代码会读环境变量。先确认变量真的在当前 shell 里：

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

变量名按 Provider 走：`OPENAI_API_KEY`、`AZURE_API_KEY`、`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`GEMINI_API_KEY`、`GROQ_API_KEY`、`MISTRAL_API_KEY`、`TOGETHER_API_KEY`。注意改完环境变量要**重启** Server 或 Dashboard 才生效。

**2. `uv pip install -e ".[dev]"` 装不上依赖？**

- 网络慢：换镜像源，例如 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install -e ".[dev]"`
- 虚拟环境里没有 pip 是正常的（本项目用 uv 管理），**不要**在 venv 里跑 `pip install`，用 `uv pip install --python .venv/Scripts/python.exe -e ".[dev]"`（Windows 路径是 `.venv/Scripts/python.exe`）。
- 装完仍然 `ModuleNotFoundError`：确认用的是 venv 里的解释器（`.venv/Scripts/python.exe`），而不是系统 Python。

**3. 连接不上 Provider（Connection refused / timeout）？**

- **Ollama**：确认服务在跑（`ollama serve`），且 `base_url` 写的是 `http://localhost:11434`；模型要先 `ollama pull` 下来。
- **Azure OpenAI**：`azure_endpoint`、`api_version`、`deployment_name` 三个都要填，且 **deployment 名不等于模型名**。
- **公司代理**：设置 `HTTPS_PROXY` / `HTTP_PROXY` 环境变量后重试。

**4. 向量库目录被占用 / 报 SQLite 锁？**

Chroma 用的是嵌入式数据库，同一个 `persist_directory` 同一时刻只能有一个进程写。**关掉正在跑的 Dashboard 或 Server 再摄取**，或者给不同集合配置不同的 `persist_directory`。

**5. Dashboard 打不开或端口被占用？**

`uv run streamlit run src/observability/dashboard/app.py --server.port 8502`换一个端口。面板读的是当前工作目录下的 `config/settings.yaml`，所以**请在仓库根目录启动**。

**6. 改了 `chunk_size` 但查询结果没变？**

旧的分块还在库里。重新摄取并强制覆盖：

```bash
uv run python scripts/ingest.py --path data/documents --collection default --force
```

**7. MCP 客户端连不上 Server？**

按顺序排查：① `command` 是不是绝对路径、是不是 venv 里的解释器；② 配置文件路径对不对（`--config` 或 `SMART_KNOWLEDGE_HUB_CONFIG`）；③ **stdout 只能有协议消息**——如果你的代码往 stdout 打印了东西，客户端就会解析失败；日志请走 stderr。手动验证一下 Server 能起来：

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}' | uv run python main.py
```

**8. 摄取很慢？**

第一次摄取要把文档切片、调用 embedding、写向量库和 BM25 索引，慢是正常的。加速可以：调大 `ingestion.batch_size`、关掉 `chunk_refiner.use_llm` 与 `metadata_enricher.use_llm`、先用 `chunk_size` 大一点的配置试跑。

---

## 🎯 谁适合用这个项目 & 怎么用

大家的背景不同——有的校招、有的社招；基础也不同——有的有 AI 项目经验、有的是转方向。因此对于这个项目的使用策略也应该不同，**请一定灵活使用，切忌生搬硬套**。

不过有一点是通用的：**整套项目背后的思路**——如何写 Spec 快速拉起一个项目、如何用 Skill 驱动 AI 自动编码和测试——这些工程化方法论适用于任何项目，值得所有人参考学习。

对于项目本身在不同场景下的使用策略，我会提供一些具体的例子，并以我自己的亲身经历展开——**如果是我自己，面对不同的情况，我会怎么使用这个项目**——给大家作为参考。

### 1. 纯学习 RAG —— 把项目当作 RAG 全流程的学习材料

这个项目本身就是一个完整的 RAG 系统，可以作为学习 RAG 的配套实战项目。

我最开始学习 RAG 的时候，看的是这本书：**《大模型RAG实战：RAG原理、应用与系统构建》**（汪鹏、谷清水、卞龙鹏等人工智能领域专家编著）。你完全可以结合这本书来学习 RAG，书中涉及的典型环节——检索、生成、向量数据库、分块策略、重排序等——其实不管你看哪本 RAG 相关的书，核心内容都是这些。

**这个项目就是把这些步骤串起来了**，所以它可以作为一个通用的 RAG 全流程项目来学习整个过程。你可以配这本书，我相信你也可以配其它的 RAG 书籍，因为流程是通的。面试 RAG 其实也无非是这些过程的组合、原理，以及在实际中遇到的困难和优化。

### 2. 时间紧迫 —— 缺一个项目拿去面试

如果你现在没有 AI 相关项目、急需一个项目去面试，那么可以：

1. **直接使用本项目**，克隆 `main` 分支，用 Setup Skill 跑起来
2. **结合 Resume Writer Skill** 写自己的简历（Skill 会根据你的背景定制化生成项目描述）
3. **尝试理解项目**，跑通核心流程，结合我后续总结的该项目面试问题，先去面试
4. **随着面试深入理解、扩展项目**——面试本身就是最好的学习驱动力

比如现在 3 月份，需要找暑期实习的同学，时间紧迫——先写上去，一边面试一边学习，有时间再扩展。解决你着急面试没有项目的燃眉之急。思路就是：**先写上去 → 去面试 → 根据面试反馈改进项目**。

通常暑期实习从 3 月到 7 月都有机会。找到了实习、有了一个大模型项目经验后，再以此为跳板继续学习——7~10 月秋招，甚至到明年 3 月春招，你有大量的时间持续积累。现在开始虽然看起来有点晚，但其实不晚。如果你能保持学习节奏，从现在到明年 3 月，学习整整一年，校招上岸大模型方向绝对没有问题。**关键在于你自己能不能保持这么长时间的学习力。**

### 3. 时间相对充分 —— 以本项目为起点进行扩展

你可以把这个项目作为起点，根据自己的发展方向进行针对性扩展。DEV_SPEC 中也写了扩展方向，这里列几个常见的：

- **想补充 Agent 知识**：自己实现 Agent 端，做一些上下文处理、Tool Calling、ReAct 逻辑，把本项目作为 Agent 的一个模块和能力，变成一个 **Agent + RAG** 的项目
- **想展示后端工程能力**：加上后端部署能力，写 Dockerfile，做 CI/CD 流水线，加上监控和日志收集
- **想把 RAG 做深入**：扩展到 Agentic RAG、Graph RAG 等高级形态，或者在检索策略上做更多优化实验

每个人发展方向不一样——就像项目配套的 Resume Writer Skill，它写简历的时候会先问你的背景和情况。你定位大模型应用开发工程师、RAG 工程师、全栈工程师，校招、社招的要求都是不一样的（具体大模型不同岗位介绍和技术栈可以看笔记的**大模型岗位介绍**部分），所以需要自己进行针对性扩展。

> **强烈建议**：不管你什么背景、怎么扩展，你大概率需要结合自己的业务来写简历。所以**至少试一下**——把你自己领域的文档（金融、法律、医疗，或者你的业务文档）丢进去，看一下检索效果。如果效果不好，再去调整和改进。这个过程本身就是最好的学习，也是面试时最有说服力的实战经验。

### 4. 时间特别充分 —— 从零体验完整工作流

如果你时间充足，我建议你从 `clean-start` 分支开始，甚至在 `clean-start` 的基础上**删掉 DEV_SPEC**，从文档设计开始，一点点体验：

**文档设计 → AI 写代码 → 改进迭代 → 测试 → 部署**

整个过程的方法论。其中 DEV_SPEC 怎么写、Skill 怎么设计，这些都在笔记项目部分的对应视频中有讲解。你可以重新设计文档、改进文档，甚至直接做 Agent 方向的东西，来走完整个流程。

这样做你会学习到**开发一个项目的完整思路**。这套方法最大的好处是**下限极低**——几乎你都能设计出来、完成整个项目。这样你既学会了思路，又学会了过程，而且项目可以高度定制。群里已经有很多朋友都这么做了。

### 5. 融入现有项目 —— 把 RAG 能力集成到你已有的项目中

这其实也是一种很好的策略，我自己可能也会用这种方式。以我亲身经历现身说法：

我之前找工作时，其实已经有 2 个 Agent 项目了，只是 RAG 流程跑得很粗。我简历上大概的写法是"Agent 项目做了啥，其中涉及了一些 RAG 的知识"。面试的时候，面试官多少都会问 RAG 的内容，然后我和他讲，但因为之前项目的 RAG 系统很浅——其实就是做了一个基本的 Embedding 向量匹配，没有粗排、重排等策略——所以面试官一问就比较浅。

做了这个项目之后，一种处理方式是**把本项目的 RAG 能力融入到之前的 Agent 项目中**，在简历上不作为独立项目，而是作为 Agent 项目的一部分来描述。例如：

> *"……项目中使用自研的模块化 RAG 系统进行知识检索，采用 BM25 + Dense Embedding 混合召回并通过 RRF 融合排序，结合 Cross-Encoder 重排序提升 Top-K 精准度；支持多模态文档处理（PDF 解析 + Image Captioning），通过 MCP 协议暴露标准化工具接口供 Agent 调用。集成 Ragas 评估框架，建立 Golden Test Set 回归测试机制，持续优化检索质量……"*

这样你原有的 Agent 项目就有了 RAG 深度，面试官再问你就有东西可讲了。

### 6. 产品经理 —— 对，你没看错，PM 也可以用这个项目

大模型产品经理面试越来越多地会考 RAG 相关知识，有些公司甚至要求产品经理自己写一个 POC（Proof of Concept）再交给开发。**这个项目和背后的方法论，完全可以帮你做到这一点。**

**为什么 PM 可以用：**

1. **面试需要**：大模型产品岗会考 RAG 的基本原理和流程，你通过这个项目可以直观感受 RAG 的整个过程——从文档摄取、分块、向量化、检索、重排到最终生成，建立起产品层面的理解
2. **POC 能力**：你完全可以用这套方法构建出整个项目——写文档（DEV_SPEC），或者直接用现有的文档，然后用 Skill 让 AI 帮你生成代码。面试的时候你说的是你的思路和产品设计，代码是 AI 帮你写的，这在当下完全合理
3. **不需要关心技术细节**：产品不用关心每一行代码怎么写，但通过跑通这个流程，你能从产品层面思考痛点——比如检索不准怎么定义指标、用户体验上怎么设计反馈机制、数据质量如何影响 RAG 效果等

**具体怎么做：**
- 克隆 `main` 分支，用 Setup Skill 跑起来，体验完整流程
- 把你自己业务领域的文档丢进去，看检索效果，思考产品层面的优化方向
- 面试时讲你的产品思路和设计思考，技术实现部分说明是用 AI 辅助完成的

> 💡 笔记中也提供了 **Vibe Coding** 相关教程（如 Tina Huang 老师的讲解），非常适合非技术背景的同学参考，用 AI 快速构建原型。

### 关于"项目浅"这件事

最后我想独立提一点（这一点适用于上面所有情况）：

**所有项目的深入优化不是一步到位的。**

如果你是转行，项目都是自己做的，多少会遇到面试官觉得你项目浅的问题。我之前也提到过这一点，但不用害怕：

1. **项目深度不是入行的必要条件。** 我去年拿了 6 个 offer，其中包括大厂 offer，即使这样，还是有面试官觉得我项目浅。面试还会考虑很多其他方面——理论基础、算法能力、背景匹配、知识广度等等。不要因为觉得自己转行项目浅就觉得自己转不了。

2. **项目是在不断优化和深入的。** 面试官说你项目浅，你听他的反馈，一定能听出来他为什么觉得浅——比如觉得你数据不够复杂，那你就造一些复杂数据；觉得你图片处理太简单，那你可以扩展多模态策略。我自己也是在面试过程中不断往项目里加东西：我之前做的 Agent 项目，随着面试推进，我往里加了部署、训练、反思数据、评估模块——整个过程都是随着面试同步进行的。

**给自己预留多一点面试时间，一边面试，一边改进和加深。** 所以这里就又提到了整套项目的思路——学会这些思路，你才能持续扩展，而且扩展的门槛很低，都是想好想法让 AI 写嘛，所以不用怕。

说一个真实数据：**这个项目从立项到完成，是我下班后用了大概 2 个月时间做的**，期间我还要上班、做自媒体、自媒体也有其他内容要产出。所以我希望你不要完全指望这个项目作为一个不用扩展就特别深入的项目，特别是社招的同学。但反过来想——两个月的下班时间就做出了这些，如果你学会了这套方法，自己扩展的速度会有多快？

方法都有了，所有的方案、过程、记录都有留档和视频讲解。**最终一定要靠你自己去扩展、迭代，做成最适合你自己的项目。**

---

## 📝 简历参考

> ⚠️ **强烈建议**：请使用项目内置的 **Resume Writer Skill** 来生成你的简历项目经历，而不是直接复制下面的示例。
>
> 简历项目经历**一定是针对性的**——需要结合你自己的业务背景、目标岗位、技术侧重来定制化生成。下面的示例仅用于展示 Skill 的输出效果和不同场景的写法参考，**直接照搬没有任何意义**。
>
> **如何使用 Resume Writer Skill**：在 VS Code 中通过 Copilot / Claude 对话框输入 `写简历` 或 `resume`，Skill 会引导你完成画像采集并自动生成四段式简历。具体使用方式和设计思路请参考笔记中 **项目部分的视频讲解**。

### Resume Writer Skill 工作方式

Skill 采用 **"写作原则 + 项目亮点 + 用户画像 = 定制化简历"** 的三角模型，流程如下：

1. **画像采集**：Skill 会询问你的目标岗位（RAG Engineer / Backend / Agent 等）、业务背景、技术侧重、特殊要求
2. **亮点匹配**：根据你的岗位方向，从项目 10 大技术亮点中筛选 3-5 个最匹配的写入 bullet points
3. **四段式生成**：严格按 **背景 → 目标 → 过程 → 结果** 结构输出，每条 bullet 遵循"动词开头 + 技术细节 + 量化效果"
4. **面试追问预测**：自动生成 3-5 条面试官可能的追问，帮你提前准备

### 示例一：校招 · RAG Engineer 方向

> 以下为 Skill 基于"校招、RAG 方向、通用框架模式"生成的示例输出：

**智能知识检索与问答系统** | 2024.09 - 2025.02 | 独立设计与开发

**背景**：针对企业级知识库场景中文档分散、检索精度不足、AI 应用难以接入私有知识的共性痛点，设计并实现了模块化 RAG 检索框架。

**目标**：构建基于混合检索 + MCP 协议的智能知识问答系统，实现精准语义检索与 AI Agent 直接调用私有知识库的能力，将文档问答准确率提升至 90% 以上。

**过程**：
- 设计 BM25 + Dense Embedding 混合召回架构，通过 RRF 融合排序平衡查全率与查准率，结合 Cross-Encoder 重排序将 Top-10 命中率提升约 25%
- 构建全链路 Ingestion Pipeline（PDF 解析 → Markdown → 语义分块 → Metadata 增强 → Embedding → Upsert），集成 Vision LLM 实现图片自动描述并缝合进 Chunk，复用纯文本链路即可"搜文字出图"
- 实现 LLM / Embedding / Reranker / VectorStore 全链路可插拔架构，定义统一抽象接口，通过配置文件一键切换后端 Provider，支持 4+ LLM Provider 零代码切换
- 集成 Ragas + Custom 双评估体系，建立 Golden Test Set 回归测试机制，覆盖 Faithfulness / Relevancy / Recall 等维度，拒绝"凭感觉"调优
- 基于 Skill 驱动全流程开发，通过 auto-coder / qa-tester / setup / package 等 5 大 Agent Skill 覆盖编码、测试、配置、打包完整生命周期，2 个月业余时间完成 68 个子任务的全量交付

**结果**：系统支撑 5000+ 篇文档的实时语义检索，检索准确率（Hit Rate@10）达 92%，端到端查询延迟控制在 800ms 以内，三层测试体系（Unit / Integration / E2E）覆盖 1200+ 测试用例。

**技术栈**：Python / LangChain / ChromaDB / BM25 / Cross-Encoder / MCP Protocol / Streamlit / Ragas / Azure OpenAI

### 示例二：社招 · 已有 Agent 项目，融入 RAG 深度

> 以下为 Skill 基于"社招、Agent 方向、Windows 平台开发业务背景"生成的示例输出（将 RAG 能力融入已有 Agent 项目）：

**Windows 平台智能知识助手** | 2024.06 - 2025.02 | 核心开发

**背景**：在 Windows 平台开发团队中，版本发布相关信息（Release Notes、变更日志、补丁公告、兼容性说明等）分散于多个 Wiki、文档仓库和内部系统，工程师排查版本差异或回答客户问题时需跨系统翻找，现有关键词搜索无法理解语义，导致检索效率低、信息遗漏频发。

**目标**：为团队构建基于 Agent + RAG 架构的智能知识助手，实现跨系统文档的语义检索与自动问答，通过 MCP 协议集成至工程师日常工具链（VS Code / Claude Desktop），将文档查找时间缩短 60% 以上。

**过程**：
- 设计 Agent + RAG 分层架构，Agent 端负责意图识别与 Tool Calling，RAG 端提供 BM25 + Dense Embedding 混合召回 + Cross-Encoder 精排的两段式检索能力，通过 MCP 协议暴露标准化工具接口供 Agent 调用
- 实现全链路 Ingestion Pipeline，支持 PDF / Markdown 多格式文档解析，集成 Vision LLM 自动生成图片描述（架构图、截图等），解决"搜文字出图"的多模态检索需求
- 构建可插拔后端架构，LLM / Embedding / Reranker / VectorStore 均定义抽象接口，支持 Azure OpenAI ↔ DeepSeek ↔ Ollama 一键切换，适配团队不同网络环境
- 搭建 Streamlit Dashboard 管理平台，提供数据浏览、Ingestion 追踪、查询追踪、评估面板六大功能页，实现全链路白盒化可观测
- 集成 Ragas 评估框架 + Golden Test Set 回归测试，在版本迭代中持续监控检索质量，Faithfulness 评分稳定在 0.85 以上
- 采用 Skill 驱动全流程开发模式，编写 DEV_SPEC 规格文档驱动 auto-coder 自动编码、qa-tester 自动测试与修复、setup 一键环境配置，5 大 Agent Skill 覆盖完整开发生命周期，2 个月业余时间完成 68 个子任务交付

**结果**：系统覆盖团队 8000+ 篇技术文档，工程师日均文档查询时间从 15 分钟缩短至 3 分钟，检索准确率 Hit Rate@10 达 90%，已通过 MCP 协议接入 3 个内部 AI 工具，累计处理查询 2 万+ 次。

**技术栈**：Python / Agent / Tool Calling / RAG / BM25 / Dense Retrieval / Cross-Encoder / MCP Protocol / ChromaDB / Streamlit / Ragas / Skill-Driven Development / Azure OpenAI

### 示例三：社招 · 后端工程师转 AI 方向

> 以下为 Skill 基于"社招转 AI、后端/架构方向、金融合规业务背景"生成的示例输出：

**合规智能文档检索系统** | 2024.10 - 2025.02 | 设计与主导开发

**背景**：在某金融机构合规部门，法规文件和内部政策文档持续增长至万级规模，合规团队在审查和咨询场景中需要快速定位特定条款，但现有全文搜索系统只能精确匹配关键词，无法理解"反洗钱"与"AML"等语义近义表达，条款定位效率低下。

**目标**：设计并实现模块化 RAG 检索系统，将语义检索能力引入合规文档管理流程，支持近义词、跨语言条款匹配，目标将合规条款定位准确率提升至 90% 以上。

**过程**：
- 主导系统架构设计，采用全链路可插拔架构，LLM / Embedding / Reranker / Splitter / VectorStore 均定义抽象接口与工厂模式，通过 YAML 配置一键切换后端，零代码修改即可适配不同部署环境
- 实现 BM25 稀疏检索 + Dense Embedding 语义检索的混合召回策略，通过 RRF 融合排序兼顾专有名词精确匹配与语义近义匹配，检索准确率较纯向量方案提升 22%
- 构建完整的数据摄取管线，支持 PDF 解析 → 语义分块 → Chunk Refinement → Metadata Enrichment → 向量化存储，实现 DocumentManager 幂等管理，保证文档更新时的数据一致性
- 搭建三层测试体系（Unit / Integration / E2E），覆盖 1200+ 测试用例，集成 Ragas 评估框架建立自动化回归机制，确保迭代过程中检索质量不退化
- 基于 MCP 协议暴露标准化工具接口，支持 GitHub Copilot / Claude Desktop 等 AI 助手直接调用，实现"一次开发、多端调用"的服务化部署
- 实践 Skill 驱动全流程工程化方法，基于 DEV_SPEC 规格文档驱动 AI Agent 自动完成编码（auto-coder）、测试（qa-tester）、环境配置（setup）、清理打包（package），68 个子任务全量由 Agent 交付，开发周期压缩至 2 个月业余时间

**结果**：系统上线后支撑 12000+ 篇合规文档的实时语义检索，条款定位准确率从 68% 提升至 91%，单次查询延迟控制在 700ms，合规团队文档审查效率提升约 50%。

**技术栈**：Python / 可插拔架构 / 工厂模式 / BM25 / Dense Retrieval / RRF / Cross-Encoder / ChromaDB / MCP Protocol / Streamlit / Ragas / Skill-Driven Development / Azure OpenAI

---

> 💡 **使用提醒与重要说明**：
>
> **1. 关于放大策略**：Resume Writer Skill 中内置了我设计的**放大策略**——AI 会在合理范围内对你的项目经历进行包装和放大（例如量化指标、业务规模等）。这是我允许的，也是简历写作的正常做法。但这意味着：**生成简历后，你必须想清楚面试官可能针对每一条追问什么、你该怎么回答**。Skill 在生成简历的同时会自动给出 3-5 条面试追问预测，请认真准备这些问题。
>
> **2. 把简历当作实践清单**：简历中写到的每一个技术点，你都应该**真正去试一下**。比如简历写了"检索准确率提升 XX%"——那你就应该在自己的数据上跑一下，看看实际效果如何，过程中遇到了什么问题，你是怎么调优解决的。这些实践经验才是面试时真正有说服力的内容，也是你真正学到东西的过程。简历里没涉及到的部分（比如你没试过多模态、没跑过评估），也可以以此为契机去做代码实验。
>
> **3. 生成的是初稿，请务必结合自身情况修改**：Skill 生成的简历是**初稿**，不是终稿。你需要根据自己的实际情况进行调整——哪些技术你确实深入用过、哪些只是了解、哪些数据需要换成你自己的。简历写作有一条铁律：**写在简历上的东西，你一定要会讲**。即使某个点是放大的，你也要想清楚面试官会怎么问、你怎么自圆其说。说不清楚的东西宁可不写，写上去就要能扛住追问。
>
> **4. 方法比模板更重要**：整个简历编写的思路是我的——包括放大策略、四段式结构（背景 → 目标 → 过程 → 结果 → 技术栈）、亮点匹配逻辑等，这些都沉淀在 Resume Writer Skill 里。如果你有自己更信任的简历模板，或者你对项目做了扩展修改，完全可以去修改 Skill 本身来适配。**学会这套"用 Skill 沉淀方法论、让 AI 按规则执行"的逻辑，比简历本身更有价值**——这个思路可以复用到你未来任何项目的简历编写中。
>
> **5. 强烈建议写上 Skill 驱动全流程**：我个人的意见是，**Skill 驱动全流程开发这个闭环，适合写在任何人的简历上**。Skill 是当下非常热门的方向，已经是面试中的必考内容，很多公司内部也在研究如何用 Skill 加速项目构建。讲清楚你是如何使用 Skill 完成整个项目从编码 → 测试 → 修复 → 配置 → 打包的完整闭环，这本身就是一个比较创新和前沿的亮点，面试官会对此印象深刻。关于 Skill 相关内容在面试中怎么讲、怎么回答追问，我后面也会提供一些例子给大家参考。

---

## ❓ 常见问题

### 1. 如何切换 Provider（比如换成 Qwen / DeepSeek / Ollama）？

**非常简单——直接问 AI 帮你完成即可。**

项目从架构设计上使用了**工厂模式（Factory Pattern）**，Provider 的扩展和切换非常方便。你只需要理解内部原理就会发现：不同 API 本质上都是类似的 HTTP 请求，甚至大多数都遵循 OpenAI 的请求格式，切换起来特别容易。

**具体操作方式有两种：**

1. **使用 Setup Skill（推荐）**：运行一键 Setup Skill，AI 会主动询问你想用哪个 Provider，引导你填入 API Key，然后自动帮你完成代码适配和配置生成。
2. **直接让 AI 帮你改**：把你想切换的 Provider 告诉 AI（如 "帮我切换到 Qwen" 或 "帮我配置 DeepSeek"），AI 能根据工厂模式的架构自动完成代码编写。

> **原理说明**：项目的 `src/libs/` 下的 LLM、Embedding、Reranker 等模块都使用工厂模式，新增一个 Provider 只需要：① 新增一个 Provider 类；② 在工厂注册；③ 更新 `settings.yaml` 配置。AI 完全可以自动完成这些步骤。

### 2. 项目评估（Custom Evaluator）与 Cross-Encoder Reranker 部分

这两个模块的**框架代码已经搭好，但尚未经过完整测试**，感兴趣的同学可以自行完善：

| 模块 | 状态 | 需要做什么 |
|------|------|-----------|
| **自定义评估（Custom Evaluator）** | 框架已有，未测试 | 定义评估方法，准备对应的测试数据集 |
| **Cross-Encoder Reranker** | 框架已有，未测试 | 需要下载本地重排模型（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`） |

**这些 AI 都能帮你写出来**。把需求描述清楚，AI 可以帮你实现评估方法、准备数据、下载模型并完成集成测试。完成这些扩展对面试也是加分项，体现了你的独立扩展能力。

### 3. 项目报错 / Bug 怎么办？

**这不是一个经过广泛测试的生产级项目，而是一个面试导向的实战项目。** 遇到报错是正常的。

- **对面试的影响**：项目的 Bug 对面试几乎没有影响——面试官不会去实际运行你的项目，他们关注的是你对架构、原理和设计决策的理解。
- **如何修复**：最简单的方式是**把错误信息直接丢给 AI**，绝大多数问题 AI 都能帮你修复。
- **参考资源**：笔记中推荐的 Tina Huang 的视频里也介绍了这种用 AI 快速修复错误的方法。

### 4. 想摄取 PDF 以外的文档格式（Word / Markdown / HTML 等）怎么办？

**直接问 AI 帮你扩展即可。**

项目的 Loader 层采用了可插拔的抽象设计（`BaseLoader`），目前默认实现了 PDF Loader。如果你需要支持 Word、Markdown、HTML 等其他格式，整体架构已经设计好了扩展点，让 AI 帮你新增一个对应的 Loader 实现就可以了。

比如告诉 AI："帮我新增一个 Word 文档的 Loader，参考现有的 PDF Loader 实现"，AI 完全可以搞定。

### 5. 如何集成到 AI 工具中（Copilot / Cursor / Claude Code 等）？

本项目是一个 **MCP Server**，可以集成到任何支持 MCP 协议的 AI 工具和 Agent 中。我的演示中已经集成到了 **GitHub Copilot** 和 **Cursor** 中，你同样可以集成到 **Claude Code** 或其他支持 MCP 框架的工具。

**如何集成？非常简单——问 AI。**

本质上就是给不同的工具写一个 MCP 的配置文件：
- **Copilot（VS Code）**：让 AI 帮你生成 MCP 配置文件即可
- **Cursor**：直接导入项目，Cursor 会自动识别
- **Claude Code / 其他框架**：问 AI 怎么配置，每个工具的配置方式略有不同，但原理都一样

当然，也推荐你去理解 MCP 协议的原理——了解 Server 和 Client 之间是如何通信的、Tool 是怎么注册和调用的。这些在面试中也是加分项。

### 6. 通用建议：善用 AI

上述大多数问题（Provider 切换、模块扩展、Bug 修复、架构理解）**AI 都能解决**：

- 🔧 **代码层面**：让 AI 帮你切换 Provider、实现评估方法、修复 Bug
- 📖 **知识层面**：项目架构问题、设计模式问题，都可以问 AI 获取解释
- 🚀 **扩展层面**：想加新功能或适配新场景，描述清楚需求让 AI 帮你实现

> 多问 AI，让它指导你。这也是这个项目想要传达的核心理念之一——**学会与 AI 协作开发**。

---

## 📌 后续安排

### ✅ 会做的
- 项目相关问题的汇总与 FAQ 整理
- 面试高频问题整理与参考答案
- 技术要点讲解（RAG 核心知识、架构设计等）
- 简历包装建议与示范
- **亲身面试实践**：我会带着这个项目去面试，把遇到的问题、怎么回答，都总结到文档中
- **欢迎投稿共建**：如果你用这个项目去面试了，可以把面试录音发给我，我来帮你分析项目相关的问题并写入文档，同时也可以听一下面试整体有哪些改进建议。这样大家能共同进步，一起总结和完善这个项目的面试问答

### ❌ 不会做的
- 不会继续扩展新功能
- 不会处理 Bug Fix、设计优化等
  - 遇到 Bug 和设计上的改进点，请在自己的项目中修复和优化
  - 后续的扩展和修复一定是要靠自己的，而且**有了 AI，这些都很容易做到**
  - 这本身就是一个很好的学习和面试加分项
  - 在理解项目的基础上独立扩展，才是真正的能力体现

### 📝 个人规划说明

我后续会去学习**大模型算法、训练**方向，会把一些笔记和思路总结在笔记中。因此对于这个项目，**不会无限扩展功能或修复 Bug**，但会非常乐意持续做的事情是：

- 总结这个项目在面试中遇到的问题
- 整理如何回答、如何迭代优化的思路
- 把面试问答沉淀到文档中，供大家参考

---

## 📚 配套资源

本项目配有完整的配套学习资源，包括：

- 🎬 **视频讲解**：项目架构设计、Skill 使用、DEV_SPEC 编写、开发全流程演示
- 📝 **面试笔记**：大模型方向面试准备、RAG 核心知识点整理
- ❓ **面试问题参考**：该项目在面试中遇到的真实问题与参考回答
- 📖 **八股整理**：大模型 / RAG / NLP 相关高频面试题

> 👉 **请关注小红书：[不转到大模型不改名](https://www.xiaohongshu.com) 获取以上所有资源。**
