# 🧠 LangGraph + Playwright + Streamlit Agentic Chatbot

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B6B?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Playwright](https://img.shields.io/badge/Playwright-1.45+-2EAD33?logo=playwright)](https://playwright.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-ready, portfolio-grade AI agent** showcasing modern agentic AI development. Built to impress employers with clean architecture, real tool integration (browser automation), multi-modal capabilities, smart memory management, and deployable Streamlit interface.

---

## ✨ Key Features (What Makes This Stand Out)

- **LangGraph-Powered Agentic Workflow**: Custom StateGraph with intelligent routing, tool calling, file-aware reasoning, and structured memory updates. Not a simple chain — a true graph-based agent.
- **Playwright Web Tools**: Dynamic browser automation for web search, page navigation, content extraction, and research tasks. Handles JavaScript-heavy sites gracefully.
- **True Multi-Modal File Support**: Upload PDFs (text + tables), images (OCR + vision LLM description), DOCX, TXT, CSV. Files are intelligently summarized and injected into context.
- **Smart Per-User Persistent Memory**: SQLite-backed storage that **automatically retains only the last 6 conversation states/turns** per user. Optimizes token usage, cost, and relevance.
- **Polished Streamlit UI**: Modern chat experience with file drag-and-drop, real-time tool execution feedback, memory inspector, user switching (for multi-tenant demo), settings panel, and export options.
- **Production-Ready Engineering**:
  - Modular, typed codebase (Pydantic + TypedDict)
  - Configurable LLM (OpenAI, Grok/xAI, Claude, local via compatible endpoints)
  - Docker + docker-compose for one-command deployment
  - GitHub Actions CI (lint + test skeleton)
  - Comprehensive error handling & logging
  - .env driven configuration

**Perfect for**: Research assistants, document analysis agents, web-augmented Q&A bots, internal knowledge tools, or as a foundation for RAG + agent systems.

---

## 🏗️ Project Architecture

### High-Level System Diagram

```mermaid
flowchart TB
    subgraph "Frontend - Streamlit"
        UI[streamlit_app.py<br/>Modern Chat + File Uploader<br/>Sidebar: Memory Viewer, Settings, User Switcher]
    end

    subgraph "Core Agent Layer"
        direction TB
        SA[Prepare State & Load Memory<br/>(last 6 turns from SQLite)]
        FP[File Processor Node<br/>Extract text / describe images<br/>Add to context]
        RT[Router / Intent Node<br/>Decide: web tool? file QA? direct answer?]
        PT[Playwright Tool Node<br/>web_search, browse_page, extract]
        LLM[LLM Call with Tools<br/>Bound Playwright + custom tools<br/>Reasoning + final answer]
        MM[Memory Manager Node<br/>Append turn → Trim to MAX=6 → Persist]
    end

    subgraph "Persistence & External"
        DB[(SQLite DB<br/>users + conversation_turns<br/>Auto-prune >6 states)]
        BR[Playwright Chromium<br/>Headless browser automation]
        LLM_PROVIDER[(OpenAI / xAI Grok / Groq / Anthropic)]
    end

    UI -->|User message + files| SA
    SA --> FP
    FP --> RT
    RT -->|needs web| PT
    PT --> LLM
    RT -->|no web| LLM
    LLM --> MM
    MM --> DB
    MM -->|response + steps| UI

    style UI fill:#f0f8ff
    style DB fill:#fff8dc
    style BR fill:#e6ffe6
```

### Detailed Component Breakdown

| Layer | File(s) | Responsibility |
|-------|---------|----------------|
| **UI** | `app/streamlit_app.py` | Chat interface, file handling, session state, invoke graph, display tool traces |
| **Agent Graph** | `app/agent/graph.py`, `state.py`, `nodes.py` | StateGraph definition, node implementations, conditional edges, tool binding |
| **Tools** | `app/agent/tools.py` | `@tool` decorated Playwright functions + file-aware helpers |
| **Memory** | `app/memory/db.py` | SQLite CRUD, automatic trim logic (`MAX_MEMORY_STATES=6`), context retrieval |
| **File Utils** | `app/utils/file_handlers.py` | PDF (pdfplumber), DOCX, images (PIL + pytesseract or vision), CSV/JSON parsing + smart summarization |
| **Browser** | `app/utils/playwright_browser.py` | Reusable Playwright wrapper class with search, navigate, screenshot (optional), robust error handling |
| **Config & LLM** | `app/config.py`, `app/utils/llm.py` | Pydantic settings, LLM factory (ChatOpenAI compatible) |
| **Deployment** | `Dockerfile`, `docker-compose.yml` | Containerized Streamlit + browser deps + volume for persistent DB |

### State Schema (Simplified)

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    file_context: list[dict]          # summaries + excerpts from uploads
    tool_results: list[dict]
    current_turn: int
    # memory automatically injected from DB
```

The graph uses `MessagesState` + extra keys. Memory manager ensures `messages` list passed to LLM never exceeds reasonable context from the last 6 turns.

### Memory Trimming Strategy

- Every conversation turn is saved as a JSON row (user_id, turn_id, timestamp, role, content, metadata).
- After insert: `DELETE FROM turns WHERE user_id = ? AND turn_id < (SELECT MAX(turn_id) - 5 FROM turns WHERE user_id = ?)`
- Retrieved context = last 6 turns formatted as chat history.
- This keeps DB small, prompt focused on recent relevant context, and fulfills the "only last 6 states" requirement perfectly.

---

## 🚀 Quickstart (Local)

1. **Clone & Setup**
   ```bash
   git clone https://github.com/Lackey43/langgraph-streamlit-playwright-chatbot.git
   cd langgraph-streamlit-playwright-chatbot
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   playwright install chromium   # Important: installs browser binaries
   cp .env.example .env
   # Edit .env with your LLM_API_KEY (OpenAI or xAI Grok recommended)
   ```

2. **Run the App**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Open http://localhost:8501 — start chatting, upload files, switch users to test memory isolation!

3. **Docker (Recommended for full features)**
   ```bash
   docker compose up --build
   ```

---

## 🛠️ Configuration & Extensibility

- Change LLM: edit `LLM_MODEL` / `LLM_BASE_URL` in `.env` (supports Grok via `https://api.x.ai/v1`)
- Memory limit: `MAX_MEMORY_STATES=6` in `.env` (easily tunable)
- Add new tools: implement in `app/agent/tools.py`, register in graph.
- Vision/images: Set `ENABLE_VISION=true` and use a multimodal model (gpt-4o, grok-vision). Images are passed as base64 when supported.
- Tracing: Plug in LangSmith easily by setting `LANGCHAIN_TRACING_V2=true` + API key.

---

## 📸 Screenshots & Demo (Add your own after running)

*(Recommended: Add GIF of chatting + file upload + seeing tool use + memory panel)*

---

## 🧪 Testing & Quality

```bash
ruff check .
black --check .
pytest tests/
```
CI workflow in `.github/workflows/ci.yml` runs on push/PR.

---

## 📈 Why This Project Impresses Employers

1. **Demonstrates Full-Stack AI Engineering**: From LLM orchestration (LangGraph) → external tools (Playwright is non-trivial) → multimodal I/O → persistent state management → beautiful UX (Streamlit) → containerized deployment.
2. **Real Production Patterns**: Memory pruning, user isolation, config-driven, error resilience, modular design (swap components easily).
3. **Modern Stack Mastery**: Shows you're up-to-date with 2025-2026 agent frameworks, not stuck on legacy chains.
4. **Portfolio-Ready**: Clean README, architecture docs, runnable demo, Docker, CI — ready to present in interviews or share link.
5. **Extensible Foundation**: Easy to evolve into full RAG system (add Chroma/FAISS), multi-agent (CrewAI/LangGraph teams), or enterprise features (auth, RBAC, audit logs).

**Skills showcased**: Agent design, tool integration, stateful apps, database design for AI, UI/UX for AI products, DevOps basics, clean code.

---

## 🔮 Future Roadmap (Ideas for contributors/forks)

- Add vector RAG over uploaded documents (Chroma + LangChain)
- LangSmith / Phoenix tracing integration
- Voice input/output
- Multi-agent collaboration (researcher + writer + critic)
- Deploy to Streamlit Cloud / Railway / Modal
- Evaluation harness (RAGAS or custom)
- User auth + conversation sharing

---

## 📄 License

MIT — feel free to use as base for your own projects or portfolio pieces.

**Built with ❤️ for the AI engineering community** by Lackey (demo project).

---

*Questions or want to collaborate? Open an issue or connect on X @LackeyZone*

**Star ⭐ this repo if you find it useful for learning or job hunting!**