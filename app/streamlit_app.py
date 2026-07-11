"""
Main Streamlit application for the LangGraph Playwright Agentic Chatbot.
Features:
- Modern chat interface with avatars
- Multi-file uploader (PDF, images, docs, csv...)
- Real-time display of tool usage and file processing
- Memory viewer (last 6 turns)
- User switching for multi-user demo
- Settings sidebar
- Clear memory / export chat
"""
import streamlit as st
import logging
from datetime import datetime
from typing import List, Dict, Any
import uuid

from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.memory.db import MemoryStore
from app.utils.file_handlers import process_uploaded_files, SUPPORTED_EXTENSIONS
from app.agent.graph import agent_graph
from app.agent.state import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title=settings.app_title,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for nicer look
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.75rem; margin-bottom: 0.75rem; }
    .stChatMessage[data-testid="chat-message-user"] { background-color: #f0f4ff; }
    .stChatMessage[data-testid="chat-message-assistant"] { background-color: #f8f9fa; }
    .tool-call { font-size: 0.85rem; background: #fff3cd; padding: 0.4rem 0.6rem; border-radius: 6px; margin: 0.3rem 0; }
    .file-chip { display: inline-block; background: #e7f5ff; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = settings.default_user_id
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "file_context" not in st.session_state:
    st.session_state.file_context = ""
if "memory_store" not in st.session_state:
    st.session_state.memory_store = MemoryStore()

memory_store: MemoryStore = st.session_state.memory_store

def get_or_create_user_id() -> str:
    """Allow user to switch or create user IDs for demoing multi-user memory."""
    with st.sidebar:
        st.header("👤 User & Memory")
        user_id = st.text_input(
            "Current User ID", 
            value=st.session_state.current_user_id,
            key="user_id_input"
        )
        if user_id != st.session_state.current_user_id:
            st.session_state.current_user_id = user_id
            st.session_state.messages = []  # Clear local chat on user switch
            st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load Last 6 Turns", use_container_width=True):
                load_memory_to_chat()
        with col2:
            if st.button("🗑️ Clear My Memory", type="secondary", use_container_width=True):
                memory_store.clear_user_memory(st.session_state.current_user_id)
                st.session_state.messages = []
                st.success("Memory cleared for this user.")
                st.rerun()
        
        # Stats
        stats = memory_store.get_stats()
        st.caption(f"DB: {stats['total_turns']} turns across {stats['unique_users']} users | Keeping last {settings.max_memory_states} per user")
    return st.session_state.current_user_id

def load_memory_to_chat():
    """Load last 6 turns from DB into the chat UI."""
    turns = memory_store.get_last_n_turns(st.session_state.current_user_id)
    st.session_state.messages = []
    for t in turns:
        if t["role"] == "human":
            st.session_state.messages.append({"role": "user", "content": t["content"]})
        else:
            st.session_state.messages.append({"role": "assistant", "content": t["content"]})
    st.success(f"Loaded {len(turns)} recent turns from memory.")
    st.rerun()

def process_files_and_update_state(uploaded_files: List[Any]):
    """Process uploaded files and store context + metadata in session."""
    if not uploaded_files:
        return
    
    processed, context_str = process_uploaded_files(uploaded_files)
    st.session_state.processed_files = processed
    st.session_state.file_context = context_str
    
    # Show chips in UI
    with st.expander("📎 Processed Files", expanded=True):
        for f in processed:
            st.markdown(f"<span class='file-chip'>{f['name']} ({f['type']}) - {f['size_kb']}KB</span>", unsafe_allow_html=True)
            if f.get("excerpt"):
                st.caption(f.get("summary", "") + " | Excerpt: " + f["excerpt"][:120] + "...")

def invoke_agent(user_input: str, user_id: str):
    """Build state with conversation history (last 6 turns), invoke the LangGraph agent, and handle response."""
    # Load last 6 turns from DB for rich context (this makes the chatbot actually remember previous turns)
    history_turns = memory_store.get_last_n_turns(user_id, n=6)
    history_messages = []
    for t in history_turns:
        if t["role"] == "human":
            history_messages.append(HumanMessage(content=t["content"]))
        elif t["role"] == "ai":
            history_messages.append(AIMessage(content=t["content"]))
    
    # Prepare initial state: history + new user message (file context added in prepare node)
    initial_state: AgentState = {
        "messages": history_messages + [HumanMessage(content=user_input)],
        "user_id": user_id,
        "file_context": st.session_state.get("file_context", ""),
        "file_metadata": st.session_state.get("processed_files", []),
        "tool_results": [],
        "final_response": "",
        "needs_web_search": False,
        "metadata": {}
    }
    
    try:
        # Invoke graph (synchronous for Streamlit)
        result = agent_graph.invoke(initial_state)
        
        # Extract final AI response
        final_ai = result.get("final_response", "No response generated.")
        tool_meta = result.get("metadata", {})
        
        # Update local chat display
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": final_ai})
        
        # Show tool usage if any
        if tool_meta.get("tools_used"):
            st.toast(f"🛠️ Tools used: {', '.join(tool_meta['tools_used'])}", icon="🔧")
        
        # Clear processed files after use (one-shot per message)
        st.session_state.processed_files = []
        st.session_state.file_context = ""
        
        return final_ai
        
    except Exception as e:
        logger.exception("Agent invocation failed")
        error_msg = f"⚠️ Agent error: {str(e)[:300]}. Check your LLM API key and try again."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return error_msg

# ============== UI LAYOUT ==============

st.title(settings.app_title)
st.caption("LangGraph • Playwright Web Tools • Multi-Modal Files • Smart Memory (last 6 states) • Streamlit")

user_id = get_or_create_user_id()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    with st.expander("LLM Configuration", expanded=False):
        st.text_input("Model", value=settings.llm_model, disabled=True)
        st.caption("Change via .env file (LLM_MODEL, LLM_BASE_URL, LLM_API_KEY)")
    
    st.divider()
    st.markdown("**Supported File Types**")
    st.caption(", ".join(SUPPORTED_EXTENSIONS))
    
    st.divider()
    if st.button("📥 Export Current Chat (JSON)", use_container_width=True):
        import json
        chat_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "Download chat.json",
            data=chat_json,
            file_name=f"chat_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# Main chat area
chat_container = st.container()

with chat_container:
    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🧠"):
            st.markdown(msg["content"])
    
    # File uploader (always visible above input for convenience)
    uploaded_files = st.file_uploader(
        "📎 Upload files (PDF, images, DOCX, CSV, TXT...)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="file_uploader",
        help="Files are processed and their content is added to the next message context."
    )
    
    if uploaded_files:
        process_files_and_update_state(uploaded_files)
    
    # Chat input
    if prompt := st.chat_input("Ask anything... (web research, document Q&A, analysis)"):
        # Show user message immediately
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Process + invoke
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🧠 Thinking & using tools if needed..."):
                response = invoke_agent(prompt, user_id)
                st.markdown(response)
        
        # Rerun to persist messages in UI properly
        st.rerun()

# Footer / info
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔒 Memory is private per User ID • Last 6 turns retained")
with col2:
    st.caption("🌐 Web tools powered by Playwright (headless Chromium)")
with col3:
    st.caption("📁 Files processed locally • Vision/OCR supported")

# Debug / advanced (collapsible)
with st.expander("🔍 Debug: Current Session State & Memory", expanded=False):
    st.json({
        "user_id": user_id,
        "local_messages_count": len(st.session_state.messages),
        "processed_files_count": len(st.session_state.get("processed_files", [])),
        "db_stats": memory_store.get_stats()
    })
    if st.button("Show last 6 raw turns from DB"):
        turns = memory_store.get_last_n_turns(user_id)
        st.json(turns)
