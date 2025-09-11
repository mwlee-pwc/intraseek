import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import streamlit as st
from dotenv import load_dotenv
from function_utils import (
    ChatCallbackHandler,
    display_docx,
    display_pdf,
    load_image,
    load_retriver,
    pain_history,
    send_message,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import load_prompt
from PIL import Image

from intraseek.chains import build_simple_chain
from intraseek.modules.llms import get_llm
from intraseek.utils.config_loader import dump_yaml, load_yaml
from intraseek.utils.path import DEMO_IMG_PATH, LOG_PATH
from intraseek.utils.rag_utils import delete_incomplete_logs, format_docs_with_source
from intraseek.utils.perplexity import fetch_perplexity_stream
import json

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("💬 RAG Chatbot")
st.caption("Ask questions about your uploaded documents")

RAG_LOG_PATH = LOG_PATH / "rag"
SUMMARY_KB_PATH = LOG_PATH / "summary_kb"
os.makedirs(RAG_LOG_PATH, exist_ok=True)
os.makedirs(SUMMARY_KB_PATH, exist_ok=True)
delete_incomplete_logs(base_path=RAG_LOG_PATH, required_files=["prompt.yaml", "rag_config.yaml"])

human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avatar = DEMO_IMG_PATH / "pwc_logo.png"

# Load avatar with fallback
try:
    ai_avatar_image = load_image(ai_avatar)
except FileNotFoundError:
    # Fallback to a simple emoji or None if image not found
    ai_avatar_image = None

# Initialize session state
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(
        return_messages=True,
        k=3,
        memory_key="chat_history",
    )

if "rag_qa_on" not in st.session_state:
    st.session_state["rag_qa_on"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "llm_temp" not in st.session_state:
    st.session_state["llm_temp"] = 0.2
if "retriever_k" not in st.session_state:
    st.session_state["retriever_k"] = 3
if "selected_rag_llm" not in st.session_state:
    st.session_state["selected_rag_llm"] = os.getenv("SELECTED_LLM")

memory = st.session_state["memory"]

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]

# Sidebar for RAG collection selection and settings
with st.sidebar:
    st.write("### 📚 Knowledge Sources")
    
    # Get RAG collections and summary KBs
    rag_logs = [f for f in os.listdir(RAG_LOG_PATH) if os.path.isdir(os.path.join(RAG_LOG_PATH, f))]
    summary_kbs = [f for f in os.listdir(SUMMARY_KB_PATH) if os.path.isdir(os.path.join(SUMMARY_KB_PATH, f))]
    
    # Create list with only RAG document sources (exclude RFP checklists)
    all_sources = []
    for log in rag_logs:
        all_sources.append(f"📄 {log} (Documents)")
    
    if not all_sources:
        st.error("❌ No knowledge sources found")
        st.info("Create collections in 'Document Upload (RAG)' or 'Document Summarization' pages")
        st.session_state["build_bot"] = False
        selected_db = None
        selected_log = None
        selected_prompt = None
        selected_rag_config = None
        selected_source_type = None
    else:
        selected_source = st.selectbox("Select Knowledge Source", all_sources)
        
        # Parse selection (only RAG documents now)
        selected_log = selected_source.replace("📄 ", "").replace(" (Documents)", "")
        selected_log_path = RAG_LOG_PATH / selected_log
        selected_source_type = "rag"

        with st.expander("⚙️ **Settings**", expanded=False):
            llm_temperature = st.slider(
                "LLM Temperature",
                min_value=0.1,
                max_value=1.0,
                value=0.2,
                step=0.1,
                disabled=st.session_state["rag_qa_on"],
                help="Controls randomness in responses"
            )
            retriever_k = st.slider(
                "Retriever K value",
                min_value=2,
                max_value=8,
                value=3,
                step=1,
                disabled=st.session_state["rag_qa_on"],
                help="Number of relevant documents to retrieve"
            )
            llm_type = st.radio(
                "LLM Model",
                [
                    "gpt-4o-mini",
                    "gemma2:27b",
                    "gemma2:9b",
                    "llama3.1:8b",
                ],
                horizontal=True,
                disabled=st.session_state["rag_qa_on"],
                help="Choose the language model",
            )

        # Conditional RFP Checklist Loading
        if summary_kbs and selected_source_type == "rag":
            st.write("### 📋 **RFP Checklist Context**")
            
            use_rfp_context = st.checkbox(
                "Include RFP checklist context in chat",
                value=False,
                help="Add RFP checklist information to enhance document-based responses"
            )
            
            if use_rfp_context:
                selected_rfp_checklist = st.selectbox(
                    "Select RFP Checklist to include",
                    summary_kbs,
                    help="Choose which RFP checklist to include as context"
                )
                
                # Show checklist details in popup
                with st.popover("📋 View Checklist Details", use_container_width=True):
                    try:
                        checklist_path = SUMMARY_KB_PATH / selected_rfp_checklist
                        checklist_files = [f for f in os.listdir(checklist_path) if f.endswith("_summary.json")]
                        
                        if checklist_files:
                            st.write(f"**RFP Checklist: {selected_rfp_checklist}**")
                            st.write("---")
                            
                            for checklist_file in checklist_files:
                                with open(checklist_path / checklist_file, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                
                                st.write(f"**📄 {data['original_file']}**")
                                st.write(f"*Created: {data['metadata']['created_at'][:19]}*")
                                st.write("**Checklist Content:**")
                                st.write(data['content'])
                                st.write("---")
                        else:
                            st.info("No checklist items found")
                    except Exception as e:
                        st.error(f"Error loading checklist: {str(e)}")
                
                # Store the selected checklist for use in chat
                st.session_state["selected_rfp_checklist"] = selected_rfp_checklist
            else:
                st.session_state["selected_rfp_checklist"] = None


        use_perplexity = st.checkbox(
            "Use Perplexity API for answers",
            value=False,
            help="When enabled, search from Perplexity API will also be used as a reference.",
        )
        # Persist choice
        st.session_state["use_perplexity"] = use_perplexity

        col1, col2 = st.columns(2)

        with col1:
            with st.popover("📋 View Details", use_container_width=True):
                try:
                    if selected_source_type == "rag":
                        selected_prompt = load_yaml(selected_log_path / "prompt.yaml")
                        selected_rag_config = load_yaml(selected_log_path / "rag_config.yaml")

                        tab1, tab2 = st.tabs(["Prompt", "Config"])

                        with tab1:
                            st.code(dump_yaml(selected_prompt), language="yaml")

                        with tab2:
                            st.code(dump_yaml(selected_rag_config), language="yaml")
                    elif selected_source_type == "summary":
                        # Show RFP checklist details
                        st.write("**RFP Checklist Contents:**")
                        checklist_files = [f for f in os.listdir(selected_log_path) if f.endswith("_summary.json")]
                        
                        if checklist_files:
                            for checklist_file in checklist_files:
                                with open(selected_log_path / checklist_file, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    st.write(f"**📄 {data['original_file']}**")
                                    st.write(f"*Analysis Type: {data['metadata']['analysis_type']}*")
                                    st.write(f"*Created: {data['metadata']['created_at'][:19]}*")
                                    st.write("---")
                        else:
                            st.info("No RFP checklists found")
                except Exception as e:
                    st.error(f"Error loading details: {str(e)}")

        with col2:
            reset_btn = st.button("🔄 Reset Chat", disabled=not all_sources, use_container_width=True)
            build_bot_btn = st.toggle("🚀 Activate Chatbot", key="build_bot", disabled=not all_sources)

        if build_bot_btn:
            st.session_state["llm_temp"] = llm_temperature
            st.session_state["retriever_k"] = retriever_k
            st.session_state["selected_rag_llm"] = (
                "gpt-4o" if os.getenv("SELECTED_LLM").startswith("gpt") else llm_type
            )
            st.session_state["selected_source_type"] = selected_source_type
            st.session_state["selected_log_path"] = selected_log_path
            st.session_state["rag_qa_on"] = True
            
            if selected_source_type == "rag":
                selected_doc_path = selected_log_path / "docs"
                selected_db_path = selected_log_path / "db"
                selected_prompt_path = selected_log_path / "prompt.yaml"
                prompt = load_prompt(selected_log_path / "prompt.yaml", encoding="utf-8")
                st.session_state["selected_prompt"] = prompt
                st.session_state["selected_doc_path"] = selected_doc_path
                st.session_state["selected_db_path"] = selected_db_path
            elif selected_source_type == "summary":
                selected_db_path = selected_log_path / "db"
                st.session_state["selected_db_path"] = selected_db_path
                # Create a simple prompt for RFP checklist Q&A
                from langchain_core.prompts import PromptTemplate
                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""You are a helpful assistant that answers questions based on RFP checklists and requirements.

Context (from RFP checklists):
{context}

Question: {question}

Answer based on the RFP checklists provided. Focus on actionable items, deadlines, and requirements. If the information is not available in the checklists, say so clearly."""
                )
                st.session_state["selected_prompt"] = prompt
            
            st.success("✅ Chatbot activated!")
        else:
            st.session_state["rag_qa_on"] = False

        if reset_btn and st.session_state["rag_qa_on"]:
            st.session_state["messages"] = []
            st.session_state["memory"].clear()
            st.success("🔄 Chat history cleared!")

# Main content area
col1, col2 = st.columns([1, 1])
with col1:
    st.write("### 📄 **Knowledge Source Preview**")
    with st.container(height=700):
        if st.session_state["rag_qa_on"]:
            source_type = st.session_state.get("selected_source_type")
            
            if source_type == "rag":
                # Show original documents
                try:
                    selected_rag_config = load_yaml(st.session_state["selected_log_path"] / "rag_config.yaml")
                    selected_doc_path = st.session_state["selected_doc_path"]
                    
                    num_tabs = (
                        min(len(selected_rag_config["documents"]), 10) if selected_rag_config["documents"] else 0
                    )
                    if num_tabs > 0:
                        tabs = st.tabs([f"📄 {i+1}" for i in range(num_tabs)])

                        for i, tab in enumerate(tabs):
                            with tab:
                                file_name = selected_rag_config["documents"][i]
                                file_path = selected_doc_path / file_name
                                st.write(f"**File:** `{file_name}`")
                                
                                try:
                                    with open(file_path, "rb") as file:
                                        file_data = file.read()

                                    if selected_rag_config["document_format"] == "pdf":
                                        pdf_display = display_pdf(file_data, scale=0.8, height=620)
                                        st.markdown(pdf_display, unsafe_allow_html=True)
                                    elif selected_rag_config["document_format"] in ["docx", "docc"]:
                                        docx_display = display_docx(file_data, scale=0.8, height=620)
                                        st.markdown(docx_display, unsafe_allow_html=True)
                                    else:
                                        st.error("Unsupported file format")
                                except Exception as e:
                                    st.error(f"Error loading document: {str(e)}")
                    else:
                        st.info("No documents found in this collection")
                except Exception as e:
                    st.error(f"Error loading RAG config: {str(e)}")
                    
            elif source_type == "summary":
                # Show summaries
                try:
                    summary_files = [f for f in os.listdir(st.session_state["selected_log_path"]) if f.endswith("_summary.json")]
                    
                    if summary_files:
                        tabs = st.tabs([f"📝 {i+1}" for i in range(min(len(summary_files), 5))])
                        
                        for i, tab in enumerate(tabs):
                            with tab:
                                if i < len(summary_files):
                                    summary_file = summary_files[i]
                                    with open(st.session_state["selected_log_path"] / summary_file, "r", encoding="utf-8") as f:
                                        data = json.load(f)
                                    
                                    st.write(f"**📄 Original:** `{data['original_file']}`")
                                    st.write(f"**📝 Summary Length:** {data['metadata']['length']}")
                                    st.write(f"**📅 Created:** {data['metadata']['created_at'][:19]}")
                                    st.write("---")
                                    st.write("**Summary Content:**")
                                    st.write(data['content'])
                                else:
                                    st.write("No summary available")
                    else:
                        st.info("No summaries found in this knowledge base")
                except Exception as e:
                    st.error(f"Error loading summaries: {str(e)}")
        else:
            st.info("🚀 Activate a chatbot to view knowledge source")

with col2:
    st.write("### 💬 **Chat Interface**")
    with st.container(height=700):
        if st.session_state["rag_qa_on"]:
            try:
                llm = get_llm(
                    model=st.session_state["selected_rag_llm"],
                    temperature=st.session_state["llm_temp"],
                    streaming=True,
                    callbacks=[ChatCallbackHandler()],
                    base_url=os.getenv("OLLAMA_BASE_URL"),
                )

                # Load retriever based on source type
                source_type = st.session_state.get("selected_source_type")
                selected_db_path = st.session_state["selected_db_path"]
                
                if source_type == "rag":
                    # Load RAG config for embedding model
                    selected_rag_config = load_yaml(st.session_state["selected_log_path"] / "rag_config.yaml")
                    embedding_model = selected_rag_config["embedding"]
                elif source_type == "summary":
                    # Use default embedding for summaries
                    embedding_model = os.getenv("SELECTED_EMBEDDING")
                    if not embedding_model:
                        embedding_model = "nomic-embed-text"  # Default fallback
                
                retriever = load_retriver(
                    db_path=selected_db_path,
                    embedding_model=embedding_model,
                    retriever_k=st.session_state["retriever_k"],
                    base_url=os.getenv("OLLAMA_BASE_URL"),
                )

                # Welcome message
                if not st.session_state["messages"]:
                    if source_type == "rag":
                        if st.session_state.get("selected_rfp_checklist"):
                            welcome_msg = "👋 Hello! I'm ready to answer questions about your documents with RFP checklist context. What would you like to know?"
                        else:
                            welcome_msg = "👋 Hello! I'm ready to answer questions about your documents. What would you like to know?"
                    elif source_type == "summary":
                        welcome_msg = "👋 Hello! I'm ready to answer questions about your RFP checklists. What would you like to know?"
                    else:
                        welcome_msg = "👋 Hello! I'm ready to help. What would you like to know?"
                    send_message(welcome_msg, "ai", save=False)
                
                # Display chat history
                pain_history()

                # Chat input
                with st._bottom:
                    if source_type == "summary":
                        message = st.chat_input("Ask a question about your RFP checklists...")
                    else:
                        if st.session_state.get("selected_rfp_checklist"):
                            message = st.chat_input("Ask a question about your documents (with RFP context)...")
                        else:
                            message = st.chat_input("Ask a question about your documents...")

                if message:
                    send_message(message, "human")
                    
                    try:
                        prompt = st.session_state["selected_prompt"]
                        chain = build_simple_chain(
                            retriever=retriever,
                            prompt=prompt,
                            llm=llm,
                            load_memory_func=load_memory,
                            format_docs_func=format_docs_with_source,
                        )

                        with st.chat_message("ai", avatar=ai_avatar_image if ai_avatar_image else "🤖"):

                            # If Perplexity mode is enabled, fetch Perplexity's answer and attach it
                            # print(st.session_state.get("use_perplexity"), os.getenv("PERPLEXITY_API_KEY"))
                            if st.session_state.get("use_perplexity") and os.getenv("PERPLEXITY_API_KEY"):
                                try:
                                    placeholder = st.empty()
                                    accumulated = []

                                    for chunk in fetch_perplexity_stream(message):
                                        accumulated.append(chunk)
                                        # update UI progressively
                                        placeholder.markdown("".join(accumulated))

                                    content = "".join(accumulated)
                                except Exception as e:
                                    print(f"Error fetching Perplexity answer: {str(e)}")
                            else:
                                content = chain.invoke(message).content

                            memory.save_context({"input": message}, {"output": content})
                            st.session_state["memory"] = memory
                    except Exception as e:
                        st.error(f"❌ Error processing your question: {str(e)}")
                        
            except Exception as e:
                st.error(f"❌ Error initializing chatbot: {str(e)}")
                st.info("Please check your knowledge source settings and try again")
        else:
            st.info("🚀 Select and activate a knowledge source to start chatting")
            
            if all_sources:
                st.write("**Available knowledge sources:**")
                for source in all_sources:
                    st.write(f"• {source}")
            else:
                st.write("**No knowledge sources available.**")
                st.write("Create collections in 'Document Upload (RAG)' or 'Document Summarization' pages!")

# Clean interface - removed instructions for better focus on chat
