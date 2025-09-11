import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import gc
import time
from datetime import datetime

import streamlit as st
import yaml
from dotenv import load_dotenv
from function_utils import check_korean, delete_log, display_docx, display_pdf
from langchain_core.prompts import load_prompt

from intraseek.modules.documents import get_docx_loader, get_pdf_loader
from intraseek.modules.llms import get_embedding
from intraseek.modules.prompts import build_qa_prompt, save_fewshot_prompt, save_prompt
from intraseek.modules.vector_db import get_splitter, get_vector_store
from intraseek.utils.config_loader import dump_yaml, load_yaml
from intraseek.utils.path import LOG_PATH, PROMPT_CONFIG_PATH
from intraseek.utils.rag_utils import delete_incomplete_logs, save_rag_configs

load_dotenv()

st.set_page_config(
    page_title="Document Upload for RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Upload for RAG")
st.caption("Upload documents to create a searchable knowledge base for question-answering")

RAG_LOG_PATH = LOG_PATH / "rag"
os.makedirs(RAG_LOG_PATH, exist_ok=True)

delete_incomplete_logs(base_path=RAG_LOG_PATH, required_files=["prompt.yaml", "rag_config.yaml"])

if "rag_name" not in st.session_state:
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    preposition = "RAG_"
    st.session_state["rag_name"] = preposition + current_datetime

if "prompt" not in st.session_state:
    st.session_state["prompt"] = None
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = None
if "doc_format" not in st.session_state:
    st.session_state["doc_format"] = "pdf"
if "selected_system_prompt" not in st.session_state:
    st.session_state["selected_system_prompt"] = "simple_qa_prompt_kor"
if "selected_example" not in st.session_state:
    st.session_state["selected_example"] = "example_template"

with st.sidebar:
    st.write("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx", "docc"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload multiple files of the same type (PDF or DOCX)"
    )

    if uploaded_files:
        extensions = set(os.path.splitext(file.name)[1].lower().lstrip(".") for file in uploaded_files)

        if len(extensions) == 1:
            st.session_state["uploaded_files"] = uploaded_files
            st.session_state["doc_format"] = next(iter(extensions))
            st.success(f"✅ {len(uploaded_files)} {st.session_state['doc_format'].upper()} files uploaded")
        else:
            st.error("❌ Please upload files of the same type only (all PDF or all DOCX)")
            st.session_state["uploaded_files"] = None
    else:
        st.session_state["uploaded_files"] = None

    st.write("### 📚 Existing RAG Collections")
    logs = [f for f in os.listdir(RAG_LOG_PATH) if os.path.isdir(os.path.join(RAG_LOG_PATH, f))]

    if not logs:
        st.info("No RAG collections found")
    else:
        selected_log = st.selectbox("Select RAG collection", logs)
        selected_log_path = RAG_LOG_PATH / selected_log

        col1, col2 = st.columns(2)

        with col1:
            with st.popover("📋 View Details", use_container_width=True):
                try:
                    selected_prompt = load_yaml(selected_log_path / "prompt.yaml")
                    selected_rag_config = load_yaml(selected_log_path / "rag_config.yaml")

                    tab1, tab2 = st.tabs(["Prompt", "Config"])

                    with tab1:
                        st.code(dump_yaml(selected_prompt), language="yaml")

                    with tab2:
                        st.code(dump_yaml(selected_rag_config), language="yaml")
                except Exception as e:
                    st.error(f"Error loading details: {str(e)}")

        with col2:
            delete_log_btn = st.button("🗑️ Delete", use_container_width=True)

        if delete_log_btn:
            if selected_log:
                delete_log(selected_log_path)
            else:
                st.warning("Please select a RAG collection to delete")

st.write("")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.write("### 📄 **Documents Preview**")
    with st.container(height=700):
        num_tabs = (
            min(len(st.session_state["uploaded_files"]), 5) if st.session_state["uploaded_files"] else 0
        )

        if num_tabs > 0:
            tabs = st.tabs([f"📄 {i+1}" for i in range(num_tabs)])

            for i, tab in enumerate(tabs):
                with tab:
                    if i < len(st.session_state["uploaded_files"]):
                        file = st.session_state["uploaded_files"][i]
                        st.write(f"**File:** `{file.name}`")

                        file_extension = os.path.splitext(file.name)[1].lower()
                        file_data = file.read()

                        if file_extension == ".pdf":
                            # Use streamlit_pdf_viewer for better PDF display
                            try:
                                from streamlit_pdf_viewer import pdf_viewer
                                import tempfile
                                import os
                                
                                # Create a temporary file and write PDF data to it
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(file_data)
                                    tmp_file_path = tmp_file.name
                                
                                # Use pdf_viewer with the temporary file path
                                pdf_viewer(tmp_file_path, width=700, height=620)
                                
                                # Clean up the temporary file
                                try:
                                    os.unlink(tmp_file_path)
                                except:
                                    pass
                                    
                            except ImportError:
                                st.error("streamlit-pdf-viewer 패키지가 설치되지 않았습니다.")
                                st.info("다음 명령어로 설치하세요: pip install streamlit-pdf-viewer")
                                # Fallback to original method
                                pdf_display = display_pdf(file_data, scale=0.9, height=620)
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"PDF 표시 오류: {str(e)}")
                                # Clean up temporary file if it exists
                                try:
                                    if 'tmp_file_path' in locals():
                                        os.unlink(tmp_file_path)
                                except:
                                    pass
                                # Fallback to original method
                                pdf_display = display_pdf(file_data, scale=0.9, height=620)
                                st.markdown(pdf_display, unsafe_allow_html=True)
                        elif file_extension in [".docx", ".docc"]:
                            docx_display = display_docx(file_data, scale=0.9, height=620)
                            st.markdown(docx_display, unsafe_allow_html=True)
                        else:
                            st.error("Unsupported file format")
                    else:
                        st.write("No document uploaded")
        else:
            st.info("📁 Upload documents to see preview here")

with col2:
    st.write("### ⚙️ **Prompt Settings**")
    with st.container(height=700):
        tab1, tab2 = st.tabs(["System Prompt", "Examples"])

        with tab1:
            st.session_state["selected_system_prompt"] = st.selectbox(
                label="Select prompt template",
                options=[
                    "simple_qa_prompt_kor",
                    "simple_qa_prompt",
                    "hr_qa_prompt_kor",
                    "hr_qa_prompt",
                ],
                index=[
                    "simple_qa_prompt_kor",
                    "simple_qa_prompt",
                    "hr_qa_prompt_kor",
                    "hr_qa_prompt",
                ].index(st.session_state["selected_system_prompt"]),
                help="Choose the prompt template for Q&A",
            )

            try:
                prompt_template = load_prompt(
                    PROMPT_CONFIG_PATH / f"{st.session_state['selected_system_prompt']}.yaml",
                    encoding="utf-8",
                )
                system_prompt = prompt_template.template
                if "Context:" in system_prompt:
                    system_prompt = system_prompt.split("Context:")[0].strip()

                st.write("")
                edited_message = st.text_area(
                    "System Prompt",
                    system_prompt,
                    height=265,
                    label_visibility="collapsed",
                )
            except Exception as e:
                st.error(f"Error loading prompt: {str(e)}")
                edited_message = "You are a helpful assistant that answers questions based on the provided context."

        with tab2:
            use_example_check = st.checkbox("Use examples", value=False)
            
            st.session_state["selected_example"] = st.selectbox(
                label="Select example template",
                options=["example_template", "hr_example_template"],
                index=["example_template", "hr_example_template"].index(
                    st.session_state["selected_example"],
                ),
                disabled=not use_example_check,
                help="Choose example template",
            )

            try:
                example_template = load_yaml(
                    PROMPT_CONFIG_PATH / f"{st.session_state['selected_example']}.yaml",
                )
                example_content = example_template.get("answer_examples", "")

                st.write("")
                few_shot_msg = st.text_area(
                    "Few-shot examples",
                    dump_yaml(example_content),
                    height=350,
                    disabled=not use_example_check,
                    label_visibility="collapsed",
                )
            except Exception as e:
                st.error(f"Error loading examples: {str(e)}")
                few_shot_msg = ""

        with st.form(key="prompt_setting", border=False):
            save_prompt_button = st.form_submit_button(label="💾 Save Prompt", use_container_width=True)
        
        if save_prompt_button:
            st.session_state["use_example"] = use_example_check
            try:
                st.session_state["prompt"] = build_qa_prompt(
                    system_message=edited_message,
                    examples=yaml.safe_load(few_shot_msg) if st.session_state["use_example"] else None,
                )
                st.success("✅ Prompt saved successfully!")
            except Exception as e:
                st.error(f"Error saving prompt: {str(e)}")

with col3:
    st.write("### 🗄️ **Vector Database Settings**")
    with st.container(height=700):
        if st.session_state["uploaded_files"]:
            doc_names = [doc.name for doc in st.session_state["uploaded_files"]]
            st.write("**Uploaded files:**")
            for name in doc_names:
                st.write(f"• {name}")
        else:
            st.info("📁 Upload documents first")

        st.write("")
        rag_name_input = st.text_input(
            "RAG Collection Name",
            value=st.session_state["rag_name"],
            help="Name for your RAG collection (English only)",
        )

        is_valid_name = not check_korean(rag_name_input)

        if check_korean(rag_name_input):
            st.warning("⚠️ Please use English characters only")

        st.write("")
        
        # Document processing settings
        st.write("**Document Processing:**")
        c1, c2 = st.columns(2)

        with c1:
            pdf_loader_type = st.selectbox(
                "PDF Loader",
                ["pymupdf", "pypdf", "pdfplumber", "pdfminer"],
                help="PDF processing method",
                disabled=(st.session_state["doc_format"] != "pdf"),
            )
            chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)

        with c2:
            text_splitter_type = st.selectbox(
                "Text Splitter",
                ["RecursiveCT", "CharacterText", "TokenText"],
                help="Text splitting method",
            )
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=20,
                max_value=200,
                value=100,
                step=10,
            )

        st.write("")
        submit_button = st.button(label="🚀 Create RAG Collection", use_container_width=True)

        if submit_button and is_valid_name:
            if "uploaded_files" not in st.session_state or not st.session_state["uploaded_files"]:
                st.warning("⚠️ Please upload documents first")

            elif "prompt" not in st.session_state or not st.session_state["prompt"]:
                st.warning("⚠️ Please save the prompt first")

            else:
                with st.status("🔄 Creating RAG collection...", expanded=True):
                    time.sleep(0.5)
                    st.write("**📁 Preparing directories...**")
                    dir_path = RAG_LOG_PATH / rag_name_input
                    prompt_path = dir_path / "prompt.yaml"
                    doc_path = dir_path / "docs"
                    db_path = dir_path / "db"
                    os.makedirs(dir_path, exist_ok=True)
                    os.makedirs(doc_path, exist_ok=True)
                    time.sleep(0.5)

                    st.write("**📄 Processing documents...**")
                    for file in st.session_state["uploaded_files"]:
                        file_path = doc_path / f"{file.name}"
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())

                    splitter = get_splitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        type=text_splitter_type,
                    )
                    
                    if st.session_state["doc_format"] == "pdf":
                        loader = get_pdf_loader(file_path=doc_path, type="directory")
                    elif st.session_state["doc_format"] in ["docx", "docc"]:
                        loader = get_docx_loader(file_path=doc_path, type="directory")
                    else:
                        raise ValueError(f"Unsupported document format: {st.session_state['doc_format']}")
                    
                    docs = loader.load()

                    st.write("**🔍 Creating vector database...**")
                    splits = splitter.split_documents(docs)
                    
                    # Get embedding model with fallback
                    embedding_model = os.getenv("SELECTED_EMBEDDING")
                    if not embedding_model:
                        # Try OpenAI first, then fallback to Ollama
                        if os.getenv("OPENAI_API_KEY"):
                            embedding_model = "text-embedding-3-large"  # OpenAI fallback
                            st.info(f"💡 Using OpenAI embedding: {embedding_model}")
                        else:
                            embedding_model = "nomic-embed-text"  # Ollama fallback
                            st.warning(f"⚠️ SELECTED_EMBEDDING not set, using Ollama default: {embedding_model}")
                            st.info("💡 To use OpenAI embeddings, set SELECTED_EMBEDDING=text-embedding-3-large and OPENAI_API_KEY")
                    
                    try:
                        embedding = get_embedding(
                            model=embedding_model,
                            base_url=os.getenv("OLLAMA_BASE_URL"),
                        )
                        st.write(f"✅ Using embedding model: {embedding_model}")
                    except Exception as embed_error:
                        st.error(f"❌ Failed to initialize embedding model '{embedding_model}': {str(embed_error)}")
                        if embedding_model.startswith("text-embedding"):
                            st.error("💡 For OpenAI embeddings, make sure OPENAI_API_KEY is set in your environment")
                        else:
                            st.error("💡 For Ollama embeddings, make sure Ollama is running and the model is available")
                        raise embed_error
                    vectorstore = get_vector_store(documents=splits, embedding=embedding)
                    vectorstore.save_local(folder_path=db_path)

                    st.write("**💾 Saving configuration...**")

                    if st.session_state["use_example"]:
                        save_fewshot_prompt(prompt=st.session_state["prompt"], save_path=prompt_path)
                    else:
                        save_prompt(prompt=st.session_state["prompt"], save_path=prompt_path)
                    
                    save_rag_configs(
                        save_path=dir_path / "rag_config.yaml",
                        document_format=st.session_state["doc_format"],
                        documents=doc_names,
                        text_splitter_type=text_splitter_type,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        loader_type="directory",
                        vectorstore_type="FAISS",
                        embedding_type=embedding_model,
                    )
                    
                    st.write("**✅ RAG collection created successfully!**")
                    st.success(f"🎉 RAG collection '{rag_name_input}' is ready for use!")

                time.sleep(2)
                keys_to_clear = ["prompt", "uploaded_files", "rag_name"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                gc.collect()
                st.rerun()
