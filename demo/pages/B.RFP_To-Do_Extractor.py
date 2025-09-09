import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from function_utils import display_docx, display_pdf
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

from intraseek.modules.documents import get_docx_loader, get_pdf_loader
from intraseek.modules.llms import get_llm, get_embedding
from intraseek.modules.vector_db import get_splitter, get_vector_store
from intraseek.utils.path import LOG_PATH

load_dotenv()

st.set_page_config(
    page_title="RFP To-Do Extractor",
    page_icon="📋",
    layout="wide",
)

st.title("📋 RFP To-Do Extractor")
st.caption("Upload RFP documents to extract To-Do requirements and checkpoints")

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = None
if "doc_format" not in st.session_state:
    st.session_state["doc_format"] = "pdf"
if "summaries" not in st.session_state:
    st.session_state["summaries"] = {}
if "summary_kb_name" not in st.session_state:
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    st.session_state["summary_kb_name"] = f"CHECKLIST_{current_datetime}"

# Create summary knowledge base directory
SUMMARY_KB_PATH = LOG_PATH / "summary_kb"
os.makedirs(SUMMARY_KB_PATH, exist_ok=True)

# Sidebar for file upload
with st.sidebar:
    st.write("### 📁 Upload RFP Documents")
    uploaded_files = st.file_uploader(
        "Upload RFP documents (PDF or DOCX)",
        type=["pdf", "docx", "docc"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload RFP documents to extract To-Do requirements and checkpoints"
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

    st.write("### 📋 RFP Checklists")
    
    # List existing RFP checklists
    rfp_checklists = [f for f in os.listdir(SUMMARY_KB_PATH) if os.path.isdir(os.path.join(SUMMARY_KB_PATH, f))]
    
    if rfp_checklists:
        selected_checklist = st.selectbox("Select RFP Checklist", rfp_checklists)
        if st.button("🗑️ Delete Checklist", use_container_width=True):
            import shutil
            shutil.rmtree(SUMMARY_KB_PATH / selected_checklist)
            st.success(f"Deleted {selected_checklist}")
            st.rerun()
    else:
        st.info("No RFP checklists found")
    
    st.write("### ⚙️ Settings")
    
    # RFP checklist name
    kb_name = st.text_input(
        "RFP Checklist Name",
        value=st.session_state["summary_kb_name"],
        help="Name for the RFP checklist"
    )
    st.session_state["summary_kb_name"] = kb_name
    
    # LLM settings
    llm_type = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gemma2:27b", "gemma2:9b", "llama3.1:8b", "gpt-4o"],
        help="Choose the language model for summarization"
    )
    
    # Save to RFP checklist option
    save_to_kb = st.checkbox("💾 Save checklists to knowledge base", value=True, help="Save RFP checklists for use in RAG chatbot")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 📄 **Document Preview**")
    with st.container(height=600):
        if st.session_state["uploaded_files"]:
            num_tabs = min(len(st.session_state["uploaded_files"]), 5)
            
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
                                pdf_display = display_pdf(file_data, scale=0.8, height=520)
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            elif file_extension in [".docx", ".docc"]:
                                docx_display = display_docx(file_data, scale=0.8, height=520)
                                st.markdown(docx_display, unsafe_allow_html=True)
                            else:
                                st.error("Unsupported file format")
                        else:
                            st.write("No document uploaded")
        else:
            st.info("📁 Upload documents to see preview here")

with col2:
    st.write("### 📋 **RFP Analysis**")
    with st.container(height=600):
        if st.session_state["uploaded_files"]:
            # Create RFP To-Do extraction prompt
            summary_prompt = PromptTemplate(
                input_variables=["text"],
                template="""You are an expert RFP (Request for Proposal) analyst. Your task is to extract To-Do requirements and checkpoints from RFP documents.

Document content:
{text}

Please extract the following information in clean bullet point format. Return ONLY the formatted content without any additional metadata or technical details:

**TO-DO REQUIREMENTS:**
• [List all actionable items, deliverables, and requirements that need to be completed]

**CHECKPOINTS:**
• [List all deadlines, milestones, review points, and key dates - each checkpoint should be on its own line with a bullet point]

IMPORTANT FORMATTING RULES:
- Each checkpoint must be on a separate line with its own bullet point (•)
- Use proper bullet point formatting for all items
- Be concise but comprehensive
- If a section doesn't apply, write "None" for that section

Return only the formatted analysis content:"""
            )
            
            # Process each document
            for i, file in enumerate(st.session_state["uploaded_files"]):
                file_name = file.name
                
                st.write(f"**📄 {file_name}**")
                
                # Check if summary already exists
                if file_name in st.session_state["summaries"]:
                    st.write("**RFP Analysis:**")
                    # Display the analysis in a clean, formatted way
                    analysis_content = st.session_state["summaries"][file_name]
                    
                    # Use a container with better styling
                    with st.container():
                        st.markdown(analysis_content)
                    
                    if st.button(f"🔄 Regenerate Analysis", key=f"regenerate_{i}"):
                        del st.session_state["summaries"][file_name]
                        st.rerun()
                else:
                    if st.button(f"📋 Extract To-Do Items", key=f"summarize_{i}"):
                        with st.spinner(f"Analyzing RFP document: {file_name}..."):
                            try:
                                # Save file temporarily
                                temp_path = f"temp_{file_name}"
                                with open(temp_path, "wb") as f:
                                    f.write(file.getvalue())
                                
                                # Load document
                                if st.session_state["doc_format"] == "pdf":
                                    loader = get_pdf_loader(file_path=temp_path, type="pypdf")
                                elif st.session_state["doc_format"] in ["docx", "docc"]:
                                    loader = get_docx_loader(file_path=temp_path, type="docx2txt")
                                
                                docs = loader.load()
                                
                                # Split document into chunks (using default size for RFP analysis)
                                splitter = get_splitter(
                                    chunk_size=3000,  # Larger chunks for better context
                                    chunk_overlap=300,
                                    type="RecursiveCT"
                                )
                                chunks = splitter.split_documents(docs)
                                
                                # Get LLM (using default temperature for consistent RFP analysis)
                                llm = get_llm(
                                    model=llm_type,
                                    temperature=0.1,  # Low temperature for consistent extraction
                                    streaming=False,
                                    base_url=os.getenv("OLLAMA_BASE_URL"),
                                )
                                
                                # Combine chunks and analyze
                                full_text = "\n\n".join([chunk.page_content for chunk in chunks])
                                
                                # Create chain
                                chain = summary_prompt | llm
                                
                                # Generate summary
                                response = chain.invoke({"text": full_text})
                                
                                # Extract only the content from the response
                                if hasattr(response, 'content'):
                                    summary = response.content
                                else:
                                    summary = str(response)
                                
                                # Clean up any remaining metadata or technical details
                                # Remove common metadata patterns
                                metadata_patterns = [
                                    'additional_kwargs',
                                    'response_metadata',
                                    'token_usage',
                                    'model_name',
                                    'system_fingerprint',
                                    'finish_reason',
                                    'logprobs',
                                    'usage_metadata',
                                    'completion_tokens',
                                    'prompt_tokens',
                                    'total_tokens',
                                    'cached_tokens',
                                    'audio_tokens',
                                    'reasoning_tokens',
                                    'accepted_prediction_tokens',
                                    'rejected_prediction_tokens',
                                    'input_tokens',
                                    'output_tokens'
                                ]
                                
                                # Split into lines and filter out metadata
                                lines = summary.split('\n')
                                clean_lines = []
                                skip_metadata = False
                                
                                for line in lines:
                                    # Check if line contains metadata
                                    if any(pattern in line for pattern in metadata_patterns):
                                        skip_metadata = True
                                        continue
                                    
                                    # Skip empty lines after metadata
                                    if skip_metadata and not line.strip():
                                        continue
                                    
                                    # Reset skip flag when we find content again
                                    if skip_metadata and line.strip() and not any(pattern in line for pattern in metadata_patterns):
                                        skip_metadata = False
                                    
                                    # Add clean content lines
                                    if not skip_metadata:
                                        clean_lines.append(line)
                                
                                summary = '\n'.join(clean_lines).strip()
                                
                                # Store summary
                                st.session_state["summaries"][file_name] = summary
                                
                                # Save to knowledge base if enabled
                                if save_to_kb:
                                    try:
                                        kb_dir = SUMMARY_KB_PATH / st.session_state["summary_kb_name"]
                                        os.makedirs(kb_dir, exist_ok=True)
                                        
                                        # Create a document from the RFP analysis
                                        summary_doc = Document(
                                            page_content=summary,
                                            metadata={
                                                "source": file_name,
                                                "type": "rfp_analysis",
                                                "analysis_type": "to_do_checkpoints",
                                                "created_at": datetime.now().isoformat()
                                            }
                                        )
                                        
                                        # Save summary as JSON
                                        summary_data = {
                                            "content": summary,
                                            "metadata": summary_doc.metadata,
                                            "original_file": file_name
                                        }
                                        
                                        summary_file = kb_dir / f"{file_name}_summary.json"
                                        with open(summary_file, "w", encoding="utf-8") as f:
                                            json.dump(summary_data, f, ensure_ascii=False, indent=2)
                                        
                                        # Create or update vector store
                                        embedding_model = os.getenv("SELECTED_EMBEDDING")
                                        if not embedding_model:
                                            embedding_model = "nomic-embed-text"  # Default fallback
                                        
                                        embedding = get_embedding(
                                            model=embedding_model,
                                            base_url=os.getenv("OLLAMA_BASE_URL"),
                                        )
                                        
                                        # Load existing summaries or create new
                                        existing_summaries = []
                                        for f in os.listdir(kb_dir):
                                            if f.endswith("_summary.json"):
                                                with open(kb_dir / f, "r", encoding="utf-8") as sf:
                                                    data = json.load(sf)
                                                    doc = Document(
                                                        page_content=data["content"],
                                                        metadata=data["metadata"]
                                                    )
                                                    existing_summaries.append(doc)
                                        
                                        if existing_summaries:
                                            vectorstore = get_vector_store(
                                                documents=existing_summaries,
                                                embedding=embedding
                                            )
                                            vectorstore.save_local(folder_path=kb_dir / "db")
                                        
                                        st.success(f"✅ RFP checklist saved to knowledge base: {st.session_state['summary_kb_name']}")
                                        
                                    except Exception as kb_error:
                                        st.warning(f"⚠️ RFP analysis completed but failed to save to KB: {str(kb_error)}")
                                
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                
                                st.success(f"✅ RFP analysis completed for {file_name}")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error generating summary: {str(e)}")
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                
                st.write("---")
            
            # Clear all analyses button
            if st.session_state["summaries"]:
                if st.button("🗑️ Clear All Analyses", use_container_width=True):
                    st.session_state["summaries"] = {}
                    st.rerun()
                    
        else:
            st.info("📁 Upload RFP documents to extract To-Do requirements and checkpoints")

# Display all analyses in a collapsible section
if st.session_state["summaries"]:
    with st.expander("📋 **View All RFP Analyses**", expanded=False):
        for file_name, analysis in st.session_state["summaries"].items():
            st.write(f"### 📄 {file_name}")
            st.markdown(analysis)
            st.write("---")
