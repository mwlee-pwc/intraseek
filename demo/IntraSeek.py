import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="IntraSeek",
    page_icon=Image.open("img/pwc_logo.png"),
    layout="wide",
)

with st.sidebar:
    st.success("Select a page above")
    st.sidebar.markdown("-------")
    
    # Environment status
    with st.expander("🔧 Environment Status", expanded=False):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check embedding model
        embedding_model = os.getenv("SELECTED_EMBEDDING")
        if embedding_model:
            if embedding_model.startswith("text-embedding"):
                st.success(f"✅ OpenAI Embedding: {embedding_model}")
                if not os.getenv("OPENAI_API_KEY"):
                    st.warning("⚠️ OPENAI_API_KEY not set")
            else:
                st.info(f"🔄 Ollama Embedding: {embedding_model}")
        else:
            st.warning("⚠️ SELECTED_EMBEDDING not set")
            if os.getenv("OPENAI_API_KEY"):
                st.info("💡 Will use OpenAI fallback")
            else:
                st.info("💡 Will use Ollama fallback")
        
        # Check LLM
        llm_model = os.getenv("SELECTED_LLM")
        if llm_model:
            st.success(f"✅ LLM: {llm_model}")
        else:
            st.warning("⚠️ SELECTED_LLM not set")
    
    st.text("Made by: D&AI 박지수, 박정수, 이민우")
    st.sidebar.markdown("-------")
    st.image("img/pwc_logo.png", width=250)

st.markdown(
    """
## IntraSeek
###

A comprehensive document processing and Q&A application with integrated workflow:

- **📄 Document Upload for RAG**: Upload documents to create a searchable knowledge base
- **📋 RFP To-Do Extractor**: Upload RFP documents to extract To-Do requirements and checkpoints
- **💬 RAG Chatbot**: Ask questions about your documents with optional RFP checklist context

### 🔄 Integrated Workflow:

1. **📄 Upload Documents for RAG**: Create detailed knowledge bases from your PDF/DOCX files
2. **📋 Extract RFP Requirements**: Upload RFP documents to extract To-Do items and checkpoints  
3. **💬 Enhanced Chat**: Ask questions about documents with optional RFP checklist context:
   - **Documents only**: Standard document-based Q&A
   - **Documents + RFP context**: Enhanced responses with requirement awareness

### ✨ Key Features:
- **Conditional Context**: Optionally include RFP checklist context in document-based chat
- **Smart Integration**: RFP checklists are automatically saved as searchable knowledge bases
- **Flexible Q&A**: Choose between document-only or enhanced RFP-aware responses
- **Document Preview**: View both original documents and extracted RFP requirements
- **Vector Database**: Efficient retrieval from both document content and RFP checklists
- **AI-Powered**: Advanced language models for both RFP analysis and Q&A

### 💡 Use Cases:
- **RFP Analysis**: Extract To-Do requirements and checkpoints from RFP documents
- **Quick Overview**: Use RFP analyses for fast insights on requirements and deadlines
- **Deep Analysis**: Use full documents for detailed, specific information
- **Project Management**: Track deliverables and milestones from RFP documents
- **Proposal Preparation**: Understand requirements and deadlines for better proposals
""",
)
