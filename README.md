# IntraSeek

A comprehensive document processing and Q&A application with integrated RFP analysis workflow.

## 🚀 Overview

IntraSeek is a Streamlit-based application that combines document processing, RFP analysis, and intelligent Q&A capabilities. It provides a seamless workflow for uploading documents, extracting RFP requirements, and asking questions with enhanced context.

## ✨ Key Features

- **📄 Document Upload for RAG**: Upload PDF/DOCX documents to create searchable knowledge bases
- **📋 RFP To-Do Extractor**: Extract To-Do requirements and checkpoints from RFP documents
- **💬 RAG Chatbot**: Ask questions about documents with optional RFP checklist context
- **🔄 Integrated Workflow**: Seamlessly combine document knowledge with RFP requirements
- **🎯 Smart Context**: Optionally include RFP checklist context for enhanced responses

## 🏗️ Architecture

```
intraseek/
│
├── demo/                           # Streamlit application
│   ├── IntraSeek.py               # Main application entry point
│   ├── pages/                     # Application pages
│   │   ├── A.Document_Upload_RAG.py
│   │   ├── B.RFP_To-Do_Extractor.py
│   │   └── C.RAG_Chatbot.py
│   ├── function_utils.py          # Utility functions
│   ├── img/                       # Application images
│   ├── env_template.txt           # Environment variables template
│   └── setup_env.py              # Environment setup script
│
├── src/intraseek/                 # Core package
│   ├── modules/                   # Core modules
│   │   ├── documents.py          # Document processing
│   │   ├── llms.py               # LLM integration
│   │   ├── prompts.py            # Prompt management
│   │   └── vector_db.py          # Vector database operations
│   ├── utils/                     # Utilities
│   │   ├── config_loader.py      # Configuration management
│   │   ├── path.py               # Path utilities
│   │   └── rag_utils.py          # RAG utilities
│   └── chains.py                  # LLM chains
│
└── configs/                       # Configuration files
    └── prompt_template/           # Prompt templates
```

## 🛠️ Requirements

- **OS**: Windows 11 Enterprise (tested)
- **Python**: 3.11.9
- **Dependencies**: See `requirements.txt`

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd intraseek
   ```

2. **Create virtual environment**:
   ```bash
   conda create -n intraseek python=3.11.9
   conda activate intraseek
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Copy the template
   cp demo/env_template.txt .env
   
   # Edit .env with your API keys and preferences
   # Or run the setup script
   python demo/setup_env.py
   ```

## ⚙️ Environment Configuration

Create a `.env` file with the following variables:

### OpenAI Configuration (Recommended)
```bash
OPENAI_API_KEY=your_openai_api_key_here
SELECTED_EMBEDDING=text-embedding-3-large
SELECTED_LLM=gpt-4o-mini
```

### Ollama Configuration (Alternative)
```bash
OLLAMA_BASE_URL=http://localhost:11434
SELECTED_EMBEDDING=nomic-embed-text
SELECTED_LLM=llama3.1:8b
```

## 🚀 Usage

1. **Start the application**:
   ```bash
   cd demo
   streamlit run IntraSeek.py
   ```

2. **Navigate through the pages**:
   - **📄 Document Upload (RAG)**: Upload documents to create knowledge bases
   - **📋 RFP To-Do Extractor**: Extract requirements from RFP documents
   - **💬 RAG Chatbot**: Ask questions with optional RFP context

## 🔄 Workflow

### 1. Document Upload for RAG
- Upload PDF/DOCX documents
- Configure chunk size, overlap, and embedding model
- Create searchable vector database
- Save RAG configuration for later use

### 2. RFP To-Do Extractor
- Upload RFP documents
- Extract To-Do requirements and checkpoints
- Generate structured bullet-point lists
- Save as RFP checklists for context enhancement

### 3. RAG Chatbot
- Select knowledge source (RAG documents)
- Optionally include RFP checklist context
- Ask questions with enhanced responses
- View document previews and chat history

## 🎯 Use Cases

- **RFP Analysis**: Extract actionable requirements and deadlines
- **Document Q&A**: Get answers from uploaded documents
- **Project Management**: Track deliverables and milestones
- **Proposal Preparation**: Understand requirements for better proposals
- **Knowledge Management**: Create searchable document repositories

## 🔧 Technical Details

- **Vector Database**: FAISS for efficient similarity search
- **Embeddings**: OpenAI or Ollama embeddings
- **LLMs**: GPT-4o-mini, Gemma2, Llama3.1 support
- **Document Processing**: PDF and DOCX support
- **UI Framework**: Streamlit with custom styling

## 📝 License

This project is developed for internal use by D&AI team.

## 👥 Contributors

- 박지수 (D&AI)
- 박정수 (D&AI)  
- 이민우 (D&AI)# Document Assistant - New Streamlit Application

A comprehensive document processing and Q&A application built with Streamlit.

## Features

### 📄 Document Upload for RAG
- Upload PDF and DOCX files to create searchable knowledge bases
- Configure document processing settings (chunk size, overlap, etc.)
- Customize prompts for question-answering
- Save collections for later use

### 📝 Document Summarization
- Upload documents to get AI-powered summaries
- Choose summary length (Brief, Medium, Detailed)
- Support for multiple documents
- Regenerate summaries as needed

### 💬 RAG Chatbot
- Ask questions about uploaded documents
- Interactive chat interface with memory
- Configurable LLM settings
- Real-time document retrieval

## How to Run

1. **Navigate to the demo directory:**
   ```bash
   cd demo
   ```

2. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage Workflow

### Step 1: Create RAG Collection
1. Go to "Document Upload (RAG)" page
2. Upload PDF or DOCX files
3. Configure prompt settings
4. Set vector database parameters
5. Click "Create RAG Collection"

### Step 2: Summarize Documents (Optional)
1. Go to "Document Summarization" page
2. Upload documents for summarization
3. Adjust LLM and summarization settings
4. Generate summaries for each document

### Step 3: Chat with Documents
1. Go to "RAG Chatbot" page
2. Select your created RAG collection
3. Activate the chatbot
4. Start asking questions about your documents

## File Structure

```
demo/
├── app.py                              # Main application entry point
├── pages/
│   ├── 01.Document_Upload_RAG.py      # RAG document uploader
│   ├── 02.Document_Summarization.py   # Document summarization
│   └── 03.RAG_Chatbot.py              # RAG chatbot interface
```

## Requirements

### Environment Variables

Create a `.env` file in the `demo` directory with the following variables:

```bash
# LLM Configuration
SELECTED_LLM=gpt-4o-mini
# Alternative options: gemma2:27b, gemma2:9b, llama3.1:8b

# Embedding Model Configuration  
SELECTED_EMBEDDING=nomic-embed-text
# Alternative options:
# - For OpenAI: text-embedding-3-large, text-embedding-3-small
# - For Ollama: mxbai-embed-large, nomic-embed-text, bge-m3, jeffh/intfloat

# Ollama Configuration (if using local models)
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Configuration (if using OpenAI models)
# OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: If environment variables are not set, the app will use default fallback values:
- Default LLM: `gpt-4o` (if available)
- Default Embedding: `nomic-embed-text`

You can use the `env_template.txt` file as a starting point.

## Dependencies

The application uses the existing `intraseek` package and its modules:
- Document processing (`intraseek.modules.documents`)
- LLM integration (`intraseek.modules.llms`)
- Vector database (`intraseek.modules.vector_db`)
- Utility functions (`intraseek.utils.*`)

## Notes

- All uploaded documents are processed and stored locally
- RAG collections are saved in the `logs/rag/` directory
- The application supports both local (Ollama) and cloud-based LLMs
- Document previews are available for uploaded files
- Chat history is maintained during the session
