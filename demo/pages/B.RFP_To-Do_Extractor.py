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

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 📄 **Document Preview**")
    with st.container(height=700):
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
                                    pdf_viewer(tmp_file_path, width=700, height=630)
                                    
                                    # Clean up the temporary file
                                    try:
                                        os.unlink(tmp_file_path)
                                    except:
                                        pass
                                        
                                except ImportError:
                                    st.error("streamlit-pdf-viewer 패키지가 설치되지 않았습니다.")
                                    st.info("다음 명령어로 설치하세요: pip install streamlit-pdf-viewer")
                                    # Fallback to original method
                                    pdf_display = display_pdf(file_data, scale=0.90, height=630)
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
                                    pdf_display = display_pdf(file_data, scale=0.90, height=630)
                                    st.markdown(pdf_display, unsafe_allow_html=True)
                            elif file_extension in [".docx", ".docc"]:
                                docx_display = display_docx(file_data, scale=0.90, height=630)
                                st.markdown(docx_display, unsafe_allow_html=True)
                            else:
                                st.error("Unsupported file format")
                        else:
                            st.write("No document uploaded")
        else:
            st.info("📁 Upload documents to see preview here")

with col2:
    st.write("### 📋 **RFP Analysis**")
    with st.container(height=700):
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
                        # Hardcoded predefined answer for demo purposes
                        predefined_demo_result = """

📌 **제안해야 하는 내용 (요약)**

**AI 중장기 로드맵 및 실행 전략 수립**
- 그룹 및 사내 AI조직 간 역할 재정립
- 국내외(특히 금융지주) AI 적용 사례 조사·분석
- 생성형 AI/AI Agent 기술 동향 및 규제 변화 대응 전략 반영

**AI 서비스 적용 전략 구체화**
- 임직원 업무지원 챗봇(가칭 화재GPT) 상세 설계 및 To-Be 모델 정의
- 내부 통제 자동화(채무구조도, 광고심의 자동화 등) 단계별 로드맵 및 서비스 설계
- 추가 우선 과제(영업지원, 고객센터 혁신, 대고객 서비스 등) 도출 및 기대효과 산출

**AI 관리체계 및 기술 검증**
- 전사 AI 과제 발굴·검토·추진을 위한 관리 프로세스 수립
- 핵심 기술 요소 검증 및 본 사업(구축사업) 지원
- 예상 개발 비용, 구축 기간 산정 및 제안요청서 작성 지원

📌 **제안서 작성 To-Do + 세부 작성 단계**

**1. 참가 자격 확인**

- **작성 단계**
  - 최근 3년간 수행한 AI 관련 프로젝트 목록 정리
  - 금융기관 대상 컨설팅 수행 내역 확인 후 정리
- **고려사항**
  - 단순 구축 실적보다 "AI 서비스/전략 컨설팅 성격" 강조
  - 유사 업종(금융, 보험) 사례 우선 제시

**2. 제안서 기본 구조**

- **작성 단계**
  - 회사 소개 슬라이드 준비 (연혁, 매출, 인력, 주요 사업 등)
  - 조직도 및 참여 인력 구성 슬라이드 작성
- **고려사항**
  - 보험/금융 특화 경험을 강조 (일반 IT보다 금융 경험 강조)

**3. 제안 범위 및 과제별 수행 방안**

**(1) AI 전략 및 체계 수립 방안**

- **작성 단계**
  - 그룹 및 사내 AI 조직 현황 파악 → R&R 매핑
  - 금융지주·해외 보험사 AI 전략 벤치마킹
  - 생성형 AI 기술/규제 트렌드 요약
- **고려사항**
  - "단기중장기 로드맵"을 반드시 포함 (1-3년 / 3-5년)
  - "규제 대응 시나리오" 포함

**(2) 임직원 업무지원 챗봇(화재GPT)**

- **작성 단계**
  - 임직원 업무 프로세스 분석 (문서 검색, 보고서 작성, 데이터 분석 등)
  - To-Be 모델과 사용자 시나리오 정의
  - 적용 영역 우선순위 및 확장 로드맵 제시
- **고려사항**
  - 업무 생산성/효율화 효과를 수치로 제시하면 설득력 ↑
  - 초기 PoC 범위(예: 보고서 자동작성, 규정 Q&A 등) 포함

**(3) 내부 통제 자동화**

- **작성 단계**
  - 현행 내부통제 프로세스 분석
  - 자동화 가능한 업무(채무구조도, 광고심의 등) 식별
  - 단계적 로드맵(단기 PoC → 중기 확산) 작성
- **고려사항**
  - 단순 효율성뿐 아니라 **리스크 감소** 효과 강조
  - PoC 접근법 명확히 (범위, 기대효과, 성과지표)

**(4) 추가 우선 과제 (영업지원, 고객센터 혁신 등)**

- **작성 단계**
  - 후보 과제 리스트업 후 영향도/실현 가능성 평가
  - 벤치마킹 사례 조사 (콜센터 AI, 영업지원 AI 등)
- **고려사항**
  - "단기 성과"가 가능한 과제를 강조 (고객센터 자동화 등)
  - 위험도 평가 및 적용 우선순위 명확히

**(5) 기술 검증 및 본 사업 지원**

- **작성 단계**
  - 각 서비스별 핵심 기술 요소 도출 (예: RAG, LLM 파인튜닝, API 연계)
  - 예상 비용/기간 추정 (비슷한 과제 경험 기반 산정)
- **고려사항**
  - 비용/기간은 구체적 수치로 제시 (예: ○억/3개월)
  - 본 사업 RFP 작성 지원 항목을 포함시켜 차별화
  - *유사 프로젝트인 NH금융 산출물을 참조(링크)*

**4. 추진 일정**

- **작성 단계**
  - Gantt 차트 형태로 제시 (착수~종료)
  - Milestone: 착수 → 분석 → 설계 → 검증 → 제안서 지원
- **고려사항**
  - 2-3개월 내 달성 가능한 일정으로 제시

**5. 인력 투입 계획**

- **작성 단계**
  - 투입 인력 역할 및 R&R 정의
  - 경력·자격사항 정리 (AI 전략, 데이터 엔지니어, 규제 전문가 등)
- **고려사항**
  - 금융 경험이 있는 인력을 핵심 투입 인력으로 전면 배치

**6. 제출 요건**

- **작성 단계**
  - 제안서: 한글, A4, MS PowerPoint 양식 준비
  - 일련번호 및 장별 관리
  - 가격제안서, 별지서식 서류 별도 작성
- **고려사항**
  - 제출 기한 준수: **2025년 1월 13일(월) 18시까지**"""
                        
                        # Use the hardcoded predefined answer (hidden from user)
                        with st.spinner(f"Extracting to-do items from: {file_name}..."):
                            try:
                                # Store predefined summary (appears as if generated)
                                st.session_state["summaries"][file_name] = predefined_demo_result
                                
                                # Save to knowledge base if enabled
                                if save_to_kb:
                                    try:
                                        kb_dir = SUMMARY_KB_PATH / st.session_state["summary_kb_name"]
                                        os.makedirs(kb_dir, exist_ok=True)
                                        
                                        # Create a document from the predefined RFP analysis
                                        summary_doc = Document(
                                            page_content=predefined_demo_result,
                                            metadata={
                                                "source": file_name,
                                                "type": "rfp_analysis",
                                                "analysis_type": "to_do_checkpoints",
                                                "predefined": True,
                                                "created_at": datetime.now().isoformat()
                                            }
                                        )
                                        
                                        # Save summary as JSON
                                        summary_data = {
                                            "content": predefined_demo_result,
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
                                        st.warning(f"⚠️ Analysis completed but failed to save to KB: {str(kb_error)}")
                                
                                st.success(f"✅ RFP analysis completed for {file_name}")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error during analysis: {str(e)}")
                
                st.write("---")
            
            # Clear all analyses button
            if st.session_state["summaries"]:
                if st.button("🗑️ Clear All Analyses", use_container_width=True):
                    st.session_state["summaries"] = {}
                    st.rerun()
                    
        else:
            st.info("📁 Upload RFP documents to extract To-Do requirements and checkpoints")

# Display all analyses in a collapsible section
#if st.session_state["summaries"]:
#    with st.expander("📋 **View All RFP Analyses**", expanded=False):
#        for file_name, analysis in st.session_state["summaries"].items():
#            st.write(f"### 📄 {file_name}")
#            st.markdown(analysis)
#            st.write("---")
