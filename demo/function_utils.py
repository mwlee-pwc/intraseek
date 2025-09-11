import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import base64
import os
import re
import shutil
from io import BytesIO
from time import time
from typing import Any, Dict, List, Tuple, Optional
from difflib import SequenceMatcher

import mammoth
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

from intraseek.modules.llms import get_embedding
from intraseek.modules.vector_db import get_splitter, get_vector_store
from intraseek.utils.path import CACHE_EMBEDDING_PATH, CACHE_FILE_PATH, DEMO_IMG_PATH

load_dotenv()

human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avartar = DEMO_IMG_PATH / "pwc_logo.png"


def load_image(image_path: str) -> bytes:
    """
    주어진 경로에서 이미지를 읽어와 바이트 형태로 반환하는 함수.

    Args:
        image_path (str): 이미지 파일의 경로.

    Returns:
        bytes: 이미지 파일의 바이트 데이터.
    """
    with open(image_path, "rb") as image_file:
        return image_file.read()


class ChatCallbackHandler(BaseCallbackHandler):
    """
    LLM 모델의 실행 중 상태를 처리하는 콜백 핸들러 클래스.

    이 핸들러는 메시지 박스를 업데이트하고, 새로운 토큰을 수신할 때마다 실시간으로
    사용자 인터페이스에 반영.

    Attributes:
        message (str): 현재까지 생성된 메시지를 저장하는 문자열.
        message_box: Streamlit에서 비어 있는 UI 요소로, 생성된 메시지를 실시간으로 업데이트하는 데 사용.
    """

    message: str = ""

    def on_llm_start(self, *args, **kwargs) -> None:
        """
        LLM 모델이 시작될 때 호출되는 메서드.
        빈 메시지 박스를 생성.

        Args:
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs) -> None:
        """
        LLM 모델이 종료될 때 호출되는 메서드.
        최종 생성된 메시지를 저장.

        Args:
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        """
        새로운 토큰을 수신할 때 호출되는 메서드.
        메시지 박스를 실시간으로 업데이트.

        Args:
            token (str): 새로 생성된 토큰.
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data
def display_pdf(file_data: bytes, scale: float = 1.0, height: int = 700) -> str:
    """
    PDF 파일 데이터를 브라우저에 표시하는 함수.

    주어진 PDF 파일 데이터를 base64로 인코딩하고, 이를 HTML iframe을 통해 PDF로 표시.

    Args:
        file_data (bytes): PDF 파일의 바이트 데이터.
        scale (float, optional): PDF를 브라우저에 표시할 때의 확대/축소 비율. 기본값은 1.0.
        height (int, optional): iframe의 높이 (픽셀 단위). 기본값은 700px.

    Returns:
        pdf_display (str): PDF를 표시하기 위한 HTML iframe 요소가 포함된 문자열
    """
    base64_pdf = base64.b64encode(file_data).decode("utf-8")

    pdf_display = f"""
    <div style="display: flex; justify-content: center;">
        <iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=0"
        width="100%" height="{height}px"
        style="border:none; transform: scale({scale}); transform-origin: top center;"
        type="application/pdf"></iframe>
    </div>
    """
    return pdf_display




def convert_docx_to_html(file_data: bytes) -> str:
    """
    docx 파일 데이터를 HTML로 변환하는 함수.

    주어진 docx 파일의 바이트 데이터를 mammoth 라이브러리를 사용하여 HTML 문자열로 변환.
    변환된 HTML에는 원본 문서의 텍스트 내용과 기본적인 서식이 포함.

    Args:
        file_data (bytes): docx 파일의 바이트 데이터.

    Returns:
        html (str): 변환된 HTML 문자열.
    """
    result = mammoth.convert_to_html(BytesIO(file_data))

    html = result.value

    return html


@st.cache_data
def display_docx(file_data: bytes, scale: float = 1.0, height: int = 700) -> str:
    """
    docx 파일을 스타일이 적용된 HTML iframe으로 변환하여 표시하는 함수.

    HTML로 변환된 docx 파일을, iframe 내에서 표시.

    Args:
        file_data (bytes): html 문자열.
        scale (float, optional): 문서를 브라우저에 표시할 때의 확대/축소 비율. 기본값은 1.0.
        height (int, optional): iframe의 높이 (픽셀 단위). 기본값은 700px.

    Returns:
        iframe_template (str): HTML iframe 템플릿 문자열 또는 오류 메시지.
    """
    try:
        html_content = convert_docx_to_html(file_data)

        styled_html = f"""
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
            .docx-container {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; font-size: 14px; }}
            .docx-container img {{ max-width: 100%; height: auto; }}
            .docx-container table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
            .docx-container td, .docx-container th {{
                border: 1px solid #ddd;
                padding: 8px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                min-width: 50px;
            }}
        </style>
        <div class="docx-container">
            {html_content}
        </div>
        """

        encoded_html = base64.b64encode(styled_html.encode("utf-8")).decode("utf-8")

        iframe_template = f"""
        <iframe src="data:text/html;base64,{encoded_html}"
        width="100%" height="{height}px"
        style="border:none; transform: scale({scale}); transform-origin: top center;">
        </iframe>
        """

        return iframe_template
    except Exception as e:
        return f"문서를 처리하는 중 오류가 발생했습니다: {str(e)}"


@st.cache_resource(show_spinner="Embedding file...")
def embed_file_with_cache(
    file,
    file_content: bytes,
    file_type: str = "pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "text-embedding-3-large",
    base_url: str = "10.99.15.72:11434",
):
    """
    파일을 임베딩하고 리트리버를 생성하는 함수.

    주어진 파일 데이터를 임베딩하고, 이를 기반으로 검색 가능한 리트리버 객체를 생성.
    PDF 또는 DOCX 파일을 지원하며, 문서를 지정된 크기로 분할한 후, 지정된 임베딩 모델을 사용해 벡터 임베딩을 생성.
    생성된 임베딩은 캐시에 저장되며, 이를 사용하여 FAISS 벡터스토어와 리트리버를 생성.

    Args:
        file: 업로드된 파일 객체.
        file_content (bytes): 업로드된 파일의 바이트 데이터.
        file_type (str, optional): 파일 유형. "pdf" 또는 "docx"로 지정. 기본값은 "pdf".
        chunk_size (int, optional): 문서를 분할할 때 사용하는 청크의 크기. 기본값은 1000.
        chunk_overlap (int, optional): 문서를 분할할 때 청크 간 중첩되는 문자 수. 기본값은 200.
        embedding_model (str, optional): 사용할 임베딩 모델의 이름. 기본값은 "text-embedding-3-large".
        base_url (str, optional): Locall LLM 사용시 서버 주소. 기본값은 "10.99.15.72:11434"

    Returns:
        retriever: 검색 가능한 리트리버 객체.
    """
    file_path = CACHE_FILE_PATH / f"{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(CACHE_EMBEDDING_PATH / f"{file.name}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type in [".docx", ".docc"]:
        loader = Docx2txtLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = get_embedding(model=embedding_model, base_url=base_url)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(
    file,
    file_content: bytes,
    file_type: str = "pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "text-embedding-3-large",
    base_url: str = "10.99.15.72:11434",
):
    """
    파일을 임베딩하고 리트리버를 생성하는 함수.

    주어진 파일 데이터를 임베딩하고, 이를 기반으로 검색 가능한 리트리버 객체를 생성.
    PDF 또는 DOCX 파일을 지원하며, 문서를 지정된 크기로 분할한 후, 지정된 임베딩 모델을 사용해 벡터 임베딩을 생성.
    이를 사용하여 FAISS 벡터스토어와 리트리버를 생성.

    Args:
        file: 업로드된 파일 객체.
        file_content (bytes): 업로드된 파일의 바이트 데이터.
        file_type (str, optional): 파일 유형. "pdf" 또는 "docx"로 지정. 기본값은 "pdf".
        chunk_size (int, optional): 문서를 분할할 때 사용하는 청크의 크기. 기본값은 1000.
        chunk_overlap (int, optional): 문서를 분할할 때 청크 간 중첩되는 문자 수. 기본값은 200.
        embedding_model (str, optional): 사용할 임베딩 모델의 이름. 기본값은 "text-embedding-3-large".
        base_url (str, optional): Locall LLM 사용시 서버 주소. 기본값은 "10.99.15.72:11434"

    Returns:
        retriever: 검색 가능한 리트리버 객체.
    """
    os.makedirs(CACHE_FILE_PATH, exist_ok=True)
    file_path = CACHE_FILE_PATH / f"{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, type="recursive")

    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type in [".docx", ".docc"]:
        loader = Docx2txtLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = get_embedding(model=embedding_model, base_url=base_url)
    vectorstore = get_vector_store(docs, embeddings, type="faiss")
    retriever = vectorstore.as_retriever()

    if os.path.exists(file_path):
        os.remove(file_path)

    return retriever


@st.cache_resource(show_spinner="Loading vectorstore...")
def load_retriver(
    db_path: str,
    embedding_model="text-embedding-3-large",
    retriever_k: int = 4,
    base_url: str = "10.99.15.72:11434",
):
    """
    로컬에서 저장된 벡터스토어를 불러와 리트리버를 생성하는 함수.

    주어진 경로에 있는 벡터스토어를 로드하고, 지정된 임베딩 모델을 사용해 검색 가능한 리트리버 객체를 생성.
    리트리버는 유사성 검색 방식을 사용하며, 검색 결과로 반환되는 문서의 수는 `retriever_k` 값으로 설정.

    Args:
        db_path (str): 로컬 벡터스토어가 저장된 경로.
        embedding_model (str, optional): 사용할 임베딩 모델의 이름. 기본값은 "text-embedding-3-large".
        retriever_k (int, optional): 검색 시 반환할 문서의 최대 개수. 기본값은 4.
        base_url (str, optional): Locall LLM 사용시 서버 주소. 기본값은 "10.99.15.72:11434"

    Returns:
        retriever: 검색 가능한 리트리버 객체.
    """
    embeddings = get_embedding(model=embedding_model, base_url=base_url)
    vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": retriever_k})
    return retriever


@st.dialog("로그 삭제")
def delete_log(selected_log_path: str) -> None:
    """
    선택한 로그를 삭제하는 함수.

    사용자는 주어진 경로에 있는 로그 삭제를 확인.
    삭제 버튼을 클릭하면 지정된 경로의 로그가 삭제.
    삭제 과정에서 오류가 발생할 경우, 오류 메시지가 표시.

    Args:
        selected_log_path (str): 삭제할 로그가 위치한 파일 경로.
    """
    st.write(f"`{selected_log_path.as_posix().split('/')[-1]}`")
    st.write("위 경로의 모델을 삭제합니까?")
    if st.button("삭제"):
        try:
            shutil.rmtree(selected_log_path)
            st.success(f"모델 '{selected_log_path.as_posix().split('/')[-1]}'이 삭제되었습니다.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"모델 삭제 중 오류가 발생했습니다: {str(e)}")
        st.rerun()


def save_message(message: str, role: str) -> None:
    """
    메시지를 세션 상태에 저장하는 함수.

    Args:
        message (str): 저장할 메시지 내용.
        role (str): 메시지 작성자의 역할 ("ai" 또는 "human").
    """
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message: str, role: str, save: bool = True) -> None:
    """
    채팅 인터페이스에 메시지를 표시하고 선택적으로 저장하는 함수.

    메시지를 채팅 UI에 표시하고, save 매개변수가 True인 경우 세션 상태에도 저장.
    각 메시지는 역할에 따라 다른 아바타 이미지와 함께 표시.

    Args:
        message (str): 표시할 메시지 내용.
        role (str): 메시지 작성자의 역할 ("ai" 또는 "human").
        save (bool, optional): 메시지를 세션 상태에 저장할지 여부. 기본값은 True.
    """
    avatar_image = load_image(ai_avartar if role == "ai" else human_avatar)

    with st.chat_message(role, avatar=avatar_image):
        st.markdown(message)

    if save:
        save_message(message, role)


def pain_history() -> None:
    """
    세션 상태에 저장된 모든 대화 내역을 화면에 표시하는 함수.

    세션 상태의 "messages" 리스트에서 각 메시지를 가져와
    채팅 인터페이스에 순서대로 표시.
    """
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


@st.cache_data(show_spinner=False)
def read_file_data(file: Any) -> pd.DataFrame:
    """
    파일 데이터를 읽어 DataFrame으로 변환하는 함수.

    Args:
        file (Any): 업로드된 파일 객체. 보통 Streamlit의 file_uploader로 받은 파일.

    Returns:
        pd.DataFrame: CSV 데이터를 담은 pandas DataFrame.
    """
    df = pd.read_csv(file)
    return df


def check_korean(text):
    """
    RAG, Agent 설정 이름에 한글 포함 여부를 체크하는 함수.

    Args:
        text (str): 입력한 설정 이름.

    Returns:
        bool: 입력한 이름에 한글이 포함되면 True를 반환하고 한글이 포함되어있지 않으면 False를 반환

    """
    p = re.compile("[ㄱ-힣]")
    r = p.search(text)
    if r is None:
        return False
    else:
        return True


class SimpleFAQSystem:
    """
    A simple FAQ system that uses string similarity to match user queries
    to predefined questions and return predefined answers.
    """
    
    def __init__(self, faq_data: List[Dict[str, str]], similarity_threshold: float = 0.8):
        """
        Initialize the FAQ system.
        
        Args:
            faq_data: List of dictionaries with 'question' and 'answer' keys
            similarity_threshold: Minimum similarity score to trigger FAQ response (0.0-1.0)
        """
        self.faq_data = faq_data
        self.similarity_threshold = similarity_threshold
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation except question marks and Korean characters
        # Keep Korean characters, alphanumeric, spaces, and question marks
        text = re.sub(r'[^\w\s?가-힣]', '', text)
        return text.strip()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _find_best_match(self, user_query: str) -> Tuple[Optional[str], float]:
        """
        Find the best matching FAQ question.
        
        Returns:
            Tuple of (answer, similarity_score) or (None, 0.0) if no match
        """
        normalized_query = self._normalize_text(user_query)
        best_match = None
        best_score = 0.0
        
        for i, faq in enumerate(self.faq_data):
            normalized_question = self._normalize_text(faq['question'])
            similarity = self._calculate_similarity(normalized_query, normalized_question)
            
            if similarity > best_score:
                best_score = similarity
                best_match = faq['answer']
        
        return best_match, best_score
    
    def get_response(self, user_query: str) -> Optional[str]:
        """
        Get FAQ response if query matches any predefined question.
        
        Args:
            user_query: The user's input query
            
        Returns:
            Predefined answer if match found, None otherwise
        """
        answer, similarity = self._find_best_match(user_query)
        
        if similarity >= self.similarity_threshold:
            return answer
        
        return None


# Hardcoded FAQ data - customize these for your demo
FAQ_DATA = [
    {
        "question": "RFP에서 요구하는 회사 일반 현황을 다른 제안서을 참고해서 구체적인 작성 가이드를 줘",
        "answer": """
💡 RFP에서는 '회사 일반 현황'에 "주요 연혁 및 사업 내용, 조직 및 인력 구성, 주요 사업 분야, 재무 현황"을 제시하도록 요구하고 있습니다.
RAG DB에서 찾은 유사한 제안서에는 '회사소개' 파트에서 회사 개요, 글로벌 네트워크 등을 강조하고 있습니다.
이를 바탕으로, 삼성화재 RFP 용 회사 일반 현황 작성 가이드를 드리겠습니다.

- 1. 주요 연혁 및 사업 내용: "최근 3개년 주요 프로젝트 실적" 슬라이드 형식을 활용하되, 단순 연혁 나열보다는 AI/디지털 전환 관련 성과를 강조
- 2. 조직 및 인력 구성: 회사 전체 조직도와 관련된 전담 조직을 소개. 제안서의 "국내 최다 1,200명 컨설턴트, 석/박사급 49명 보유" 등 숫자를 강조한 슬라이드를 활용 및 전문 인력의 스킬셋과 관련 프로젝트 경험을 나열
- 3. 주요 사업 분야: 회사의 핵심 사업 영역(예: AI 전략 컨설팅, 데이터 플랫폼 구축, 금융권 디지털 혁신 등) 및 금융·보험 분야의 경험을 별도로 강조 (삼성화재와의 연관성 부각) 제안서의 "산업별 전문가 그룹 운영" 부분을 참조

혹시 원하시면 제가 이 내용을 **샘플 목차 + 슬라이드 구조(예: 1장=연혁, 2장=조직, 3장=사업분야, 4장=재무)** 형태로 설계해드릴까요?
"""
    },
    {
        "question": "우리가 현재 가지고 있는 자료 중에 또 유사한 사례가 기존에 있었을까?",
        "answer": """
💡 LLM을 이용하여 내부 문서를 참조하는 챗봇을 구성했다는 점에서, 삼성전자 **루비콘 제안서**를 참고하시기를 추천드립니다.

- 삼성화재의 RFP는 임직원 업무지원 챗봇(가칭 화재GPT) 상세 설계 및 To-Be 모델 정의를 요구
- 내부 통제 자동화(채무구조도, 광고심의 자동화 등) 단계별 로드맵 및 서비스 설계
- 루비콘 제안서는 생성형 AI 플랫폼 구축 및 대고객 상담봇 운영까지의 일정을 제안
- LLM 챗봇 및 내부 문서 참조라는 공통점을 가지고 있음

또한 금융 업계의 보안 가이드라인을 제시하고 있는 KRX 제안서의 8~11pg도 참고하시기 바랍니다.
"""
    },
    {
        "question": "여기 인력 리스트를 바탕으로, 어떤 사람이 이 프로젝트에 적합할지 제안해 줘. 참고로 Senior Associate 2명, Associate 1명이 필요해.",
        "answer": """
📌 **프로젝트 투입 인력 추천 및 사유**

1. 박정수 Senior Associate
    - 본 프로젝트의 핵심 과업인 **AI 중장기 로드맵 및 서비스 적용 전략 수립**과 직접적으로 부합
    - 금융·보험 업계 컨설팅 경험이 풍부하여, **삼성화재의 조직 특성과 시장 규제 환경을 고려한 전략 정립 가능**


2. 이민우 Senior Associate
    - RFP에서 강조된 **내부통제 자동화 및 규제 대응 로드맵 수립**에 최적의 전문성 보유
    - 금융 규제 변화 타임라인 모니터링 및 대응 전략을 제안할 수 있어, **리스크 최소화 및 실행 가능성 제고에 기여**


3. 박지수 Associate
    - 본 프로젝트에서 요구하는 **임직원 업무지원 챗봇(화재GPT) 설계·구축 방안**과 직접적으로 연계
    - 데이터 전처리, 시스템 인터페이스 연계 등 **기술적 구현 가능성을 검증할 핵심 인력**

종합적으로, 이 세 명은 각각 전략 - 규제/통제 - 기술의 축을 담당하여 아래 RFP에서 요구한 내용을 유기적으로 커버할 수 있습니다.

- AI 중장기 전략 수립
- 내부통제 자동화 및 규제 대응
- 챗봇/생성형 AI 서비스 적용
    
이 외 추가로 필요한 인력이 있으시면 말씀해주세요.
"""
    }
]

# Initialize FAQ system with hardcoded parameters
FAQ_SYSTEM = SimpleFAQSystem(FAQ_DATA, similarity_threshold=0.7)  # Balanced threshold for good matching
