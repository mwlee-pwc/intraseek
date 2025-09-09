from typing import List, Union

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.vectorstores import FAISS, Chroma


def get_vector_store(
    documents: List[Document],
    embedding: Embeddings,
    type: str = "faiss",
) -> Union[FAISS]:
    """
    주어진 Document 객체 목록과 임베딩을 사용하여 지정된 유형의 벡터 스토어를 반환하는 함수.

    Args:
        documents (List[Document]): 임베딩을 생성할 Document 객체 목록.
        embedding (Embeddings): 사용할 임베딩 모델.
        type (str, optional): 생성할 벡터 스토어의 유형. 기본값은 'faiss'.

    Returns:
        Union[FAISS]: 지정된 유형의 벡터 스토어 객체를 반환.

    Raises:
        ValueError: 지원되지 않는 벡터 스토어 유형이 입력된 경우 발생.
    """
    if type.lower().startswith("faiss"):
        vector_store = FAISS.from_documents(documents=documents, embedding=embedding)
    elif type.lower().startswith("chroma"):
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding)
    else:
        raise ValueError(f"Unsupported vector store type: {type}")

    return vector_store


def get_splitter(
    chunk_size: int,
    chunk_overlap: int,
    type: str = "recursive",
) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter]:
    """
    청크 크기, 청크 중첩 값 및 분할기 유형에 따라 적절한 텍스트 분할기를 반환하는 함수.

    Args:
        chunk_size (int): 텍스트를 분할할 때 각 청크의 크기.
        chunk_overlap (int): 청크 간의 중첩 크기.
        type (str, optional): 사용할 분할기의 유형 ("recursive", "character", "token"). 기본값은 "recursive".
        model_name (str, optional): 사용할 모델의 이름. 기본값은 "gpt-4o".

    Returns:
        Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter]: 지정된 유형의 텍스트 분할기를 반환.

    Raises:
        ValueError: 지원되지 않는 분할기 유형을 입력한 경우 발생.
    """
    if type.lower().startswith("recursive"):
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif type.lower().startswith("character"):
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif type.lower().startswith("token"):
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Unsupported splitter type: {type}")

    return splitter
