from typing import List, Union

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_llm(
    model: str = "gpt-4o",
    temperature: float = 0.2,
    streaming: bool = False,
    callbacks: List = [],
    base_url: str = "10.99.15.72:11434",
    num_predict: int = 500,
) -> Union[ChatOpenAI, ChatAnthropic, ChatOllama]:
    """
    주어진 모델 이름에 따라 적절한 LLM(언어 모델) 객체를 반환하는 함수.

    지원되는 모델:
    - 'gpt'로 시작하는 모델: ChatOpenAI 객체 반환
    - 'claude'로 시작하는 모델: ChatAnthropic 객체 반환
    - 'llama', 'gemma', 'mistral'로 시작하는 모델: ChatOllama 객체 반환

    Args:
        model (str): 사용할 LLM 모델 이름. 기본값은 'gpt-4o'
        temperature (float): 생성된 텍스트의 창의성 수준을 조절하는 값. 기본값은 0.2
        streaming (bool): 스트리밍 모드를 사용할지 여부. 기본값은 False
        callbacks (List): 생성 중 실행할 콜백 함수 목록. 기본값은 빈 리스트
        num_predict (int): Ollama 모델의 출력 토큰 최대 길이. 기본값은 300

    Returns:
        Union[ChatOpenAI, ChatAnthropic, ChatOllama]: 주어진 모델 이름에 해당하는 LLM 객체.

    Raises:
        ValueError: 지원되지 않는 모델 이름이 입력된 경우 발생합니다.
    """
    if model.startswith("gpt"):
        llm = ChatOpenAI(model=model, temperature=temperature, streaming=streaming, callbacks=callbacks)
    elif model.startswith("claude"):
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
        )
    elif model.startswith(("llama", "gemma", "cow/gemma2_tools", "mistral", "EEVE", "qwen")):
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            base_url=base_url,
            num_predict=num_predict,
        )
    else:
        raise ValueError(f"지원되지 않는 모델: {model}")

    return llm


def get_embedding(
    model: str = "text-embedding-3-large",
    base_url: str = "10.99.15.72:11434",
) -> Union[OpenAIEmbeddings, OllamaEmbeddings]:
    """
    주어진 모델 이름에 따라 적절한 Embeddings 객체를 반환하는 함수.

    Args:
        model (str): 사용할 Embeddings 모델의 이름. 기본값은 'text-embedding-3-large'

    Returns:
        Union[OpenAIEmbeddings, OllamaEmbeddings]: 주어진 모델에 해당하는 Embeddings 객체.

    Raises:
        ValueError: 지원되지 않는 모델 이름이 주어졌을 때 발생.
    """
    if model is None:
        raise ValueError("Embedding model cannot be None. Please set SELECTED_EMBEDDING environment variable.")
    elif model.startswith("text-embedding"):
        embeddings = OpenAIEmbeddings(model=model)
    elif model.startswith(("mxbai-embed-large", "nomic-embed-text", "bge-m3", "jeffh/intfloat")):
        embeddings = OllamaEmbeddings(model=model, base_url=base_url)
    else:
        raise ValueError(f"지원되지 않는 모델: {model}")

    return embeddings
