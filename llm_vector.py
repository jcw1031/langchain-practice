import os
import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# Step 2: 텍스트를 청크로 분할
def get_web_content(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    view_con = [tag.text.strip() for tag in soup.select(".view-con")]
    return view_con[0]


# Step 3: 청크를 벡터로 임베딩하고 FAISS에 저장
def build_faiss_index(chunks: list[str], embedding_model) -> tuple[faiss.IndexFlatL2, list[str]]:
    embeddings = embedding_model.embed_documents(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks


# Step 4: 질문에 대해 관련 청크 검색
def retrieve_chunks(query: str, index: faiss.IndexFlatL2, chunks: list[str], embedding_model, k: int = 3):
    query_embedding = embedding_model.embed_query(query)
    _, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]


# Step 5: GPT 모델을 통해 답변 생성
def generate_answer(query: str, context_chunks: list[str], model_name: str = "gpt-4.1-nano"):
    context = "\n\n".join(context_chunks)
    prompt = f"""
다음은 웹페이지에서 추출한 정보입니다:

{context}

사용자 질문: "{query}"

해당 정보를 바탕으로 정리된 답변을 제공하세요.
"""
    llm = ChatOpenAI(model=model_name)
    response = llm([HumanMessage(content=prompt)])
    return response.content


if __name__ == "__main__":
    url = "https://www.kongju.ac.kr"
    query = "교내 장학금 신청 방법이 어떻게 되나요?"

    print("[1] 웹 페이지 수집 중...")
    html_text = get_web_content(url)

    print("[2] 텍스트 분할 중...")
    chunks = chunk_text(html_text)

    print("[3] 임베딩 및 벡터 인덱스 생성 중...")
    embeddings = OpenAIEmbeddings()
    index, chunk_list = build_faiss_index(chunks, embeddings)

    print("[4] 관련 문서 검색 중...")
    related_chunks = retrieve_chunks(query, index, chunk_list, embeddings)

    print("[5] LLM 응답 생성 중...\n")
    answer = generate_answer(query, related_chunks)
    print("📘 답변:\n", answer)
