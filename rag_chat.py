# rag_chat.py
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ====== .env 로드 ======
load_dotenv()

# ====== 환경 변수 ======
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")
LM_MODEL = os.getenv("LM_MODEL")
DB_PATH = os.getenv("EMBEDDING_DB_PATH")

# 토크나이저 병렬 처리 경고 제거
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====== 1) LLM 연결 ======
llm = ChatOpenAI(
    model=LM_MODEL,
    openai_api_key="not-needed",
    openai_api_base=LM_STUDIO_URL
)

# ====== 2) 벡터DB 로드 및 chunk 분리 ======
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# retriever: 한 번에 가져올 문서 수 k=3으로 제한
retriever = db.as_retriever(search_kwargs={"k": 3})

# ====== 3) QA 체인 생성 ======
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# ====== 4) helper: 입력 길이 제한 및 안전 처리 ======
def safe_query(input_text: str, max_len: int = 5000):
    # 입력 텍스트가 너무 길면 자르기
    if len(input_text) > max_len:
        return input_text[:max_len]
    return input_text

# ====== 5) 터미널 채팅 루프 ======
print("💬 RAG Chat 개선판 시작 (종료하려면 'exit' 또는 'quit' 입력)")

while True:
    query = input("\n질문: ").strip()
    if query.lower() in ["exit", "quit", "종료"]:
        print("👋 채팅을 종료합니다.")
        break

    # 입력 안전 처리
    query = safe_query(query)

    try:
        result = qa.invoke({"query": query})
        print(f"\n답변:\n{result['result']}\n")
    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
        print("💡 힌트: 질문이 너무 길거나, DB에서 가져온 문서가 많을 수 있습니다.")
