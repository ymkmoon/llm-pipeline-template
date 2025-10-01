## 터미널 채팅
## python rag_chat.py

# rag_chat.py
import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

# ----------------------------
# 환경변수 로드
# ----------------------------
load_dotenv()
DB_PATH = os.getenv("EMBEDDING_DB_PATH", "./data_collection_db")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_MODEL = os.getenv("LM_MODEL", "openai/gpt-oss-20b")

# 멀티스레드 관련 설정 (Segmentation fault 예방)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ----------------------------
# 임베딩 & 벡터 DB 로드
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})  # k값 기본 3, 필요시 조정

# ----------------------------
# LLM 연결 (LM Studio)
# ----------------------------
llm = ChatOpenAI(
    model=LM_MODEL,
    openai_api_key="not-needed",       # LM Studio는 API key 불필요
    openai_api_base=LM_STUDIO_URL,
    temperature=0.0
)

# ----------------------------
# RAG QA 체인 생성
# ----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# ----------------------------
# 안전한 터미널 채팅 루프
# ----------------------------
def main():
    print("💬 RAG Chat (환경변수 기반) 시작")
    print("종료하려면 'exit', 'quit', '종료' 입력")

    while True:
        try:
            query = input("\n질문: ").strip()
            if query.lower() in ["exit", "quit", "종료"]:
                print("👋 채팅을 종료합니다.")
                break

            if not query:
                print("⚠️ 질문을 입력해주세요.")
                continue

            result = qa.invoke({"query": query})
            print("답변:", result.get("result", "답변 없음"))

        except KeyboardInterrupt:
            print("\n👋 채팅을 종료합니다 (Ctrl+C).")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

# ----------------------------
if __name__ == "__main__":
    main()