# rag_chat.py
import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# 토크나이저 병렬 처리 경고 제거
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1) LLM (LM Studio API 서버 연결)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",    # LM Studio 실행 중인 모델 이름
    openai_api_key="not-needed",   # LM Studio는 API key 불필요
    openai_api_base="http://localhost:1234/v1"
)

# 2) 벡터DB 로드 (MiniLM으로 통일)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory="./blog_db", embedding_function=embeddings)
retriever = db.as_retriever()

# 3) QA 체인 생성
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 4) 터미널 채팅 루프
print("💬 RAG Chat 시작 (종료하려면 'exit' 또는 'quit' 입력)")
while True:
    query = input("\n질문: ")
    if query.lower() in ["exit", "quit", "종료"]:
        print("👋 채팅을 종료합니다.")
        break
    
    try:
        result = qa.invoke({"query": query})
        print("답변:", result["result"])
    except Exception as e:
        print("⚠️ 오류 발생:", e)
