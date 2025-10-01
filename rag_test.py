## LLM 파이프라인 테스트 코드
## python rag_test.py

# rag_test.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ====== .env 로드 ======
load_dotenv()

# ====== 환경 변수 ======
DB_PATH = os.getenv("EMBEDDING_DB_PATH")

# 1) LLM (LM Studio API 서버 연결)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",    # ✅ LM Studio에서 실행 중인 모델 이름
    openai_api_key="not-needed",   # LM Studio는 API key 불필요
    openai_api_base="http://localhost:1234/v1"
)

# 2) 벡터DB 로드 (👉 MiniLM으로 통일)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever()

# 3) QA 체인 생성
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 4) 테스트 질문
query = "Java Stream API를 어떻게 활용할 수 있어?"
print("Q:", query)
# run() 대신 invoke() 권장
result = qa.invoke({"query": query})
print("A:", result["result"])
