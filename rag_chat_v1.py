## 터미널 채팅
## python rag_chat_v1.py

# rag_chat_v1.py
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import textwrap

# ----------------------------
# 환경변수 로드
# ----------------------------
load_dotenv()
DB_PATH = os.getenv("EMBEDDING_DB_PATH", "./data_collection_db")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_MODEL = os.getenv("LM_MODEL", "openai/gpt-oss-20b")

# 멀티스레드 및 토크나이저 경고 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ----------------------------
# 임베딩 & 벡터 DB 로드
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2})

# ----------------------------
# LLM (LM Studio 연결)
# ----------------------------
llm = ChatOpenAI(
    model=LM_MODEL,
    openai_api_key="not-needed",
    openai_api_base=LM_STUDIO_URL,
    temperature=0.3,  # 약간의 창의성 추가
)

# ----------------------------
# 긴 문서 요약 함수
# ----------------------------
def summarize_text(text: str, max_length: int = 2000):
    """문서가 너무 길면 자동으로 요약"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if len(text) <= max_length:
        return text
    chunks = textwrap.wrap(text, max_length)
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  └ 문서 요약 중 ({i+1}/{len(chunks)}) ...")
        prompt = (
            f"다음 텍스트를 한국어로 간결하게 요약하세요. 중요 정보는 유지해주세요.\n\n{chunk}"
        )
        summary = llm.invoke(prompt).content
        summaries.append(summary)
    return "\n".join(summaries)

# ----------------------------
# RAG용 프롬프트 (자연스러운 대화형)
# ----------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "당신은 친절하고 지적인 한국어 AI 어시스턴트입니다.\n"
        "아래의 문서를 참고하여 사용자의 질문에 정확하고 자연스럽게 대답하세요.\n"
        "만약 문서에 답이 없으면, 일반적인 지식이나 상식 선에서 답변하세요.\n\n"
        "질문: {question}\n\n"
        "관련 문서 내용:\n{context}\n\n"
        "🧠 답변:"
    ),
)

# ----------------------------
# RAG 체인 (자동 요약 + Fallback 포함)
# ----------------------------
def rag_answer(query: str):
    # 문서 검색
    docs = retriever.invoke(query)
    print(f"📄 검색된 문서 수: {len(docs)}")
    if len(docs) == 0:
        print("⚠️ 관련 문서 없음 → 일반 답변 모드로 전환")
        response = llm.invoke(f"질문: {query}\n자연스럽게 답변해주세요.")
        return response.content

    # 문서 길이 출력 및 요약 처리
    summarized_docs = []
    for i, doc in enumerate(docs):
        print(f"  ├ 문서 {i+1} 길이: {len(doc.page_content)}자")
        if len(doc.page_content) > 1500:
            print(f"  └ 문서 {i+1} 요약 중 ...")
            doc.page_content = summarize_text(doc.page_content)
        summarized_docs.append(doc.page_content)

    # 문맥 결합
    context = "\n\n".join(summarized_docs)

    # 최종 질의 구성
    final_prompt = RAG_PROMPT.format(question=query, context=context)

    # LLM 호출
    try:
        response = llm.invoke(final_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ LLM 호출 오류 발생: {e}")
        # Fallback: 문맥 없이 일반 답변
        response = llm.invoke(f"질문: {query}\n간단히 답변해주세요.")
        return response.content.strip()

# ----------------------------
# 터미널 채팅 루프
# ----------------------------
def main():
    print("💬 RAG Chat 시작")
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

            answer = rag_answer(query)
            print(f"\n🧠 답변: {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 채팅을 종료합니다 (Ctrl+C).")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

# ----------------------------
if __name__ == "__main__":
    main()
