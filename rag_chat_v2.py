## 터미널 채팅
## 파이썬 프로세스 실행 동안 대화 내용이 메모리에 유지 됨
## python rag_chat_v2.py

# rag_chat_v2.py
import os
import textwrap
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# ----------------------------
# 환경설정
# ----------------------------
load_dotenv()
DB_PATH = os.getenv("EMBEDDING_DB_PATH", "./data_collection_db")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_MODEL = os.getenv("LM_MODEL", "openai/gpt-oss-20b")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ----------------------------
# 임베딩 + 벡터DB
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2})

# ----------------------------
# LLM 연결 (LM Studio)
# ----------------------------
llm = ChatOpenAI(
    model=LM_MODEL,
    openai_api_key="not-needed",
    openai_api_base=LM_STUDIO_URL,
    temperature=0.3,
)

# ----------------------------
# 대화 요약 + 기억
# ----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def summarize_text(text: str, max_length: int = 2000):
    """문서가 너무 길면 자동 요약"""
    if len(text) <= max_length:
        return text
    chunks = textwrap.wrap(text, max_length)
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  └ 문서 요약 중 ({i+1}/{len(chunks)}) ...")
        prompt = (
            f"다음 텍스트를 한국어로 간결하게 요약하세요. 핵심 정보는 유지하세요.\n\n{chunk}"
        )
        summary = llm.invoke(prompt).content
        summaries.append(summary)
    return "\n".join(summaries)

# ----------------------------
# 대화형 RAG 프롬프트
# ----------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template=(
        "당신은 지적이고 따뜻한 한국어 AI 어시스턴트입니다.\n"
        "다음은 지금까지의 대화 내용입니다:\n"
        "{chat_history}\n\n"
        "아래의 문서를 참고하여 질문에 정확하고 자연스럽게 답변하세요.\n"
        "만약 문서에 정보가 부족하면 상식적인 대화로 응답하세요.\n\n"
        "문서의 작성자는 모두 유명기 입니다.\n"
        "유명기는 주로 ymkmoon 라는 닉네임을 사용합니다.\n"
        "질문: {question}\n\n"
        "관련 문서 내용:\n{context}\n\n"
        "🧠 답변:"
    )
)

# ----------------------------
# 대화 기억형 RAG 체인
# ----------------------------
def rag_with_memory(query: str):
    # 문서 검색
    docs = retriever.invoke(query)
    print(f"📄 검색된 문서 수: {len(docs)}")

    if len(docs) == 0:
        print("⚠️ 관련 문서 없음 → 일반 대화 모드로 전환")
        response = llm.invoke(f"질문: {query}\n자연스럽게 대화해줘.")
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        return response.content

    # 문서 요약
    summarized_docs = []
    for i, doc in enumerate(docs):
        print(f"  ├ 문서 {i+1} 길이: {len(doc.page_content)}자")
        if len(doc.page_content) > 1500:
            print(f"  └ 문서 {i+1} 요약 중 ...")
            doc.page_content = summarize_text(doc.page_content)
        summarized_docs.append(doc.page_content)

    context = "\n\n".join(summarized_docs)
    chat_history = "\n".join(
        [f"사용자: {m.content}" if m.type == "human" else f"AI: {m.content}" for m in memory.chat_memory.messages]
    )

    final_prompt = RAG_PROMPT.format(question=query, context=context, chat_history=chat_history)

    # 답변 생성
    try:
        response = llm.invoke(final_prompt)
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
        response = llm.invoke(f"질문: {query}\n간단히 답변해줘.")
        return response.content.strip()

# ----------------------------
# 터미널 실행 루프
# ----------------------------
def main():
    print("💬 대화 기억형 RAG Chat 시작")
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

            answer = rag_with_memory(query)
            print(f"\n🧠 답변: {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 채팅을 종료합니다 (Ctrl+C).")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

# ----------------------------
if __name__ == "__main__":
    main()
