## 터미널 채팅
## 대화 내용을 영구 보관
## python rag_chat_v3.py
import os
import json
import textwrap
import itertools
import threading
import time
from dotenv import load_dotenv
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
SEARCH_KWARGS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ----------------------------
# 닉네임 입력
# ----------------------------
def get_nickname():
    while True:
        nickname = input("👤 닉네임 (5글자) 입력: ").strip()
        if len(nickname) == 5 or nickname == "ymkmoon":
            return nickname
        print("⚠️ 닉네임은 정확히 5글자여야 합니다.")

nickname = get_nickname()
CHAT_HISTORY_PATH = f"./chat_history_{nickname}.json"

# ----------------------------
# 임베딩 + 벡터DB
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": SEARCH_KWARGS})

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
# 대화 기억
# ----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 이전 chat_history 복원
if os.path.exists(CHAT_HISTORY_PATH):
    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            messages = json.load(f)
            for msg in messages:
                if msg['type'] == 'human':
                    memory.chat_memory.add_user_message(msg['content'])
                else:
                    memory.chat_memory.add_ai_message(msg['content'])
        print(f"✅ 이전 대화 {len(messages)}건을 복원했습니다. (닉네임: {nickname})")
    except Exception as e:
        print(f"⚠️ chat_history 복원 실패: {e}")

# ----------------------------
# chat_history 저장
# ----------------------------
def save_chat_history():
    try:
        with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in memory.chat_memory.messages], f, ensure_ascii=False, indent=2)
        print(f"💾 chat_history 저장 완료 ({len(memory.chat_memory.messages)}건)")
    except Exception as e:
        print(f"⚠️ chat_history 저장 실패: {e}")

# ----------------------------
# 문서 요약 + 전체 진행률 표시
# ----------------------------
def summarize_text(text: str, doc_index: int = 0, total_docs: int = 1, max_length: int = 2000):
    """문서가 너무 길면 자동 요약 + 전체 진행률 표시"""
    if len(text) <= max_length:
        return text

    chunks = textwrap.wrap(text, max_length)
    summaries = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        stop_spinner = False

        def spinner():
            for c in itertools.cycle("|/-\\"):
                if stop_spinner:
                    break
                overall_progress = ((doc_index + i / total_chunks) / total_docs) * 100
                print(f"\r  └ 전체 문서 요약 진행: {overall_progress:.1f}% ... {c}", end="", flush=True)
                time.sleep(0.1)

        t = threading.Thread(target=spinner)
        t.start()

        prompt = f"다음 텍스트를 한국어로 간결하게 요약하세요. 핵심 정보는 유지하세요.\n\n{chunk}"
        summary = llm.invoke(prompt).content
        summaries.append(summary)

        stop_spinner = True
        t.join()

    return "\n".join(summaries)

# ----------------------------
# RAG 프롬프트
# ----------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template=(
        "당신은 지적이고 따뜻한 한국어 AI 어시스턴트입니다.\n"
        "다음은 지금까지의 대화 내용입니다:\n"
        "{chat_history}\n\n"
        "아래의 문서를 참고하여 질문에 정확하고 자연스럽게 답변하세요.\n"
        "문서의 작성자는 모두 유명기 입니다.\n"
        "유명기는 주로 ymkmoon 라는 닉네임을 사용합니다.\n"
        "만약 문서에 정보가 부족하면 상식적인 대화로 응답하세요.\n\n"
        "질문: {question}\n\n"
        "관련 문서 내용:\n{context}\n\n"
        "🧠 답변:"
    )
)

# ----------------------------
# 관련성 기반 RAG 체인 최적화
# ----------------------------
RELEVANCE_THRESHOLD = 0.3
MAX_CONTEXT_CHARS = 8000
MAX_CHAT_HISTORY_CHARS = 2000

def rag_with_memory(query: str):
    # 일반 대화 모드
    if not (nickname.startswith("ymkmoon") and query.startswith("ymkmoon")):
        response = llm.invoke(f"질문: {query}\n자연스럽게 대화해줘.")
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        save_chat_history()
        return response.content.strip()

    # ----------------------------
    # 관련 문서 검색 및 요약
    # ----------------------------
    docs = retriever.invoke(query)
    print(f"📄 검색된 문서 수: {len(docs)}")
    context_docs = []

    valid_docs = [doc for doc in docs if len(doc.page_content.strip()) > 0 and getattr(doc, "score", 1.0) >= RELEVANCE_THRESHOLD]
    total_docs = len(valid_docs)

    for idx, doc in enumerate(valid_docs):
        if len(doc.page_content) > 1500:
            doc.page_content = summarize_text(doc.page_content, doc_index=idx, total_docs=total_docs)
        context_docs.append(doc.page_content)

    # context 생성 + 길이 제한
    context = "\n\n".join(context_docs)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n... (생략)"

    # ----------------------------
    # 최근 대화 추출 + 길이 제한
    # ----------------------------
    recent_msgs = reversed(memory.chat_memory.messages)
    chat_history = ""
    for m in recent_msgs:
        line = f"사용자: {m.content}" if m.type=="human" else f"AI: {m.content}"
        if len(chat_history) + len(line) + 1 > MAX_CHAT_HISTORY_CHARS:
            break
        chat_history = line + "\n" + chat_history

    # ----------------------------
    # 최종 프롬프트 생성
    # ----------------------------
    final_prompt = RAG_PROMPT.format(
        question=query,
        context=context,
        chat_history=chat_history
    )

    # ----------------------------
    # LLM 호출 + chat_history 저장
    # ----------------------------
    try:
        response = llm.invoke(final_prompt)
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        save_chat_history()
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
        response = llm.invoke(f"질문: {query}\n간단히 답변해줘.")
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        save_chat_history()
        return response.content.strip()

# ----------------------------
# 터미널 실행 루프
# ----------------------------
def main():
    print(f"💬 대화 기억형 RAG Chat 시작 (닉네임: {nickname})")
    print("종료하려면 'exit', 'quit', '종료' 입력")

    while True:
        try:
            query = input("\n질문: ").strip()
            if query.lower() in ["exit", "quit", "종료"]:
                print("👋 채팅을 종료합니다.")
                save_chat_history()
                break
            if not query:
                print("⚠️ 질문을 입력해주세요.")
                continue

            answer = rag_with_memory(query)
            print(f"\n🧠 답변: {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 채팅 종료 (Ctrl+C)")
            save_chat_history()
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

# ----------------------------
if __name__ == "__main__":
    main()
