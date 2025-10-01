## Github 블로그 인덱싱
## python crawl_and_index_blog.py

# crawl_and_index_blog.py
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gc
from tqdm import tqdm
import backoff
from dotenv import load_dotenv

# ====== .env 로드 ======
load_dotenv()

# ====== 환경 변수 ======
DB_PATH = os.getenv("EMBEDDING_DB_PATH")

# ====== 블로그 URL ======
BASE_URL = "https://ymkmoon.github.io"
TAG_URL = f"{BASE_URL}/tags/"

# ====== 설정 ======
BATCH_SIZE = 10
MAX_THREADS = 5
CHUNK_SIZE = 1000  # 글자 단위 chunk

# ----------------------------
# 블로그 글 목록 가져오기
# ----------------------------
def get_blog_links(tag_url: str):
    res = requests.get(tag_url, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    links = []
    for a in soup.select("a"):
        href = a.get("href")
        if href and href.startswith("/") and "/tags/" not in href:
            full_url = BASE_URL + href
            if full_url not in links:
                links.append(full_url)
    return links

# ----------------------------
# 글 내용 크롤링 + 재시도
# ----------------------------
@backoff.on_exception(backoff.expo, requests.RequestException, max_tries=3)
def get_blog_content(url: str):
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else "제목 없음"
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    content = "\n".join(paragraphs)
    return f"{title_text}\n{content}"

# ----------------------------
# 배치 + chunk 저장
# ----------------------------
def save_batch_to_vectorstore(docs, db, start_idx=0):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)

    ids = [f"blog-{i+start_idx}" for i in range(len(all_chunks))]
    db.add_texts(all_chunks, ids=ids)
    docs.clear()
    gc.collect()
    return len(all_chunks)

# ----------------------------
# 메인
# ----------------------------
def main():
    print(f"🔹 블로그 글 목록 가져오는 중: {TAG_URL}")
    links = get_blog_links(TAG_URL)
    total_links = len(links)
    print(f"총 {total_links}개의 블로그 글 발견\n")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    batch_docs = []
    global_idx = 0

    with tqdm(total=total_links, desc="블로그 크롤링", unit="글", ncols=100, dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(get_blog_content, link): link for link in links}
            for future in as_completed(futures):
                content = future.result()
                pbar.update(1)
                if content:
                    batch_docs.append(content)
                if len(batch_docs) >= BATCH_SIZE:
                    added = save_batch_to_vectorstore(batch_docs, db, start_idx=global_idx)
                    global_idx += added
                    pbar.set_postfix_str(f"배치 저장 완료 ({global_idx}문서)")

        # 남은 글 저장
        if batch_docs:
            added = save_batch_to_vectorstore(batch_docs, db, start_idx=global_idx)
            global_idx += added
            print(f"💾 마지막 배치 저장 완료 (총 {global_idx}문서)")

    print("\n✅ 블로그 임베딩 DB 저장 완료")

if __name__ == "__main__":
    main()
