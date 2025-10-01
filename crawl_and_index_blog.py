## 크롤링 => 임베딩 DB 저장
## python crawl_and_index_blog.py

# crawl_and_index_blog.py
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gc
from tqdm import tqdm

# ====== 블로그 URL ======
BASE_URL = "https://ymkmoon.github.io"
TAG_URL = f"{BASE_URL}/tags/"

# ====== 설정 ======
BATCH_SIZE = 10
MAX_THREADS = 5

# ====== 1. 블로그 글 목록 가져오기 ======
def get_blog_links(tag_url: str):
    res = requests.get(tag_url, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    links = []

    for a in soup.select("a"):
        href = a.get("href")
        if href and href.startswith("/"):
            if "/tags/" not in href:
                full_url = BASE_URL + href
                if full_url not in links:
                    links.append(full_url)
    return links

# ====== 2. 개별 블로그 글 크롤링 ======
def get_blog_content(url: str):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "제목 없음"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        content = "\n".join(paragraphs)
        return f"{title_text}\n{content}"
    except Exception as e:
        print(f"❌ 크롤링 실패: {url} ({e})")
        return None

# ====== 3. 벡터DB 저장 ======
def save_batch_to_vectorstore(docs, db, start_idx=0):
    ids = [f"blog-{i+start_idx}" for i in range(len(docs))]
    db.add_texts(docs, ids=ids)
    docs.clear()
    gc.collect()

# ====== 4. 메인 실행 ======
def main():
    print(f"🔹 블로그 글 목록 가져오는 중: {TAG_URL}")
    links = get_blog_links(TAG_URL)
    total_links = len(links)
    print(f"총 {total_links}개의 블로그 글 발견\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="./data_collection_db", embedding_function=embeddings)

    docs = []
    global_idx = 0

    # tqdm 진행률 바 적용
    with tqdm(total=total_links, desc="블로그 크롤링", unit="글", ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(get_blog_content, link): link for link in links}

            for future in as_completed(futures):
                content = future.result()
                pbar.update(1)  # 진행률 업데이트

                if content:
                    docs.append(content)

                if len(docs) >= BATCH_SIZE:
                    save_batch_to_vectorstore(docs, db, start_idx=global_idx)
                    global_idx += BATCH_SIZE
                    pbar.set_postfix_str(f"배치 저장 완료 ({global_idx}문서)")

        # 남은 글 저장
        if docs:
            save_batch_to_vectorstore(docs, db, start_idx=global_idx)
            global_idx += len(docs)
            print(f"💾 마지막 배치 저장 완료 (총 {global_idx}문서)")

    print("\n✅ 블로그 임베딩 DB 저장 완료")

# ====== 실행 ======
if __name__ == "__main__":
    main()
