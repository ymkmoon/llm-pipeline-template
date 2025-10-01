# crawl_and_index_safe.py
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import gc
from tqdm import tqdm

# ----------------------------
# 환경변수 로드
# ----------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DB_PATH = os.getenv("EMBEDDING_DB_PATH", "./data_collection_db")

if not GITHUB_TOKEN:
    raise ValueError("⚠️ 환경변수 GITHUB_TOKEN이 설정되어 있지 않습니다.")

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# ----------------------------
# 설정
# ----------------------------
BATCH_SIZE = 10
MAX_THREADS = 5
CHUNK_SIZE = 1000  # 단어 기준 chunk 분리
BASE_BLOG_URL = "https://ymkmoon.github.io"
TAG_URL = f"{BASE_BLOG_URL}/tags/"

# GitHub 레포 & 브랜치
REPOS = {
    "ymkmoon/llm-pipeline-template": "main",
    "ymkmoon/mqtt-broker-template": "main",
    "ymkmoon/springboot-consumer-template": "develop",
    "ymkmoon/kafka-broker-template": "main",
    "ymkmoon/springboot-jpa-template": "develop",
    "ymkmoon/toyseven": "develop-jdk-17",
    "ymkmoon/slack-bot-notifiaction-template": "develop",
    "ymkmoon/react-web-template": "develop",
    "ymkmoon/springboot-admin-template": "develop",
    "ymkmoon/sprintboot-canvas-template": "develop",
    "ymkmoon/toyseven-react": "develop",
    "ymkmoon/cs-study": "main"
}

# ----------------------------
# 헬퍼: 문서 chunk 분리
# ----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# ----------------------------
# 블로그 크롤링
# ----------------------------
def get_blog_links(tag_url: str):
    try:
        res = requests.get(tag_url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        links = []
        for a in soup.select("a"):
            href = a.get("href")
            if href and href.startswith("/") and "/tags/" not in href:
                full_url = BASE_BLOG_URL + href
                if full_url not in links:
                    links.append(full_url)
        return links
    except Exception as e:
        print(f"❌ 블로그 링크 가져오기 실패: {e}")
        return []

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
        print(f"❌ 블로그 크롤링 실패: {url} ({e})")
        return None

# ----------------------------
# GitHub 파일 가져오기
# ----------------------------
def get_repo_files(repo_full_name, path="", branch="main"):
    try:
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}?ref={branch}"
        res = requests.get(url, headers=HEADERS, timeout=30)
        res.raise_for_status()
        files = []
        for item in res.json():
            if item["type"] == "file":
                files.append(item["download_url"])
            elif item["type"] == "dir":
                new_path = f"{path}/{item['name']}".strip("/")
                files.extend(get_repo_files(repo_full_name, path=new_path, branch=branch))
        return files
    except Exception as e:
        print(f"❌ 레포 파일 가져오기 실패: {repo_full_name} ({e})")
        return []

def download_file(file_url):
    try:
        res = requests.get(file_url, headers=HEADERS, timeout=30)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"❌ 파일 다운로드 실패: {file_url} ({e})")
        return None

# ----------------------------
# 벡터 DB 저장
# ----------------------------
def save_batch_to_vectorstore(docs, db, prefix="doc", start_idx=0):
    ids = [f"{prefix}-{i+start_idx}" for i in range(len(docs))]
    db.add_texts(docs, ids=ids)
    docs.clear()
    gc.collect()

# ----------------------------
# 메인 실행
# ----------------------------
def main():
    # PyTorch/Transformers 안전 환경변수
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    # 임베딩/DB 생성 (메인 스레드)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    global_idx = 0

    # ===== 블로그 =====
    print("🔹 블로그 글 인덱싱 시작")
    links = get_blog_links(TAG_URL)
    batch_docs = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(get_blog_content, link): link for link in links}
        with tqdm(total=len(links), desc="블로그", unit="글", ncols=100) as pbar:
            for future in as_completed(futures):
                content = future.result()
                pbar.update(1)
                if content:
                    for chunk in chunk_text(content):
                        batch_docs.append(chunk)
                if len(batch_docs) >= BATCH_SIZE:
                    save_batch_to_vectorstore(batch_docs, db, prefix="blog", start_idx=global_idx)
                    global_idx += BATCH_SIZE
                    pbar.set_postfix_str(f"배치 저장 완료 ({global_idx}문서)")
            if batch_docs:
                save_batch_to_vectorstore(batch_docs, db, prefix="blog", start_idx=global_idx)
                global_idx += len(batch_docs)
    print(f"✅ 블로그 완료 (총 {global_idx}문서)")

    # ===== GitHub =====
    print("\n🔹 GitHub 레포 인덱싱 시작")
    batch_docs = []

    for repo, branch in REPOS.items():
        print(f"🔹 처리 중: {repo} (브랜치: {branch})")
        file_urls = get_repo_files(repo, branch=branch)
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(download_file, url): url for url in file_urls}
            with tqdm(total=len(file_urls), desc=repo, unit="파일", ncols=100) as pbar:
                for future in as_completed(futures):
                    content = future.result()
                    pbar.update(1)
                    if content:
                        for chunk in chunk_text(content):
                            batch_docs.append(chunk)
                    if len(batch_docs) >= BATCH_SIZE:
                        save_batch_to_vectorstore(batch_docs, db, prefix="github", start_idx=global_idx)
                        global_idx += BATCH_SIZE
                        pbar.set_postfix_str(f"배치 저장 완료 ({global_idx}문서)")
            if batch_docs:
                save_batch_to_vectorstore(batch_docs, db, prefix="github", start_idx=global_idx)
                global_idx += len(batch_docs)

    print(f"\n✅ GitHub 완료 (총 {global_idx}문서)")

# ----------------------------
if __name__ == "__main__":
    main()
