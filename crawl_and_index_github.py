## Github 저장소 인덱싱
## python crawl_and_index_github.py

# crawl_and_index_github.py
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import gc
from tqdm import tqdm
import backoff

# .env 로드
load_dotenv()

# ====== 환경 변수 ======
DB_PATH = os.getenv("EMBEDDING_DB_PATH")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("⚠️ 환경변수 GITHUB_TOKEN이 설정되어 있지 않습니다.")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# 레포 & 브랜치 설정
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

BATCH_SIZE = 20
MAX_THREADS = 5
CHUNK_SIZE = 1000  # 글자 단위 chunk

# ----------------------------
# GitHub API에서 파일 URL 가져오기
# ----------------------------
def get_repo_files(repo_full_name, path="", branch="main"):
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}?ref={branch}"
    res = requests.get(url, headers=HEADERS, timeout=15)
    res.raise_for_status()
    files = []

    for item in res.json():
        if item["type"] == "file":
            files.append(item["download_url"])
        elif item["type"] == "dir":
            new_path = f"{path}/{item['name']}".strip("/")
            files.extend(get_repo_files(repo_full_name, path=new_path, branch=branch))
    return files

# ----------------------------
# 파일 다운로드 + 재시도
# ----------------------------
@backoff.on_exception(backoff.expo, requests.RequestException, max_tries=3)
def download_file(file_url):
    res = requests.get(file_url, headers=HEADERS, timeout=30)
    res.raise_for_status()
    return res.text

# ----------------------------
# 배치 + chunk 저장
# ----------------------------
def save_batch_to_vectorstore(docs, db, prefix="github", start_idx=0):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=50
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)

    ids = [f"{prefix}-{i+start_idx}" for i in range(len(all_chunks))]
    db.add_texts(all_chunks, ids=ids)
    docs.clear()
    gc.collect()
    return len(all_chunks)

# ----------------------------
# 메인 크롤링 + 인덱싱
# ----------------------------
def crawl_and_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    global_idx = 0

    for repo, branch in REPOS.items():
        print(f"\n🔹 처리 중: {repo} (브랜치: {branch})")
        try:
            file_urls = get_repo_files(repo, branch=branch)
            total_files = len(file_urls)
            print(f"총 {total_files}개의 파일 발견")

            batch_docs = []
            with tqdm(total=total_files, desc=f"{repo}", unit="file",
                      ncols=100, dynamic_ncols=True, unit_scale=True) as pbar:
                with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                    futures = {executor.submit(download_file, url): url for url in file_urls}
                    for future in as_completed(futures):
                        content = future.result()
                        pbar.update(1)
                        if content:
                            batch_docs.append(content)

                        if len(batch_docs) >= BATCH_SIZE:
                            added = save_batch_to_vectorstore(batch_docs, db, start_idx=global_idx)
                            global_idx += added
                            pbar.set_postfix_str(f"배치 저장 완료 ({global_idx}문서)")

                # 남은 문서 처리
                if batch_docs:
                    added = save_batch_to_vectorstore(batch_docs, db, start_idx=global_idx)
                    global_idx += added
                    print(f"💾 마지막 배치 저장 완료 (총 {global_idx}문서)")

        except Exception as e:
            print(f"❌ 레포 접근 실패: {repo} ({e})")

    print("\n✅ GitHub 레포 임베딩 DB 저장 완료")

# ----------------------------
if __name__ == "__main__":
    crawl_and_index()
