import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import gc
from tqdm import tqdm

# .env 로드
load_dotenv()
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

# ----------------------------
# GitHub API에서 파일 URL 가져오기
# ----------------------------
def get_repo_files(repo_full_name, path="", branch="main"):
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}?ref={branch}"
    res = requests.get(url, headers=HEADERS)
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
# 파일 다운로드
# ----------------------------
def download_file(file_url):
    try:
        res = requests.get(file_url, headers=HEADERS, timeout=30)
        res.raise_for_status()
        return res.text
    except Exception as e:
        return None

# ----------------------------
# 배치 저장
# ----------------------------
def save_batch_to_vectorstore(docs, db, prefix="github", start_idx=0):
    ids = [f"{prefix}-{i+start_idx}" for i in range(len(docs))]
    db.add_texts(docs, ids=ids)
    docs.clear()
    gc.collect()

# ----------------------------
# 메인 크롤링 + 인덱싱
# ----------------------------
def crawl_and_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="./data_collection_db", embedding_function=embeddings)

    global_idx = 0
    batch_docs = []

    for repo, branch in REPOS.items():
        print(f"\n🔹 처리 중: {repo} (브랜치: {branch})")
        try:
            file_urls = get_repo_files(repo, branch=branch)
            total_files = len(file_urls)
            print(f"총 {total_files}개의 파일 발견")

            # tqdm 진행률 바 + ETA 자동 표시
            with tqdm(total=total_files, desc=f"{repo}", unit="file", ncols=100) as pbar:
                with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                    futures = {executor.submit(download_file, url): url for url in file_urls}

                    for future in as_completed(futures):
                        content = future.result()
                        pbar.update(1)  # 진행률 바 업데이트

                        if content:
                            batch_docs.append(content)

                        if len(batch_docs) >= BATCH_SIZE:
                            save_batch_to_vectorstore(batch_docs, db, start_idx=global_idx)
                            global_idx += BATCH_SIZE
                            pbar.set_postfix_str(f"배치 저장 완료 ({global_idx}문서)")

                # 남은 문서 처리
                if batch_docs:
                    save_batch_to_vectorstore(batch_docs, db, start_idx=global_idx)
                    global_idx += len(batch_docs)
                    print(f"💾 마지막 배치 저장 완료 (총 {global_idx}문서)")

        except Exception as e:
            print(f"❌ 레포 접근 실패: {repo} ({e})")

    print("\n✅ GitHub 레포 인덱싱 완료")

# ----------------------------
if __name__ == "__main__":
    crawl_and_index()
