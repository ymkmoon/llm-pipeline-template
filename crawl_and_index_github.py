## 크롤링 => 임베딩 DB 저장
## python crawl_and_index_github.py

# crawl_and_index_github.py

import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ====== GitHub 설정 ======
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("⚠️ 환경변수 GITHUB_TOKEN이 설정되어 있지 않습니다.")


HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# ====== 인덱싱할 레포 리스트 (repo: branch) ======
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
    # "ymkmoon/toyfive": "main",
    "ymkmoon/cs-study": "main"
}

# ====== 1. 레포 파일 목록 가져오기 ======
def get_repo_files(repo_full_name, path="", branch="main"):
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}?ref={branch}"
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    files = []

    for item in res.json():
        if item["type"] == "file":
            files.append(item["download_url"])
        elif item["type"] == "dir":
            # 하위 디렉토리 재귀 탐색
            new_path = f"{path}/{item['name']}".strip("/")
            files.extend(get_repo_files(repo_full_name, path=new_path, branch=branch))
    return files

# ====== 2. 파일 내용 가져오기 ======
def get_file_content(file_url):
    res = requests.get(file_url, headers=HEADERS)
    res.raise_for_status()
    return res.text

# ====== 3. 벡터DB 저장 ======
def save_to_vectorstore(docs, prefix="github"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="./data_collection_db", embedding_function=embeddings)
    
    for i, doc in enumerate(docs):
        db.add_texts([doc], ids=[f"{prefix}-{i}"])
    print("✅ GitHub 레포 인덱싱 완료")

# ====== 실행 ======
if __name__ == "__main__":
    all_docs = []
    for repo, branch in REPOS.items():
        print(f"🔹 처리 중: {repo} (브랜치: {branch})")
        try:
            file_urls = get_repo_files(repo, branch=branch)
            print(f"총 {len(file_urls)}개의 파일 발견")
            for url in file_urls:
                try:
                    content = get_file_content(url)
                    all_docs.append(content)
                except Exception as e:
                    print(f"❌ 파일 가져오기 실패: {url} ({e})")
        except Exception as e:
            print(f"❌ 레포 접근 실패: {repo} ({e})")

    if all_docs:
        save_to_vectorstore(all_docs)
