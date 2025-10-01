## 크롤링 => 임베딩 DB 저장
## python crawl_and_index_blog.py

# crawl_and_index_blog.py
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ====== 블로그 URL ======
BASE_URL = "https://ymkmoon.github.io"

# ====== 1. 블로그 글 목록 가져오기 ======
def get_blog_links(tag_url: str):
    res = requests.get(tag_url)
    soup = BeautifulSoup(res.text, "html.parser")
    links = []

    for a in soup.select("a"):
        href = a.get("href")
        if href and href.startswith("/"):  # 내부 링크만
            if "/tags/" not in href:       # 태그 페이지는 제외
                full_url = BASE_URL + href
                if full_url not in links:
                    links.append(full_url)
    return links

# ====== 2. 개별 블로그 글 크롤링 ======
def get_blog_content(url: str):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else "제목 없음"
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    content = "\n".join(paragraphs)
    return f"{title_text}\n{content}"

# ====== 3. 벡터DB 저장 ======
def save_to_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="./data_collection_db", embedding_function=embeddings)
    
    for i, doc in enumerate(docs):
        db.add_texts([doc], ids=[f"blog-{i}"])
    print("✅ 블로그 글 인덱싱 완료")

# ====== 실행 ======
if __name__ == "__main__":
    tag_url = "https://ymkmoon.github.io/tags/"
    links = get_blog_links(tag_url)
    print(f"총 {len(links)}개의 블로그 글 발견")

    docs = []
    for link in links:
        try:
            content = get_blog_content(link)
            docs.append(content)
            print(f"크롤링 완료: {link}")
        except Exception as e:
            print(f"❌ 크롤링 실패: {link} ({e})")

    if docs:
        save_to_vectorstore(docs)
