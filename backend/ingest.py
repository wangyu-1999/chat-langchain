"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, ConfigDict

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager
from langchain.indexes import index
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from datetime import datetime
from utils.markdown_saver import MarkdownSaver
import random
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PREFIXES_TO_IGNORE = [
    "https://",
    "http://",
    "mailto:",
    "javascript:",
    "#",
    "/"
]
PREFIXES_TO_IGNORE_REGEX = "|".join(map(re.escape, PREFIXES_TO_IGNORE))

SUFFIXES_TO_IGNORE = [
    ".jpg",
    ".jpeg",
    ".gif",
    ".png",
    ".svg",
    ".pdf",
    ".css",
    ".js",
    ".ico"
]
SUFFIXES_TO_IGNORE_REGEX = "|".join(map(re.escape, SUFFIXES_TO_IGNORE))

def get_embeddings_model() -> Embeddings:
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {
        "normalize_embeddings": True,
        "query_instruction": "" # bge-m3 需要空的 query_instruction
    }
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

def load_api_news():
    """Load news from mock API endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/mock")
        data = response.json()
        
        news_docs = []
        for news in data.get("news_results", []):
            position = news.get("position")
            
            if "highlight" in news:
                main_link = news["highlight"].get("link")
                main_date = news["highlight"].get("date")
                stories = news.get("stories", [])
                
                success = False
                try:
                    news_loader = RecursiveUrlLoader(
                        url=main_link,
                        max_depth=1,
                        extractor=simple_extractor,
                        prevent_outside=True,
                        use_async=True,
                        timeout=20,
                        check_response_status=True,
                        headers=get_headers()
                    )
                    
                    docs = news_loader.load()
                    
                    # 检查是否成功获取到内容
                    if docs and len(docs) > 0 and docs[0].page_content.strip():
                        # 成功获取页面，添加元数据
                        for doc in docs:
                            doc.metadata.update({
                                "source": main_link,
                                "date": main_date,
                                "position": position
                            })
                        news_docs.extend(docs)
                        success = True
                    else:
                        logger.warning(f"从 {main_link} 获取的内容为空，尝试stories中的链接")
                    
                except Exception as e:
                    logger.warning(f"访问 {main_link} 失败（{str(e)}），尝试stories中的链接")
                
                # 如果highlight链接失败或内容为空，依次尝试stories中的链接
                if not success and stories:
                    for story in stories:
                        story_link = story.get("link")
                        story_date = story.get("date")
                        
                        try:
                            news_loader = RecursiveUrlLoader(
                                url=story_link,
                                max_depth=1,
                                extractor=simple_extractor,
                                prevent_outside=True,
                                use_async=True,
                                timeout=20,
                                check_response_status=True,
                                headers=get_headers()
                            )
                            
                            docs = news_loader.load()
                            
                            # 检查是否成功获取到内容
                            if docs and len(docs) > 0 and docs[0].page_content.strip():
                                # 使用stories的链接成功，但保持原有position
                                for doc in docs:
                                    doc.metadata.update({
                                        "source": story_link,
                                        "date": story_date,
                                        "position": position
                                    })
                                news_docs.extend(docs)
                                success = True
                                break
                            else:
                                logger.warning(f"从 {story_link} 获取的内容为空")
                                continue
                            
                        except Exception as e:
                            logger.warning(f"访问 {story_link} 失败（{str(e)}）")
                            continue
                    
                    if not success:
                        logger.warning(f"访问 {main_link} 及其所有替代链接均失败，没有更多代替")
            
            else:
                # 没有highlight的情况，使用普通的link和date
                link = news.get("link")
                date = news.get("date")
                
                if link:
                    try:
                        news_loader = RecursiveUrlLoader(
                            url=link,
                            max_depth=1,
                            extractor=simple_extractor,
                            prevent_outside=True,
                            use_async=True,
                            timeout=20,
                            check_response_status=True,
                            headers=get_headers()
                        )
                        
                        docs = news_loader.load()
                        
                        for doc in docs:
                            doc.metadata.update({
                                "source": link,
                                "date": date,
                                "position": position
                            })
                        news_docs.extend(docs)
                        
                    except Exception as e:
                        logger.error(f"Error loading news from {link}: {e}")
                        continue
        
        logger.info(f"Loaded {len(news_docs)} news documents")
        
        # 使用 MarkdownSaver 保存文档
        if news_docs:
            markdown_saver = MarkdownSaver()
            saved_count = markdown_saver.save_docs(news_docs)
            logger.info(f"Saved {saved_count} documents to markdown files")
            
        return news_docs
        
    except Exception as e:
        logger.error(f"Error loading news from mock API: {e}")
        return []


def ingest_docs():
    WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    client = weaviate.Client(
        url=WEAVIATE_URL,
    )
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")
    docs_from_news = load_api_news()
    logger.info(f"Loaded {len(docs_from_news)} docs from News API")


    return

    docs_transformed = text_splitter.split_documents(
        docs_from_api + docs_from_langsmith + docs_from_news
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )


if __name__ == "__main__":
    ingest_docs()
