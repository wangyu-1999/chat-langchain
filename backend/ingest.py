"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
import re
from dotenv import load_dotenv

import weaviate
from config import WEAVIATE_DOCS_INDEX_NAME
from langchain.indexes import SQLRecordManager, index
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from embeddings import get_embeddings_model
from langchain.schema import Document
from html_cleaner import strip_html_tags
from summarizer import ChatModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# 将WEAVIATE_URL定义为全局常量
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
chat_model = ChatModel()


def load_api_news():
    """Load news from BBC RSS feed API endpoint"""
    try:
        # 使用全局WEAVIATE_URL
        client = weaviate.Client(url=WEAVIATE_URL)

        response = requests.get(
            "https://march42-rsshub.hf.space/bbc?format=json",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        data = response.json()

        news_docs = []
        for news in data.get("items", []):
            url = news.get("url", "")
            if not url.startswith("https://www.bbc.com/news/articles/"):
                continue

            # 检查 URL 是否已存在
            query = (
                client.query.get(WEAVIATE_DOCS_INDEX_NAME, ["source"])
                .with_where(
                    {"path": ["source"], "operator": "Equal", "valueString": url}
                )
                .with_limit(1)
                .do()
            )

            # 如果已存在该URL，跳过处理
            if query["data"]["Get"][WEAVIATE_DOCS_INDEX_NAME]:
                continue

            content = strip_html_tags(news.get("content_html", "").strip())
            date = news.get("date_published")

            if content:
                res = chat_model.chat(content)

                if res.get("english_summary", "") != "":
                    doc = Document(
                        page_content=res.get("english_summary", ""),
                        metadata={
                            "source": url,
                            "date": date,
                            "title_cn": res.get("title_cn", ""),
                            "title_en": res.get("title_en", ""),
                            "subject": res.get("subject", ""),
                            "location": res.get("location", ""),
                            "chinese_summary": res.get("summary", ""),
                        },
                    )
                    news_docs.append(doc)
                else:
                    logger.warning(f"Chat模型返回的english_summary为空: {url}")
            else:
                logger.warning(f"新闻 {news.get('title')} 内容为空")

        logger.info(f"已加载 {len(news_docs)} 条新闻")
        return news_docs

    except Exception as e:
        logger.error(f"从BBC RSS获取新闻失败: {e}")
        return []


def ingest_docs():
    # 删除重复的WEAVIATE_URL定义，直接使用全局常量
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
        attributes=[
            "source",
            "date",
            "title_cn",
            "title_en",
            "subject",
            "location",
            "chinese_summary",
        ],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()
    docs_from_news = load_api_news()
    logger.info(f"Loaded {len(docs_from_news)} docs from News API")

    docs_transformed = text_splitter.split_documents(docs_from_news)
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

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
    logger.info(f"LangChain now has this many vectors: {num_vecs}")


if __name__ == "__main__":
    ingest_docs()
