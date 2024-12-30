"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
import re
from dotenv import load_dotenv
import aiohttp

import weaviate
from config import WEAVIATE_DOCS_INDEX_NAME
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from embeddings import get_embeddings_model
from langchain.schema import Document
from html_cleaner import strip_html_tags
from summarizer import ChatModel
from create_schema import create_schema_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# 将WEAVIATE_URL定义为全局常量
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
chat_model = ChatModel()


async def load_api_news():
    """Load news from BBC RSS feed API endpoint"""
    try:
        # 使用全局WEAVIATE_URL
        client = weaviate.Client(url=WEAVIATE_URL)

        # 使用 aiohttp 替代 requests 进行异步请求
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://march42-rsshub.hf.space/bbc?format=json",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            ) as response:
                data = await response.json()

        stats = {
            "skipped_urls": [],
            "processed_urls": [],
            "empty_content_urls": [],
            "failed_summary_urls": [],
        }

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

            if query["data"]["Get"][WEAVIATE_DOCS_INDEX_NAME]:
                stats["skipped_urls"].append(url)
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
                    stats["processed_urls"].append(url)
                else:
                    stats["failed_summary_urls"].append(url)
            else:
                stats["empty_content_urls"].append(url)

        logger.info(f"已加载 {len(news_docs)} 条新闻")
        return news_docs, stats

    except Exception as e:
        logger.error(f"从BBC RSS获取新闻失败: {e}")
        return [], {}


async def ingest_docs():
    """处理文档摄入的异步函数"""
    try:
        # 确保schema存在
        create_schema_if_not_exists()

        # 在子进程中初始化模型
        embedding = get_embeddings_model()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200
        )

        client = weaviate.Client(url=WEAVIATE_URL)
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

        docs_from_news, load_stats = await load_api_news()
        logger.info(f"Loaded {len(docs_from_news)} docs from News API")

        docs_transformed = text_splitter.split_documents(docs_from_news)
        docs_transformed = [
            doc for doc in docs_transformed if len(doc.page_content) > 10
        ]

        await vectorstore.aadd_documents(docs_transformed)

        return {
            "load_stats": {
                "total_skipped": len(load_stats["skipped_urls"]),
                "total_processed": len(load_stats["processed_urls"]),
                "total_empty": len(load_stats["empty_content_urls"]),
                "total_failed_summary": len(load_stats["failed_summary_urls"]),
                "details": load_stats,
            },
            "index_stats": {"num_added": len(docs_transformed)},
        }

    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        return {
            "error": str(e),
            "load_stats": load_stats if "load_stats" in locals() else None,
        }


if __name__ == "__main__":
    ingest_docs()
