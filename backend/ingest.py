"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
import re
from dotenv import load_dotenv
import aiohttp
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor

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

# 创建一个线程池执行器
thread_pool = ThreadPoolExecutor(max_workers=3)


async def load_api_news():
    """异步加载新闻数据"""
    try:
        stats = {
            "skipped_urls": [],
            "processed_urls": [],
            "empty_content_urls": [],
            "failed_summary_urls": [],
        }

        async with aiohttp.ClientSession() as session:
            # 异步获取新闻数据
            async with session.get(
                "https://march42-rsshub.hf.space/bbc?format=json"
            ) as response:
                data = await response.json()

            news_docs = []
            # 使用 asyncio.gather 并发处理新闻
            tasks = []
            for news in data.get("items", []):
                if news.get("url", "").startswith("https://www.bbc.com/news/articles/"):
                    tasks.append(process_news_item(news, stats))

            results = await asyncio.gather(*tasks)
            news_docs = [doc for doc in results if doc is not None]

        return news_docs, stats

    except Exception as e:
        logger.error(f"从BBC RSS获取新闻失败: {e}")
        return [], {}


async def process_news_item(news, stats):
    """处理单个新闻项"""
    url = news.get("url", "")

    # 将 Weaviate 查询移到线程池
    client = weaviate.Client(url=WEAVIATE_URL)
    query_future = asyncio.get_event_loop().run_in_executor(
        thread_pool,
        lambda: client.query.get(WEAVIATE_DOCS_INDEX_NAME, ["source"])
        .with_where({"path": ["source"], "operator": "Equal", "valueString": url})
        .with_limit(1)
        .do(),
    )
    query = await query_future

    if query["data"]["Get"][WEAVIATE_DOCS_INDEX_NAME]:
        stats["skipped_urls"].append(url)
        return None

    content = strip_html_tags(news.get("content_html", "").strip())
    date = news.get("date_published")

    if not content:
        stats["empty_content_urls"].append(url)
        return None

    # 使用线程池处理 chat_model 调用
    chat_future = asyncio.get_event_loop().run_in_executor(
        thread_pool, chat_model.chat, content
    )
    res = await chat_future

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
        stats["processed_urls"].append(url)
        return doc
    else:
        stats["failed_summary_urls"].append(url)
        return None


async def ingest_docs():
    """处理文档摄入的异步函数"""
    try:
        # 1. 将同步的 Weaviate 客户端初始化移到线程池
        client_future = asyncio.get_event_loop().run_in_executor(
            thread_pool, lambda: weaviate.Client(url=WEAVIATE_URL)
        )
        client = await client_future

        # 2. 将同步的 embeddings 模型初始化移到线程池
        embedding_future = asyncio.get_event_loop().run_in_executor(
            thread_pool, get_embeddings_model
        )
        embedding = await embedding_future

        # 3. 异步加载新闻
        docs_from_news, load_stats = await load_api_news()

        if not docs_from_news:
            return {
                "load_stats": {
                    "total_skipped": 0,
                    "total_processed": 0,
                    "total_empty": 0,
                    "total_failed_summary": 0,
                }
            }

        # 4. 将文本分割移到线程池
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_docs_future = asyncio.get_event_loop().run_in_executor(
            thread_pool, text_splitter.split_documents, docs_from_news
        )
        docs_transformed = await split_docs_future

        # 5. 将 Weaviate 添加文档操作移到线程池
        vectorstore = Weaviate(
            client=client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            by_text=False,
            attributes=["source", "title_en", "date", "location", "subject"],
        )

        # 批量处理文档，每批50个
        batch_size = 50
        for i in range(0, len(docs_transformed), batch_size):
            batch = docs_transformed[i : i + batch_size]
            add_docs_future = asyncio.get_event_loop().run_in_executor(
                thread_pool, partial(vectorstore.add_documents, batch)
            )
            await add_docs_future

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
        return {"error": str(e)}


if __name__ == "__main__":
    ingest_docs()
