"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Sequence

import weaviate
from config import WEAVIATE_DOCS_INDEX_NAME
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from embeddings import get_embeddings_model
from langchain.schema import Document
from summarizer import ChatModel
from create_schema import create_schema_if_not_exists
from news_sources.bbc_processor import BBCNewsProcessor
from storage.azure_table import AzureTableStorage
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.retrievers import BaseRetriever
from retriever_chain import create_retriever_chain

logger = logging.getLogger("backend")

load_dotenv()

# 将WEAVIATE_URL定义为全局常量
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

# 创建一个线程池执行器
thread_pool = ThreadPoolExecutor(max_workers=3)

chat_model = ChatModel()


def process_news_item(content: str, metadata: Dict, chat_model: ChatModel) -> Document:
    """处理单个新闻内容"""
    res = chat_model.chat(content)

    if isinstance(res, dict) and res.get("english_summary"):
        # 创建完整的元数据字典
        full_metadata = {
            "source": metadata["source"],
            "source_name": metadata["source_name"],
            "date": metadata["date"],
            "title_cn": res.get("title_cn", ""),
            "title_en": res.get("title_en", ""),
            "subject": res.get("subject", ""),
            "location": res.get("location", ""),
            "chinese_summary": res.get("chinese_summary", ""),
        }

        return Document(
            page_content=res["english_summary"],
            metadata=full_metadata,  # 保存完整元数据
        )
    return None


async def check_existing_url(client: weaviate.Client, url: str) -> bool:
    """
    检查URL是否已存在于Weaviate数据库中

    Args:
        client: Weaviate客户端实例
        url: 要检查的URL

    Returns:
        bool: 如果URL已存在返回True，否则返回False
    """
    try:
        query = {
            "class": WEAVIATE_DOCS_INDEX_NAME,
            "where": {"path": ["source"], "operator": "Equal", "valueString": url},
        }

        result = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: client.query.get(WEAVIATE_DOCS_INDEX_NAME, ["source"])
            .with_where(query["where"])
            .do(),
        )

        # 检查是否有匹配的记录
        entries = (
            result.get("data", {}).get("Get", {}).get(WEAVIATE_DOCS_INDEX_NAME, [])
        )
        return len(entries) > 0

    except Exception as e:
        logger.error(f"检查URL时出错: {str(e)}")
        return False


async def ingest_docs():
    """处理文档摄入的异步函数"""
    try:
        # 初始化客户端和模型
        client = await asyncio.get_event_loop().run_in_executor(
            thread_pool, lambda: weaviate.Client(url=WEAVIATE_URL)
        )

        # 添加这一行来创建 schema
        await asyncio.get_event_loop().run_in_executor(
            thread_pool, lambda: create_schema_if_not_exists(client)
        )

        embedding = await asyncio.get_event_loop().run_in_executor(
            thread_pool, get_embeddings_model
        )

        # 初始化新闻处理器
        news_processors = [
            BBCNewsProcessor(),
        ]

        # 统计信息
        stats_by_source = {}
        all_raw_items = []

        # 首先收集所有新闻源的新闻
        for processor in news_processors:
            source_name = processor.source_name
            raw_news_items = await processor.fetch_raw_news()
            all_raw_items.extend(raw_news_items)

            # 初始化统计信息
            stats_by_source[source_name] = {
                "skipped": 0,
                "processed": 0,
                "failed_summary": 0,
                "details": {
                    "skipped_urls": [],
                    "processed_urls": [],
                    "failed_summary_urls": [],
                },
            }

        # 批量检查所有URL
        all_urls = [item["source"] for item in all_raw_items]
        existing_urls = set()

        # 使用批量查询获取已存在的URL
        query = {
            "class": WEAVIATE_DOCS_INDEX_NAME,
            "where": {
                "operator": "Or",
                "operands": [
                    {"path": ["source"], "operator": "Equal", "valueString": url}
                    for url in all_urls
                ],
            },
        }

        result = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: client.query.get(WEAVIATE_DOCS_INDEX_NAME, ["source"])
            .with_where(query["where"])
            .do(),
        )

        if result and "data" in result:
            entries = result["data"]["Get"][WEAVIATE_DOCS_INDEX_NAME]
            existing_urls = {entry["source"] for entry in entries}

        # 处理新闻内容
        all_docs = []
        azure_storage = AzureTableStorage()

        for item in all_raw_items:
            source_name = item["source_name"]

            if item["source"] in existing_urls:
                stats_by_source[source_name]["skipped"] += 1
                stats_by_source[source_name]["details"]["skipped_urls"].append(
                    item["source"]
                )
                continue

            # 处理内容
            doc = process_news_item(
                item["content"],
                {
                    "source": item["source"],
                    "source_name": item["source_name"],
                    "date": item["date"],
                },
                chat_model,
            )

            if doc:
                # 使用同步方式存储到 Azure Table
                azure_storage.store_document(
                    item["source"],
                    {
                        "title_cn": doc.metadata["title_cn"],
                        "title_en": doc.metadata["title_en"],
                        "subject": doc.metadata["subject"],
                        "location": doc.metadata["location"],
                        "chinese_summary": doc.metadata["chinese_summary"],
                        "english_summary": doc.page_content,
                        "source_name": doc.metadata["source_name"],
                        "date": doc.metadata["date"],
                    },
                )

                # 只保存 source 和用于生成向量的文本到 Weaviate
                weaviate_doc = Document(
                    page_content=doc.page_content,
                    metadata={"source": doc.metadata["source"]},
                )
                all_docs.append(weaviate_doc)
                stats_by_source[source_name]["processed"] += 1
                stats_by_source[source_name]["details"]["processed_urls"].append(
                    item["source"]
                )
            else:
                stats_by_source[source_name]["failed_summary"] += 1
                stats_by_source[source_name]["details"]["failed_summary_urls"].append(
                    item["source"]
                )

        # 将文本分割移到线程池
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_docs_future = asyncio.get_event_loop().run_in_executor(
            thread_pool, text_splitter.split_documents, all_docs
        )
        docs_transformed = await split_docs_future

        # 将 Weaviate 添加文档操作移到线程池
        vectorstore = Weaviate(
            client=client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            by_text=False,
            attributes=["source"],
        )

        # 批量处理文档，每批50个
        batch_size = 50
        for i in range(0, len(docs_transformed), batch_size):
            batch = docs_transformed[i : i + batch_size]
            # 确保每个文档只包含必要的信息
            for doc in batch:
                # 使用 english_summary (page_content) 作为向量生成的文本
                doc.metadata = {"source": doc.metadata["source"]}
                # page_content 已经是 english_summary，不需要修改

            add_docs_future = asyncio.get_event_loop().run_in_executor(
                thread_pool, partial(vectorstore.add_documents, batch)
            )
            await add_docs_future

        # 计算总体统计信息
        total_stats = {
            "skipped": sum(s["skipped"] for s in stats_by_source.values()),
            "processed": sum(s["processed"] for s in stats_by_source.values()),
            "failed_summary": sum(
                s["failed_summary"] for s in stats_by_source.values()
            ),
        }

        return {
            "load_stats": {"total": total_stats, "by_source": stats_by_source},
            "index_stats": {"num_added": len(docs_transformed)},
        }

    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # 修改主入口，确保正确运行异步函数
    asyncio.run(ingest_docs())
