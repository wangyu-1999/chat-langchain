"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from config import DOCS_INDEX_NAME
from embeddings import get_embeddings_model
from summarizer import ChatModel
from create_schema import create_schema_if_not_exists
from news_sources.news_processor import NewsProcessor, NEWS_SOURCES
from storage.azure_table import AzureTableStorage
from pymilvus import connections, Collection

logger = logging.getLogger("backend")

load_dotenv()

# 创建一个线程池执行器
thread_pool = ThreadPoolExecutor(max_workers=3)

chat_model = ChatModel()


class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


async def process_news_item(
    content: str, metadata: Dict, chat_model: ChatModel
) -> Document:
    """处理单个新闻内容"""
    if not content or content.strip() == "":
        logger.warning(f"跳过空内容的新闻，来源：{metadata.get('source', 'unknown')}")
        return None

    # 直接使用 await 调用异步方法
    res = await chat_model.chat(content)

    if isinstance(res, dict) and res.get("english_summary"):
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
            metadata=full_metadata,
        )
    return None


async def process_batch(items, chat_model, existing_urls):
    """并发处理一批新闻"""
    tasks = []
    for item in items:
        if item["source"] not in existing_urls:
            tasks.append(
                process_news_item(
                    item["content"],
                    {
                        "source": item["source"],
                        "source_name": item["source_name"],
                        "date": item["date"],
                    },
                    chat_model,
                )
            )

    if not tasks:
        return []

    return await asyncio.gather(*tasks)


async def ingest_docs():
    """处理文档摄入的异步函数"""
    try:
        # 确保 collection 存在
        create_schema_if_not_exists()

        # 连接数据库并获取 collection
        connections.connect(
            alias="default",
            uri=os.getenv("ZILLIZ_CLOUD_URI"),
            token=os.getenv("ZILLIZ_CLOUD_TOKEN"),
        )

        collection = Collection(DOCS_INDEX_NAME)
        collection.load()

        embedding = get_embeddings_model()

        # 初始化新闻处理器
        news_processors = [
            NewsProcessor(source_config) for source_config in NEWS_SOURCES
        ]

        # 统计信息
        stats_by_source = {}
        all_raw_items = []

        # 收集所有新闻源的新闻
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

        # 使用 Azure Table 检查已存在的 URL
        azure_storage = AzureTableStorage()
        existing_urls = set()

        # 批量检查 URL 是否存在
        for item in all_raw_items:
            if azure_storage.document_exists(item["source"]):
                existing_urls.add(item["source"])
                source_name = item["source_name"]
                stats_by_source[source_name]["skipped"] += 1
                stats_by_source[source_name]["details"]["skipped_urls"].append(
                    item["source"]
                )

        # 将新闻分批处理，每批 5 个
        batch_size = 5
        for i in range(0, len(all_raw_items), batch_size):
            batch = all_raw_items[i : i + batch_size]
            processed_docs = await process_batch(batch, chat_model, existing_urls)

            # 处理结果
            for i, doc in enumerate(processed_docs):
                if doc:
                    source_name = doc.metadata[
                        "source_name"
                    ]  # 直接从doc中获取source_name
                    # 存储到 Azure Table
                    azure_storage.store_document(
                        doc.metadata["source"],  # 直接从doc中获取source
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

                    # 获取文本的嵌入向量
                    embedding_vector = embedding.embed_query(doc.page_content)

                    # 准备插入数据
                    data = [
                        [doc.metadata["source"]],  # source 字段
                        [doc.metadata["date"]],  # publish_time 字段
                        [embedding_vector],  # vector 字段
                    ]

                    collection.insert(data)

                    stats_by_source[source_name]["processed"] += 1
                    stats_by_source[source_name]["details"]["processed_urls"].append(
                        item["source"]
                    )
                else:
                    stats_by_source[source_name]["failed_summary"] += 1
                    stats_by_source[source_name]["details"][
                        "failed_summary_urls"
                    ].append(item["source"])

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
            "index_stats": {"num_added": total_stats["processed"]},
        }

    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # 修改主入口，确保正确运行异步函数
    asyncio.run(ingest_docs())
