import weaviate
import os
from dotenv import load_dotenv
from config import WEAVIATE_DOCS_INDEX_NAME
import logging

logger = logging.getLogger(__name__)

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")


def create_schema_if_not_exists():
    client = weaviate.Client(url=WEAVIATE_URL)

    # 检查schema是否已存在
    try:
        schema = client.schema.get(WEAVIATE_DOCS_INDEX_NAME)
        if schema:
            logger.info(f"Schema '{WEAVIATE_DOCS_INDEX_NAME}' already exists")
            return False
    except Exception:
        # schema不存在，创建新的
        schema = {
            "classes": [
                {
                    "class": WEAVIATE_DOCS_INDEX_NAME,
                    "description": "A collection of news articles with their summaries and metadata",
                    "vectorizer": "none",  # 使用外部向量化器(BGE-M3)
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                            "description": "The main content (English summary) of the article",
                        },
                        {
                            "name": "source",
                            "dataType": ["string"],
                            "description": "The source URL of the article",
                        },
                        {
                            "name": "date",
                            "dataType": ["string"],
                            "description": "Publication date of the article",
                        },
                        {
                            "name": "title_cn",
                            "dataType": ["string"],
                            "description": "Chinese title of the article",
                        },
                        {
                            "name": "title_en",
                            "dataType": ["string"],
                            "description": "English title of the article",
                        },
                        {
                            "name": "subject",
                            "dataType": ["string"],
                            "description": "Main subject or entity of the article",
                        },
                        {
                            "name": "location",
                            "dataType": ["string"],
                            "description": "Location mentioned in the article",
                        },
                        {
                            "name": "chinese_summary",
                            "dataType": ["text"],
                            "description": "Chinese summary of the article",
                        },
                    ],
                }
            ]
        }

        # 创建schema
        client.schema.create(schema)
        logger.info(
            f"Schema for class '{WEAVIATE_DOCS_INDEX_NAME}' created successfully!"
        )
        return True
