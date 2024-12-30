import weaviate
import os
from dotenv import load_dotenv
from config import WEAVIATE_DOCS_INDEX_NAME
import logging

logger = logging.getLogger(__name__)

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")


def create_schema_if_not_exists(client):
    try:
        # 检查schema是否已存在
        schema = client.schema.get()
        existing_classes = (
            [c["class"] for c in schema["classes"]] if schema.get("classes") else []
        )

        if WEAVIATE_DOCS_INDEX_NAME not in existing_classes:
            # 定义schema
            class_obj = {
                "class": WEAVIATE_DOCS_INDEX_NAME,
                "description": "News documents collection",
                "properties": [
                    {
                        "name": "page_content",
                        "dataType": ["text"],
                        "description": "The main content of the document",
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "Source URL of the document",
                    },
                    {
                        "name": "source_name",
                        "dataType": ["string"],
                        "description": "Name of the news source",
                    },
                    {
                        "name": "date",
                        "dataType": ["string"],
                        "description": "Publication date",
                    },
                    {
                        "name": "title_cn",
                        "dataType": ["string"],
                        "description": "Chinese title",
                        "moduleConfig": {"text2vec-openai": {"skip": True}},
                    },
                    {
                        "name": "title_en",
                        "dataType": ["string"],
                        "description": "English title",
                    },
                    {
                        "name": "subject",
                        "dataType": ["string"],
                        "description": "News subject in English",
                    },
                    {
                        "name": "location",
                        "dataType": ["string"],
                        "description": "Related location in English",
                    },
                    {
                        "name": "chinese_summary",  # 对应config中的summary
                        "dataType": ["text"],
                        "description": "Summary in Chinese",
                        "moduleConfig": {"text2vec-openai": {"skip": True}},
                    },
                ],
            }

            client.schema.create_class(class_obj)
            logger.info(f"已创建schema: {WEAVIATE_DOCS_INDEX_NAME}")
        else:
            logger.info(f"Schema {WEAVIATE_DOCS_INDEX_NAME} 已存在")

    except Exception as e:
        logger.error(f"创建schema时出错: {str(e)}")
        raise
