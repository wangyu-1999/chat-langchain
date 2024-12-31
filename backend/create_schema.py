import weaviate
import os
from dotenv import load_dotenv
from config import WEAVIATE_DOCS_INDEX_NAME
import logging

logger = logging.getLogger("backend")

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
                        "name": "source",
                        "dataType": ["string"],
                        "description": "Source URL of the document",
                    }
                ],
            }

            client.schema.create_class(class_obj)
            logger.info(f"已创建schema: {WEAVIATE_DOCS_INDEX_NAME}")
        else:
            logger.info(f"Schema {WEAVIATE_DOCS_INDEX_NAME} 已存在")

    except Exception as e:
        logger.error(f"创建schema时出错: {str(e)}")
        raise
