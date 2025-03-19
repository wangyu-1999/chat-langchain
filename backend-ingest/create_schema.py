from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from dotenv import load_dotenv
import os
from config import DOCS_INDEX_NAME
import logging

logger = logging.getLogger("backend")

load_dotenv()


def connect_db():
    """连接到 Zilliz Cloud"""
    uri = os.getenv("ZILLIZ_CLOUD_URI")
    token = os.getenv("ZILLIZ_CLOUD_TOKEN")

    logger.info(f"Connecting to DB: {uri}")
    connections.connect(alias="default", uri=uri, token=token)
    logger.info("Success!")


def create_schema_if_not_exists():
    try:
        connect_db()

        # 定义 collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="publish_time", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]

        schema = CollectionSchema(
            fields=fields, description="News documents collection"
        )

        # 检查 collection 是否存在
        if not utility.has_collection(DOCS_INDEX_NAME):
            collection = Collection(name=DOCS_INDEX_NAME, schema=schema)

            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index(field_name="embedding", index_params=index_params)

            logger.info(f"已创建collection和索引: {DOCS_INDEX_NAME}")
        else:
            logger.warning(f"Collection {DOCS_INDEX_NAME} 已存在")

    except Exception as e:
        logger.error(f"创建collection时出错: {str(e)}")
        raise
    finally:
        connections.disconnect("default")
