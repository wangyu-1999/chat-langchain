from azure.data.tables import TableServiceClient, UpdateMode
from datetime import datetime
import os
from typing import Dict, Optional
import base64
import azure.core.exceptions
import logging

logger = logging.getLogger("backend")


class AzureTableStorage:
    def __init__(self):
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        # 保存为实例变量，这样其他方法可以访问
        self.table_service_client = TableServiceClient.from_connection_string(
            connection_string
        )
        self.table_client = self.table_service_client.get_table_client("newsContent")

        # 确保表存在
        try:
            self.table_service_client.create_table("newsContent")
        except:
            pass

    def _encode_key(self, key: str) -> str:
        """将 URL 编码为安全的 key"""
        return base64.b64encode(key.encode()).decode()

    def _decode_key(self, encoded_key: str) -> str:
        """将编码的 key 解码回 URL"""
        return base64.b64decode(encoded_key.encode()).decode()

    def store_document(self, source: str, content: Dict) -> None:
        """同步方式存储文档到 Table Storage"""
        # 使用域名的 base64 编码作为 PartitionKey
        domain = source.split("/")[2]  # 获取域名部分
        entity = {
            "PartitionKey": self._encode_key(domain),
            "RowKey": self._encode_key(source),  # 使用完整 URL 的 base64 编码
            "title_cn": content.get("title_cn", ""),
            "title_en": content.get("title_en", ""),
            "subject": content.get("subject", ""),
            "location": content.get("location", ""),
            "chinese_summary": content.get("chinese_summary", ""),
            "english_summary": content.get("english_summary", ""),
            "source_name": content.get("source_name", ""),
            "date": content.get("date", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "original_url": source,  # 保存原始 URL 以便查询
        }

        # 使用同步方式更新实体
        self.table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    def get_document_sync(self, source: str) -> Optional[Dict]:
        """同步方式获取文档内容"""
        try:
            domain = source.split("/")[2]
            partition_key = self._encode_key(domain)
            row_key = self._encode_key(source)

            entity = self.table_client.get_entity(partition_key, row_key)
            return {
                "title_cn": entity.get("title_cn"),
                "title_en": entity.get("title_en"),
                "subject": entity.get("subject"),
                "location": entity.get("location"),
                "chinese_summary": entity.get("chinese_summary"),
                "english_summary": entity.get("english_summary"),
                "source_name": entity.get("source_name"),
                "date": entity.get("date"),
            }
        except Exception as e:
            logger.error(f"获取文档失败: {str(e)}")
            return None

    def document_exists(self, url: str) -> bool:
        try:
            domain = url.split("/")[2]  # 获取域名部分
            self.table_client.get_entity(
                partition_key=self._encode_key(domain), row_key=self._encode_key(url)
            )
            return True
        except azure.core.exceptions.ResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"检查文档存在时发生错误: {str(e)}")
            return False

    def store_clusters(self, entity):
        """
        将所有聚类结果作为一个整体存储到Azure Table

        Args:
            entity: 包含所有聚类信息的字典，必须包含PartitionKey和RowKey
        """
        try:
            table_client = self.table_service_client.get_table_client("clusters")

            # 确保表存在
            try:
                self.table_service_client.create_table("clusters")
            except Exception as e:
                logger.warning(f"创建clusters表时出现警告: {str(e)}")

            # 存储实体
            table_client.upsert_entity(entity)
            logger.info(f"已保存聚类结果: {entity['RowKey']}")

        except Exception as e:
            logger.error(f"存储聚类结果时出错: {str(e)}")
            raise
