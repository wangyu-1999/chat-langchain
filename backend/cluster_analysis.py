import json
import weaviate
import os
from storage.azure_table import AzureTableStorage
from config import WEAVIATE_DOCS_INDEX_NAME
import logging
from typing import List, Dict
from dotenv import load_dotenv
import xmltodict
import requests
from utils.text_cleaner import strip_html_tags
from embeddings import get_embeddings_model
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

load_dotenv()
logger = logging.getLogger(__name__)

# 添加以下日志配置
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def parse_rss_items(rss_dict: dict, source_name: str) -> list:
    """解析不同格式的RSS源"""
    if "rdf:RDF" in rss_dict:
        items = rss_dict["rdf:RDF"]["item"]
    elif "rss" in rss_dict:
        items = rss_dict["rss"]["channel"]["item"]
    else:
        logger.warning(f"无法识别的RSS格式: {source_name}")
        return []

    # 确保items是列表
    return items if isinstance(items, list) else [items]


def parse_news_date(item: dict) -> datetime:
    """解析新闻日期，支持多种格式"""
    date_fields = ["pubDate", "published", "dc:date"]

    for field in date_fields:
        if field in item:
            date_str = item[field]
            try:
                return parsedate_to_datetime(date_str)
            except:
                try:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except:
                    try:
                        return datetime.strptime(date_str, "%A %b %d %Y %H:%M:%S")
                    except:
                        continue
    return None


def get_news_content(item: dict) -> str:
    """获取新闻内容"""
    if "description" in item and item["description"]:
        return strip_html_tags(item["description"])
    elif "title" in item and item["title"]:
        return strip_html_tags(item["title"])
    return None


def get_recent_news_clusters():
    # RSS源列表
    rss_sources = {
        "NYTimes": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "CNN": "http://rss.cnn.com/rss/edition_world.rss",
        "FoxNews": "https://abcnews.go.com/abcnews/internationalheadlines",
        # europe
        "BBC": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "Reuters": "https://rsshub.app/reuters/world",
        "DW": "http://rss.dw-world.de/rdf/rss-en-top",
        "Guardian": "https://www.theguardian.com/world/rss",
        "SkyNews": "https://feeds.skynews.com/feeds/rss/world.xml",
        "TheSun": "https://thesun.my/rss/world",
        # asia
        "Aljazeera": "http://www.aljazeera.com/xml/rss/all.xml",
        "TimesOfIndia": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
        "ChannelNewsAsia": "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6311",
        # other
        "GlobalNews": "https://globalnews.ca/world/feed/",
        "SMH": "https://www.smh.com.au/rss/world.xml",
        "Capi24": "https://feeds.capi24.com/v1/Search/articles/news24/World/rss",
        "IFPNews": "https://ifpnews.com/feed/",
        # Russia
        "MoscowTimes": "https://themoscowtimes.com/feeds/main.xml",
        # China
        "SCMP": "https://www.scmp.com/rss/91/feed",
        "ChinaNews": "https://www.chinanews.com.cn/rss/world.xml",
        "People": "http://www.people.com.cn/rss/world.xml",
    }

    current_time = datetime.now(timezone.utc)
    embeddings_model = get_embeddings_model()
    recent_news = []  # 存储24小时内的新闻

    for source_name, url in rss_sources.items():
        try:
            response = requests.get(url, timeout=10)  # 添加超时设置
            rss_dict = xmltodict.parse(response.content)

            items = parse_rss_items(rss_dict, source_name)

            for item in items:
                content = get_news_content(item)
                if not content:
                    logger.debug(f"跳过：描述和标题都为空: {source_name}")
                    continue

                pub_date = parse_news_date(item)
                if not pub_date:
                    logger.debug(f"跳过：无法解析日期: {source_name}")
                    continue

                pub_date = pub_date.astimezone(timezone.utc)
                if (current_time - pub_date).total_seconds() <= 48 * 3600:
                    recent_news.append(
                        {
                            "source": source_name,
                            "content": content,
                            "pub_date": pub_date,
                            "embedding": embeddings_model.embed_query(content),
                        }
                    )

        except Exception as e:
            logger.error(f"处理新闻源时出错 {source_name}: {str(e)}")
            continue

    # 处理embeddings聚类
    if recent_news:
        embeddings = np.array([news["embedding"] for news in recent_news])
        clustering = DBSCAN(eps=0.4, min_samples=2, metric="cosine").fit(embeddings)

        # 整理聚类结果
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # 排除噪声点
                clusters[label].append(recent_news[idx])

        # 准备JSON输出数据，按聚类大小降序排序
        clusters_output = []
        # 按聚类大小排序
        sorted_clusters = sorted(
            clusters.items(), key=lambda x: len(x[1]), reverse=True
        )

        for i, (cluster_id, news_list) in enumerate(sorted_clusters):
            # 计算聚类中心
            cluster_embeddings = np.array([news["embedding"] for news in news_list])
            cluster_center = np.mean(cluster_embeddings, axis=0)

            clusters_output.append(
                {
                    "id": i,  # 使用新的顺序索引作为ID
                    "size": len(news_list),
                    "center": cluster_center.tolist(),
                }
            )

        output_data = {
            "clusters": clusters_output,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        return output_data


def analyze_news_clusters(weaviate_url: str = "http://localhost:8080"):
    """分析新闻聚类结果,找出每个聚类中心附近的代表性新闻并保存到Azure Table"""
    try:
        # 获取聚类结果
        clusters_data = get_recent_news_clusters()
        if not clusters_data:
            logger.warning("没有找到最近24小时内的新闻聚类")
            return

        # 连接Weaviate
        weaviate_client = weaviate.Client(weaviate_url)
        if not weaviate_client.schema.exists(WEAVIATE_DOCS_INDEX_NAME):
            raise Exception(f"索引 {WEAVIATE_DOCS_INDEX_NAME} 不存在")

        azure_storage = AzureTableStorage()

        # 准备所有聚类中心的向量
        all_centers = []
        for cluster in clusters_data["clusters"]:
            all_centers.append(
                {
                    "id": cluster["id"],
                    "size": cluster["size"],
                    "vector": cluster["center"],
                }
            )

        # 存储所有聚类结果
        all_cluster_results = []
        timestamp = clusters_data["timestamp"]

        logger.info(f"开始处理 {len(all_centers)} 个聚类")

        # 为每个聚类中心分别查询
        for cluster in all_centers:
            logger.info(f"处理聚类 #{cluster['id']}, 大小: {cluster['size']}")

            query_result = (
                weaviate_client.query.get(WEAVIATE_DOCS_INDEX_NAME, ["source"])
                .with_near_vector({"vector": cluster["vector"]})
                .with_additional(["distance", "vector"])
                .with_limit(3)
                .do()
            )

            if "data" in query_result and "Get" in query_result["data"]:
                articles = query_result["data"]["Get"][WEAVIATE_DOCS_INDEX_NAME]

                cluster_result = {"size": cluster["size"], "articles": []}

                for article in articles:
                    similarity = 1 - article["_additional"]["distance"]
                    if similarity > 0.75:
                        cluster_result["articles"].append(
                            {
                                "url": article["source"],
                                "similarity": round(similarity, 3),
                            }
                        )
                        logger.info(
                            f"添加相关新闻: {article['source']} (相似度: {round(similarity, 3)})"
                        )

                if cluster_result["articles"]:
                    all_cluster_results.append(cluster_result)

        # 如果有有效的聚类结果，则保存到Azure Table
        if all_cluster_results:
            logger.info(f"保存 {len(all_cluster_results)} 个有效聚类结果到Azure Table")
            entity = {
                "PartitionKey": timestamp.split("T")[0],
                "RowKey": timestamp,
                "timestamp": timestamp,
                "clusters": json.dumps(all_cluster_results, ensure_ascii=False),
            }

            azure_storage.store_clusters(entity)

    except Exception as e:
        logger.error(f"分析聚类时出错: {str(e)}")
        import traceback

        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    try:
        analyze_news_clusters(weaviate_url)
    except Exception as e:
        print(f"执行失败: {str(e)}")
