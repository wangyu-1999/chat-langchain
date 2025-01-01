"""新闻处理模块"""

import aiohttp
from typing import Dict, List
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.text_cleaner import strip_html_tags

logger = logging.getLogger("backend")

# 新闻源配置
NEWS_SOURCES = [
    {
        "name": "BBC",
        "url": "https://march42-rsshub.hf.space/bbc?format=json",
        "article_pattern": "https://www.bbc.com/news/articles/",
    },
    {
        "name": "AP News",
        "url": "https://march42-rsshub.hf.space/apnews/api/apf-topnews?format=json",
        "article_pattern": "https://apnews.com/article/",
    },
    {
        "name": "CNBC",
        "url": "https://march42-rsshub.hf.space/cnbc/rss?format=json",
        "article_pattern": "https://www.cnbc.com/",
    },
    {
        "name": "Tass",
        "url": "https://march42-rsshub.hf.space/tass/world?format=json",
        "article_pattern": "https://tass.com/world/",
    },
    {
        "name": "Sputnik News",
        "url": "https://march42-rsshub.hf.space/sputniknews/world?format=json",
        "article_pattern": "https://sputniknews.com/",
    },
    {
        "name": "Economist",
        "url": "https://march42-rsshub.hf.space/economist/latest?format=json",
        "article_pattern": "https://www.economist.com/",
    },
    {
        "name": "Straits Times",
        "url": "https://march42-rsshub.hf.space/straitstimes/world?format=json",
        "article_pattern": "https://www.straitstimes.com/world/",
    },
    {
        "name": "Huanqiu",
        "url": "https://march42-rsshub.hf.space/huanqiu/news/world?format=json",
        "article_pattern": "https://world.huanqiu.com/article/",
    },
    {
        "name": "Zaobao",
        "url": "https://march42-rsshub.hf.space/zaobao/znews/world?format=json",
        "article_pattern": "https://www.zaobao.com/news/world/",
    },
]


class NewsProcessor:
    def __init__(self, source_config: Dict):
        self.source_name = source_config["name"]
        self.base_url = source_config["url"]
        self.article_pattern = source_config["article_pattern"]

    async def fetch_raw_news(self) -> List[Dict]:
        """获取原始新闻数据"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    data = await response.json()

            news_items = []
            for news in data.get("items", []):
                if news.get("url", "").startswith(self.article_pattern):
                    content = strip_html_tags(news.get("content_html", "").strip())
                    if content:
                        news_items.append(
                            {
                                "content": content,
                                "date": news.get("date_published"),
                                "source": news.get("url"),
                                "source_name": self.source_name,
                            }
                        )

            return news_items

        except Exception as e:
            logger.error(f"从{self.source_name} RSS获取新闻失败: {e}")
            return []
