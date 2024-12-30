"""BBC新闻处理模块"""

import aiohttp
from typing import Dict, List
import logging
from bs4 import BeautifulSoup
import re


logger = logging.getLogger(__name__)


def strip_html_tags(html_text: str) -> str:
    """
    移除HTML文本中的所有标签,仅保留文本内容

    Args:
        html_text: 包含HTML标签的字符串

    Returns:
        str: 清理后的纯文本内容
    """
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_text, "html.parser")

    # 获取所有文本内容
    text = soup.get_text(separator=" ", strip=True)

    # 清理多余的空白字符
    text = re.sub(r"\s+", " ", text)

    return text.strip()


class BBCNewsProcessor:
    def __init__(self):
        self.source_name = "BBC"
        self.base_url = "https://march42-rsshub.hf.space/bbc?format=json"

    async def fetch_raw_news(self) -> List[Dict]:
        """获取原始新闻数据"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    data = await response.json()

            news_items = []
            for news in data.get("items", []):
                if news.get("url", "").startswith("https://www.bbc.com/news/articles/"):
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
            logger.error(f"从BBC RSS获取新闻失败: {e}")
            return []
