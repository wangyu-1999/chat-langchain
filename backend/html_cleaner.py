from bs4 import BeautifulSoup
import re


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
