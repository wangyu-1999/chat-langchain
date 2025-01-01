"""文本清理工具模块"""

from bs4 import BeautifulSoup


def strip_html_tags(html_text):
    if not html_text:
        return ""
    if not isinstance(html_text, str):
        html_text = str(html_text)
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(strip=True)
