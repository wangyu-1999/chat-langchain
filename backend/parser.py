import re
from typing import Generator

from bs4 import BeautifulSoup, Doctype, NavigableString, Tag


def langchain_docs_extractor(html: str) -> str:
    """从HTML提取文本内容
    Args:
        html: HTML字符串
    Returns:
        str: 提取的文本内容
    """
    soup = BeautifulSoup(html, 'lxml')
    
    # 需要移除的无关标签
    REMOVE_TAGS = [
        "script", "style", "noscript", "iframe", "svg", "img", 
        "audio", "video", "canvas", "form", "input", "button",
        "select", "textarea", "nav", "footer", "aside", "header",
        "object", "embed"
    ]
    
    # 移除所有无关标签
    [tag.decompose() for tag in soup.find_all(REMOVE_TAGS)]
    
    # 移除所有不需要的属性
    for tag in soup.find_all(True):
        attrs_to_remove = [
            attr for attr in tag.attrs if attr in [
                'style', 'src', 'alt', 'title', 'role', 'tabindex'
            ] or attr.startswith(('aria-', 'on', 'data-'))
        ]
        for attr in attrs_to_remove:
            del tag.attrs[attr]

    def get_text(tag) -> str:
        for child in tag.children:
            if isinstance(child, Doctype):
                continue

            if isinstance(child, NavigableString):
                yield child
            elif isinstance(child, Tag):
                if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    yield f"{'#' * int(child.name[1:])} {child.get_text()}\n\n"
                elif child.name == "p":
                    yield from get_text(child)
                    yield "\n\n"
                elif child.name == "ul":
                    for li in child.find_all("li", recursive=False):
                        yield "- "
                        yield from get_text(li)
                        yield "\n"
                elif child.name == "ol":
                    for i, li in enumerate(child.find_all("li", recursive=False)):
                        yield f"{i + 1}. "
                        yield from get_text(li)
                        yield "\n"
                else:
                    yield from get_text(child)

    joined = "".join(get_text(soup))
    return re.sub(r"\n\n+", "\n\n", joined).strip()
